import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM


MODEL = "/dev/shm/gu-matrix-tofu/hub/models--open-unlearning--tofu_Llama-3.1-8B-Instruct_full/snapshots/1a5c5b1a557f8c99bdadecd5168ebd03f640b00e"
EXPECTED_UUIDS = {
    0: "GPU-a8adbe48-1936-af09-4bf8-28a955b87f39",
    1: "GPU-f297c801-cb98-80b8-74cc-e7e27b18c809",
}
LARGEST = 525336576
SELECTED = 1179676672
THRESHOLD = 129393.9


def nvml_state(pid, physical_index):
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    gpu_row = None
    for line in gpu.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if int(fields[0]) == physical_index:
            gpu_row = {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "device_used_mib": int(fields[3]),
            }
            break
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    process_used = None
    for line in processes.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if int(fields[0]) == pid:
            process_used = int(fields[2])
            break
    if gpu_row is None or process_used is None:
        raise RuntimeError("NVML did not resolve rank process")
    gpu_row["process_used_mib"] = process_used
    return gpu_row


def tensor_hash(tensor):
    data = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def all_rank_equal(value):
    values = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(values, value)
    return len(set(values)) == 1, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or local_rank not in {0, 1}:
        raise RuntimeError("smoke requires exactly two local ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(0)
    started = time.monotonic()
    result = {"rank": rank, "local_rank": local_rank, "status": "running"}
    try:
        initial = nvml_state(os.getpid(), local_rank)
        if initial["uuid"] != EXPECTED_UUIDS[local_rank]:
            raise RuntimeError("physical GPU UUID mismatch")
        current = AutoModelForCausalLM.from_pretrained(
            MODEL,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        ).to(device)
        current.config.use_cache = False
        current.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        reference = AutoModelForCausalLM.from_pretrained(
            MODEL,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        ).to(device)
        reference.config.use_cache = False
        reference.eval()
        reference.requires_grad_(False)
        ddp = DistributedDataParallel(
            current,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        moments_1 = [torch.empty_like(p, dtype=torch.float32) for p in current.parameters()]
        moments_2 = [torch.empty_like(p, dtype=torch.float32) for p in current.parameters()]
        torch.cuda.synchronize()
        static = nvml_state(os.getpid(), local_rank)
        torch.cuda.reset_peak_memory_stats(device)

        input_ids = torch.arange(512, device=device).remainder(current.config.vocab_size)[None, :]
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, :256] = -100
        with torch.no_grad():
            ref_logits = reference(input_ids=input_ids, attention_mask=attention_mask).logits
            ref_logits_cpu = ref_logits.detach().to("cpu")
        del ref_logits
        output = ddp(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, :-1, :]
        shifted = labels[:, 1:]
        valid = shifted.ne(-100)
        selected_logits = logits[valid]
        selected_labels = shifted[valid]
        targets = selected_logits.detach().clone()
        targets.scatter_(1, selected_labels[:, None], float("-inf"))
        probabilities = torch.softmax(targets, dim=-1)
        loss = torch.nn.functional.cross_entropy(selected_logits, probabilities)
        loss.backward()
        torch.cuda.synchronize()
        forward_torch_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        forward_torch_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        forward_nvml = nvml_state(os.getpid(), local_rank)
        del output, logits, selected_logits, selected_labels, targets, probabilities, loss
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        component = torch.full((1_000_000,), float(rank * 2 + 1), device=device)
        dist.all_reduce(component, op=dist.ReduceOp.SUM)
        component.div_(world_size)
        component_digest = tensor_hash(component[:4096])
        component_equal, component_hashes = all_rank_equal(component_digest)
        if not component_equal:
            raise RuntimeError("component all-reduce hashes differ")

        torch.cuda.reset_peak_memory_stats(device)
        basis = torch.empty((8, LARGEST), dtype=torch.bfloat16, device=device)
        forget_component = torch.empty(LARGEST, dtype=torch.float32, device=device)
        kl_component = torch.empty(LARGEST, dtype=torch.float32, device=device)
        metric = torch.empty(LARGEST, dtype=torch.float32, device=device)
        projected = torch.empty(LARGEST, dtype=torch.float32, device=device)
        temporary = torch.empty(LARGEST, dtype=torch.float32, device=device)
        basis[:, :4096].fill_(0)
        basis[0, :4096].fill_(1)
        forget_component[:4096].copy_(component[:4096])
        kl_component[:4096].fill_(1)
        metric[:4096].fill_(1)
        temporary[:4096] = forget_component[:4096] * metric[:4096]
        coefficient = (temporary[:4096] * basis[0, :4096].float()).sum()
        projected[:4096] = temporary[:4096] - coefficient / 4096 * basis[0, :4096]
        basis_digest = tensor_hash(basis[:, :4096])
        projected_digest = tensor_hash(projected[:4096])
        basis_equal, basis_hashes = all_rank_equal(basis_digest)
        projected_equal, projected_hashes = all_rank_equal(projected_digest)
        if not basis_equal or not projected_equal:
            raise RuntimeError("basis or projected gradient hashes differ")
        torch.cuda.synchronize()
        projection_torch_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        projection_torch_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        projection_nvml = nvml_state(os.getpid(), local_rank)
        peak = max(
            forward_torch_reserved,
            projection_torch_reserved,
            forward_nvml["process_used_mib"],
            projection_nvml["process_used_mib"],
        )
        result.update(
            status="measured",
            initial_nvml=initial,
            static_nvml=static,
            forward_torch_allocated_mib=forward_torch_allocated,
            forward_torch_reserved_mib=forward_torch_reserved,
            forward_nvml=forward_nvml,
            projection_torch_allocated_mib=projection_torch_allocated,
            projection_torch_reserved_mib=projection_torch_reserved,
            projection_nvml=projection_nvml,
            peak_mib=peak,
            pass_threshold_mib=THRESHOLD,
            passed=peak < THRESHOLD,
            component_hash=component_digest,
            component_hashes=component_hashes,
            basis_hash=basis_digest,
            basis_hashes=basis_hashes,
            projected_gradient_hash=projected_digest,
            projected_gradient_hashes=projected_hashes,
            model_numel=sum(p.numel() for p in current.parameters()),
            selected_numel=SELECTED,
            moment_tensor_count=len(moments_1) + len(moments_2),
            ref_logits_cpu_bytes=ref_logits_cpu.numel() * ref_logits_cpu.element_size(),
            elapsed_seconds=time.monotonic() - started,
        )
    except Exception as error:
        result.update(
            status="process_failed",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=time.monotonic() - started,
        )
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, result)
    if rank == 0:
        overall = {
            "schema_version": 1,
            "metric": int(all(item.get("passed") is True for item in gathered)),
            "status": "measured" if all(item.get("status") == "measured" for item in gathered) else "process_failed",
            "world_size": world_size,
            "ranks": gathered,
            "cross_rank_pass": all(item.get("passed") is True for item in gathered),
            "forbidden_artifacts": [],
        }
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(overall, indent=2, sort_keys=True, allow_nan=False) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if result["status"] != "measured" or not result.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
