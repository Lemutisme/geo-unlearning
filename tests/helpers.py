import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput


def make_lm_batch(batch_size=2, sequence_length=6, vocab_size=13, seed=0):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
    )
    labels = input_ids.clone()
    labels[:, :2] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def make_unlearn_batch(
    batch_size=2,
    sequence_length=6,
    vocab_size=13,
    seed=0,
):
    return {
        "forget": make_lm_batch(
            batch_size,
            sequence_length,
            vocab_size,
            seed,
        ),
        "retain": make_lm_batch(
            batch_size,
            sequence_length,
            vocab_size,
            seed + 1,
        ),
    }


def nested_collator(features):
    return {
        component: {
            key: torch.stack([feature[component][key] for feature in features])
            for key in features[0][component]
        }
        for component in features[0]
    }


class TinyCausalLM(nn.Module):
    main_input_name = "input_ids"

    def __init__(self, vocab_size=13, hidden_size=7):
        super().__init__()
        self.config = PretrainedConfig(vocab_size=vocab_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask
        logits = self.lm_head(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)
