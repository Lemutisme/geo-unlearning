# UAM Smoke Cleanup Edge Cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents are forbidden for this task.

**Goal:** Close signal-cleanup, checkpoint-deletion, report-rollback, and validation gaps in the UAM smoke tooling without running a real model.

**Architecture:** Keep process ownership layered: the matrix gives each arm enough time to terminate and reap its own training process group before applying a fallback KILL. Keep checkpoint cleanup NUL-delimited, publish analyzer outputs transactionally with same-directory backup files, and reject unsafe inputs before environment bootstrap or filesystem writes.

**Tech Stack:** Bash, Python 3, pytest, Ruff

---

### Task 1: Grace hierarchy and orphan fallback

**Files:**
- Modify: `tests/test_uam_scripts.py`
- Modify: `scripts/uam_smoke_arm.sh`
- Modify: `scripts/uam_smoke_matrix.sh`

- [ ] Add a regression that launches the real matrix and arm scripts with a fake `accelerate` process group that ignores TERM. Set `UAM_ARM_TERM_GRACE_SECONDS=1` and `UAM_MATRIX_TERM_GRACE_SECONDS=3`, terminate the matrix, and assert both trainers and their descendants are gone and each active arm has exactly one manifest row.
- [ ] Add validation cases proving the arm rejects a malformed/negative arm grace and the matrix rejects malformed/negative grace or `matrix_grace <= arm_grace` before Conda or path writes.
- [ ] Run only the new grace tests and confirm failures show the hard-coded equal deadlines and missing validation.
- [ ] In the arm, default `UAM_ARM_TERM_GRACE_SECONDS` to `5`, validate it as a nonnegative integer before Conda, and use it for the training-group TERM deadline.
- [ ] In the matrix, default arm grace to `5` and matrix grace to `15`, validate both as nonnegative integers with `matrix > arm`, export the arm value to children, and use matrix grace for the arm TERM deadline.
- [ ] Re-run the grace tests and the existing signal tests; require all to pass.

### Task 2: NUL-safe checkpoint deletion

**Files:**
- Modify: `tests/test_uam_scripts.py`
- Modify: `scripts/uam_smoke_arm.sh`

- [ ] Extend the fake evaluator to accept an exact checkpoint directory name and add a regression using `checkpoint-10\n--preserve-root`; place a sentinel at the newline suffix path in the subprocess working directory.
- [ ] Run the regression and confirm the current newline-delimited `mapfile` either leaves the actual checkpoint or removes the sentinel.
- [ ] Change checkpoint discovery to `find ... -print0` consumed by `mapfile -d ''`, leaving the existing symlink and checkpoint-file audits before deletion.
- [ ] Re-run the new deletion regression and the pre-deletion audit regressions; require the exact checkpoint directory to be removed and sentinels to remain.

### Task 3: Atomic report rollback

**Files:**
- Modify: `tests/test_uam_scripts.py`
- Modify: `scripts/analyze_uam_smoke.py`

- [ ] Add parameterized tests for prior-present and prior-absent Markdown/JSON destinations. Monkeypatch `os.replace` to fail on the second publication replacement, then assert both exact prior bytes/existence and no temporary files.
- [ ] Run the new tests and confirm Markdown changes under the current two-replace implementation.
- [ ] Prepare same-directory, fsynced backup temporary files for existing destinations before publication. If either publication fails, atomically replace prior files from backups or unlink destinations that were previously absent, then remove every unused new-output or backup temporary.
- [ ] Re-run the new rollback tests plus serialization and temp-fsync failure tests; require all to pass.

### Task 4: Validation constraints

**Files:**
- Modify: `tests/test_uam_scripts.py`
- Modify: `scripts/analyze_uam_smoke.py`
- Modify: `scripts/uam_smoke_arm.sh`

- [ ] Add negative analyzer fixtures for every norm-like geometry value and every actual-delta norm, including zero `parameter_delta_norm`; add GPU `2`; add relative and symlink-component local-root cases that assert rejection before training and writes.
- [ ] Run the new validation tests and confirm they expose the absent constraints.
- [ ] Require all numeric geometry fields ending in `_norm` to be nonnegative while retaining positive requested/effective perturbation norm and ratio requirements. Require actual gradient norms to be nonnegative and `parameter_delta_norm` to be positive.
- [ ] Restrict arm GPU values to exactly `0` or `1` before Conda.
- [ ] Validate `UAM_LOCAL_ROOT` as absolute, reject every existing symlink/non-directory component, canonicalize the root and arm path, and prove the arm path remains below the root before creating directories.
- [ ] Re-run the new validation tests and relevant existing validation/audit tests; require all to pass.

### Task 5: Final verification and commit

**Files:**
- Verify: `scripts/uam_smoke_arm.sh`
- Verify: `scripts/uam_smoke_matrix.sh`
- Verify: `scripts/analyze_uam_smoke.py`
- Verify: `tests/test_uam_scripts.py`

- [ ] Run `PYTHONDONTWRITEBYTECODE=1 timeout 90 /root/miniconda3/envs/unlearning/bin/pytest -p no:cacheprovider -q tests/test_uam_scripts.py`.
- [ ] Run `bash -n scripts/uam_smoke_arm.sh scripts/uam_smoke_matrix.sh`.
- [ ] Run Ruff check and format check for the analyzer and tests, then `git diff --check`.
- [ ] Inspect recorded fake-process PIDs and `/proc` to ensure no test descendants remain.
- [ ] Review the final diff for scope, stage the five changed files, and commit exactly as `fix: close UAM smoke cleanup edge cases`.
