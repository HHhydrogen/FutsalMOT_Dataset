# Camera Distribution Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved P2-5 Camera Distribution Strategy using fixed C1-C5 anchors plus an explicit optional Partial Camera extension, without adding a new camera subsystem.

**Architecture:** Extend the existing `simulation.camera.distribution` configuration and `simulation.camera.profiles` set. The distribution policy will select a deterministic camera ID list before runtime; Coverage and Placement remain profile concerns, while `ResolvedTask`, `camera_distribution.py`, `run_task.py`, Sequence, and MRQ retain their existing responsibilities.

**Tech Stack:** Python 3.9, Pydantic, pytest, existing Unreal Python runtime, existing MRQ path.

**Spec:** `docs/superpowers/specs/camera-distribution-strategy-design.md`

## Global Constraints

- C1-C5 remain mandatory canonical Static Surveillance anchors.
- C1-C5 are never randomly deleted or replaced by Partial Cameras.
- Current supported Partial Camera is P01 only.
- P01 is optional and is only valid when its profile and UE mapping are both present.
- Dynamic Broadcast and B-series cameras remain future-only and are rejected by the current contract.
- Camera Distribution only selects the Camera IDs required by an episode.
- Coverage, placement, runtime application, Sequence, and MRQ continue using existing modules.
- Do not add Camera Manager, Coverage Manager, Camera Database, Camera Registry, scheduler, or probability service.
- Do not modify ValidationResult, Pipeline State, Run Manifest, Audit/Cleanup, or `camera_state.jsonl`.
- Do not create or modify UE assets, Blueprint, Level Sequence, Camera Cut, or MRQ architecture.
- Preserve legacy tasks that do not define the new distribution policy.
- Do not commit or push without explicit user confirmation.

## File Map

- Modify: `src/grf_ue_bridge/config/models.py` to add the minimal explicit episode distribution policy to the existing `CameraDistributionConfig`.
- Modify: `src/grf_ue_bridge/config/loader.py` to validate the selected camera set against profiles and mappings while preserving legacy behavior.
- Modify: `src/grf_ue_bridge/config/resolver.py` to carry the normalized distribution policy and selected camera IDs through `ResolvedTask` without adding a parallel artifact.
- Modify: `ue/camera_distribution.py` to resolve deterministic selected Camera IDs from the existing distribution policy and existing profile/mapping data.
- Modify: `ue/run_task.py` only where necessary to consume the resolved selected Camera IDs; the existing camera apply, Sequence, Camera Cut, and MRQ calls remain unchanged.
- Test: `tests/test_task_config.py` for policy parsing and contract validation.
- Test: `tests/test_task_resolver.py` for resolved-task propagation.
- Test: `tests/test_camera_distribution.py` for deterministic selection and legacy fallback.
- Test: `tests/test_ue_resolved_task.py` for serialized runtime contract.
- Test: `tests/test_task_cli.py` for current fixture behavior.

### Task 1: Add Explicit Episode Camera Distribution Policy

**Files:**
- Modify: `src/grf_ue_bridge/config/models.py`
- Modify: `tests/test_task_config.py`

**Interfaces:**
- Add an `episode_profile` field to the existing `CameraDistributionConfig`.
- Allowed values:
  - `anchor_only`
  - `anchor_plus_one_partial`
  - `anchor_plus_two_partial`
- Default to `anchor_only` so existing tasks retain C1-C5 runtime behavior.
- Do not add a new top-level configuration block.

- [ ] **Step 1: Write failing model tests**

```python
@pytest.mark.parametrize(
    "episode_profile",
    ["anchor_only", "anchor_plus_one_partial", "anchor_plus_two_partial"],
)
def test_camera_distribution_accepts_episode_profile(tmp_path, episode_profile):
    fields = _camera_task_fields()
    fields["simulation"]["camera"]["distribution"]["episode_profile"] = episode_profile
    tf = _write_task(tmp_path, **fields)

    task = loader.load_task_config(tf)

    assert task.simulation.camera.distribution.episode_profile == episode_profile


def test_camera_distribution_defaults_to_anchor_only(tmp_path):
    task = loader.load_task_config(_write_task(tmp_path, **_camera_task_fields()))

    assert task.simulation.camera.distribution.episode_profile == "anchor_only"


def test_camera_distribution_rejects_unknown_episode_profile(tmp_path):
    fields = _camera_task_fields()
    fields["simulation"]["camera"]["distribution"]["episode_profile"] = "random_cameras"
    tf = _write_task(tmp_path, **fields)

    with pytest.raises(ValueError, match="episode_profile"):
        loader.load_task_config(tf)
```

- [ ] **Step 2: Run the focused tests and verify the expected failure**

Run: `uv run python -m pytest tests/test_task_config.py -q`

Expected: FAIL because `CameraDistributionConfig` does not yet expose `episode_profile`.

- [ ] **Step 3: Implement the minimal model field**

```python
episode_profile: Literal[
    "anchor_only",
    "anchor_plus_one_partial",
    "anchor_plus_two_partial",
] = "anchor_only"
```

Keep `anchors` unchanged as the mandatory ordered C1-C5 list. Do not add ratios or random sampling logic.

- [ ] **Step 4: Run the focused model tests**

Run: `uv run python -m pytest tests/test_task_config.py -q`

Expected: PASS for the new policy tests and all existing task configuration tests.

### Task 2: Validate Policy Against Profiles and Mappings

**Files:**
- Modify: `src/grf_ue_bridge/config/loader.py`
- Modify: `src/grf_ue_bridge/config/resolver.py`
- Modify: `tests/test_task_config.py`
- Modify: `tests/test_task_resolver.py`

**Interfaces:**
- `anchor_only` requires C1-C5 profiles and C1-C5 mappings; P01 may exist but is not selected.
- `anchor_plus_one_partial` requires P01 profile and P01 mapping.
- `anchor_plus_two_partial` is a declared future policy and must fail clearly until a second approved Partial ID exists; it must not accept P02/P03.
- Mapping IDs remain limited to C1-C5 plus P01.
- Legacy tasks with no `simulation` block remain unchanged.

- [ ] **Step 1: Write failing validation tests**

```python
    fields = _partial_camera_task_fields()
    del fields["ue"]["camera_mapping"]["P01"]
    fields["simulation"]["camera"]["distribution"]["episode_profile"] = (
        "anchor_plus_one_partial"
    )
    tf = _write_task(tmp_path, **fields)

    with pytest.raises(ValueError, match="P01"):
        loader.load_task_config(tf)


def test_anchor_only_allows_optional_p01_profile_and_mapping(tmp_path):
    task = loader.load_task_config(_write_task(tmp_path, **_partial_camera_task_fields()))

    assert task.simulation.camera.distribution.episode_profile == "anchor_only"


def test_anchor_plus_two_partial_is_rejected_until_second_partial_exists(tmp_path):
    fields = _partial_camera_task_fields()
    fields["simulation"]["camera"]["distribution"]["episode_profile"] = (
        "anchor_plus_two_partial"
    )
    tf = _write_task(tmp_path, **fields)

    with pytest.raises(ValueError, match="anchor_plus_two_partial|P02"):
        loader.load_task_config(tf)
```

- [ ] **Step 2: Run the validation tests and confirm they fail for the missing policy behavior**

Run: `uv run python -m pytest tests/test_task_config.py::test_anchor_plus_one_partial_requires_p01_mapping tests/test_task_config.py::test_anchor_plus_two_partial_is_rejected_until_second_partial_exists -q`

Expected: FAIL because policy-specific validation is not implemented.

- [ ] **Step 3: Add policy validation at the existing loader/model boundary**

Implement these exact rules after profile and mapping IDs are known:

```python
if episode_profile == "anchor_plus_one_partial":
    require "P01" in profiles and "P01" in mapping
elif episode_profile == "anchor_plus_two_partial":
    raise ValueError("anchor_plus_two_partial 当前需要第二个已实现 Partial Camera，例如 P02")
```

Keep C1-C5 mandatory and retain the existing unknown-ID rejection. Do not add a scheduler or probabilistic selection.

- [ ] **Step 4: Add resolver propagation coverage**

Assert that `ResolvedTask.simulation["camera"]["distribution"]["episode_profile"]` equals the selected policy and that P01 remains in `ResolvedTask.simulation.camera.profiles` when supplied.

- [ ] **Step 5: Run focused tests**

Run: `uv run python -m pytest tests/test_task_config.py tests/test_task_resolver.py -q`

Expected: PASS.

### Task 3: Resolve Deterministic Camera IDs

**Files:**
- Modify: `ue/camera_distribution.py`
- Modify: `tests/test_camera_distribution.py`

**Interfaces:**
- Add `resolve_episode_camera_ids(resolved_task: Dict[str, Any]) -> list[str]`.
- Return exactly:
  - `anchor_only` -> `['C1', 'C2', 'C3', 'C4', 'C5']`
  - `anchor_plus_one_partial` -> `['C1', 'C2', 'C3', 'C4', 'C5', 'P01']`
- Reject `anchor_plus_two_partial` with a clear error.
- Keep `resolve_runtime_camera_selection()` default behavior unchanged for callers that do not request a policy.

- [ ] **Step 1: Write failing deterministic selection tests**

```python
    resolved = _resolved_with_profiles_and_mapping(episode_profile="anchor_only")

    assert resolve_episode_camera_ids(resolved) == ["C1", "C2", "C3", "C4", "C5"]


def test_anchor_plus_one_partial_selects_p01_after_anchors():
    resolved = _resolved_with_profiles_and_mapping(
        episode_profile="anchor_plus_one_partial",
        include_p01=True,
    )

    assert resolve_episode_camera_ids(resolved) == [
        "C1", "C2", "C3", "C4", "C5", "P01"
    ]


def test_camera_selection_is_repeatable():
    resolved = _resolved_with_profiles_and_mapping(
        episode_profile="anchor_plus_one_partial",
        include_p01=True,
    )

    assert resolve_episode_camera_ids(resolved) == resolve_episode_camera_ids(resolved)
```

- [ ] **Step 2: Run the tests to verify the expected missing-function failure**

Run: `uv run python -m pytest tests/test_camera_distribution.py -q`

Expected: FAIL because `resolve_episode_camera_ids` does not exist.

- [ ] **Step 3: Implement the fixed policy table**

Use a small literal mapping inside `camera_distribution.py`; do not derive IDs from arbitrary prefixes or use random sampling.

```python
_EPISODE_CAMERA_IDS = {
    "anchor_only": ["C1", "C2", "C3", "C4", "C5"],
    "anchor_plus_one_partial": ["C1", "C2", "C3", "C4", "C5", "P01"],
}
```

Verify every selected ID has a profile and, for P01, a mapping. Return a new list so callers cannot mutate the policy table.

- [ ] **Step 4: Integrate policy selection into runtime selection**

When no explicit `camera_ids` override is supplied, use `resolve_episode_camera_ids()` for canonical task selection. Preserve legacy fallback when `simulation` or `ue.camera_mapping` is absent. Keep the existing P01-only explicit selection path available for smoke/debug use.

- [ ] **Step 5: Run focused camera tests**

Run: `uv run python -m pytest tests/test_camera_distribution.py -q`

Expected: PASS.

### Task 4: Carry Selected Camera Set Through `run_task.py`

**Files:**
- Modify: `ue/run_task.py`
- Modify: `tests/test_ue_resolved_task.py`
- Modify: `tests/test_task_cli.py`

**Interfaces:**
- `run_task.py` consumes the selected entry list from `camera_distribution.py`.
- For `anchor_plus_one_partial`, the sequence list becomes C1-C5 plus the existing P01 mapping.
- Existing `_apply_resolved_anchor_cameras()` is reused for P01; no separate runtime pipeline is added.
- Existing `C5_CAMERA_IDS=P01` explicit override remains supported.

- [ ] **Step 1: Write failing serialization/runtime contract tests**

```python
    _make_resolved_with_p01(tmp_path, pin_repo_root)
    task_path = pin_repo_root / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["simulation"]["camera"]["distribution"]["episode_profile"] = (
        "anchor_plus_one_partial"
    )
    task_path.write_text(json.dumps(task), encoding="utf-8")

    resolved = resolver.resolve_task(task_path)

    assert resolved.simulation["camera"]["distribution"]["episode_profile"] == (
        "anchor_plus_one_partial"
    )
    assert resolved.ue_profile["camera_mapping"]["P01"] == {
        "actor": "CineCam_P01",
        "sequence": "LS_Cam_P01",
    }
```

- [ ] **Step 2: Run the contract tests and verify the expected failure**

Run: `uv run python -m pytest tests/test_ue_resolved_task.py tests/test_task_cli.py -q`

Expected: FAIL until the runtime policy is consumed by the existing selection path.

- [ ] **Step 3: Update `run_task.py` minimally**

Use the resolved selection returned by `resolve_runtime_camera_selection()`. Do not add any asset creation. Preserve:

- explicit position application
- explicit rotation application
- focal length application
- resolution consistency check
- existing sequence loading
- existing Camera Cut behavior
- existing asynchronous MRQ behavior

- [ ] **Step 4: Run focused runtime contract tests**

Run: `uv run python -m pytest tests/test_ue_resolved_task.py tests/test_task_cli.py tests/test_camera_distribution.py -q`

Expected: PASS.

### Task 5: Full Verification and Scope Audit

**Files:**
- No new files.

- [ ] **Step 1: Run the full Python suite**

Run: `uv run python -m pytest`

Expected: all selected tests pass; GRF integration remains deselected according to repository configuration.

- [ ] **Step 2: Run whitespace verification**

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 3: Inspect the intended diff**

Run: `git diff --stat -- src/grf_ue_bridge/config/models.py src/grf_ue_bridge/config/loader.py src/grf_ue_bridge/config/resolver.py ue/camera_distribution.py ue/run_task.py tests/test_task_config.py tests/test_task_resolver.py tests/test_camera_distribution.py tests/test_ue_resolved_task.py tests/test_task_cli.py`

Confirm no changes to UE assets, Blueprint, Sequence, Camera Cut, MRQ architecture, ValidationResult, Pipeline State, Run Manifest, Audit/Cleanup, or `camera_state.jsonl`.

- [ ] **Step 4: Run a contract-only smoke**

Verify through Python that:

```text
anchor_only -> C1 C2 C3 C4 C5
anchor_plus_one_partial -> C1 C2 C3 C4 C5 P01
```

Do not run a full dataset render and do not create or modify UE assets.

- [ ] **Step 5: Report limitations**

Report that `anchor_plus_two_partial` remains a declared future policy and is rejected until a second approved Partial Camera exists. Report that Dynamic Broadcast remains outside the current model.
