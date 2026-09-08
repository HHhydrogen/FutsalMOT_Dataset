# Camera Distribution Anchors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first Camera Distribution Model slice for fixed canonical anchors `C1..C5`, flowing from Task Spec through ResolvedTask into the existing UE Sequence/MRQ pipeline and producing a minimal `camera_state.jsonl` artifact.

**Architecture:** Add explicit `simulation.camera` and `ue.camera_mapping` data to the existing Task Spec and ResolvedTask contracts. Keep `C1..C5` as dataset semantic IDs, resolve them to already-existing UE Actor/Sequence bindings, and apply only explicitly configured camera state before reusing the existing Sequence, Camera Cut, annotation, and MRQ workflow. C5 is accepted only when confirmed in the live UE Editor/MCP; code must fail clearly when its mapping or Actor is absent.

**Tech Stack:** Python 3.9, Pydantic, pytest, Unreal Engine 5.8 Editor Python, existing `run_task.py`, Level Sequence, Camera Cut, Movie Render Queue, Unreal MCP.

**Spec:** `docs/superpowers/specs/2026-09-06-camera-distribution-design.md`

## Global Constraints

- Implement only fixed `C1..C5` anchor cameras in this phase; do not implement random cameras, partial cameras, or dynamic broadcast behavior.
- Add schema and resolver support before changing UE behavior.
- Do not create `Camera Manager`, `Simulation Framework`, `Dataset Management System`, `Camera Database`, or any parallel configuration system.
- Do not automatically create UE Camera Actors. Only bind and configure Actors that exist and are confirmed in the current UE Editor.
- Do not assume `CineCam_01..04` are semantically equal to `C1..C4`; use explicit mapping.
- Do not assume C5 exists. Missing C5 mapping or Actor is a fail-fast implementation error.
- Keep `ResolvedTask` as the only P1/P2 runtime contract. UE scripts must not read the original Task Spec.
- `camera_state.jsonl` first version records actual applied state only; it does not compute overlap, topology, visibility, or other complex analysis.
- Do not modify Blueprint, MCP plugin, ValidationResult, Pipeline State, or Run Manifest in this phase.
- Work in `Content/FutsalMOT/code/` as the inner Dataset repository. Do not commit or push without explicit user confirmation.
- Python-side tests run in the repository Python 3.9 environment; UE-side scripts run only in Unreal Editor Python.
- All new technical documentation, comments, and docstrings use Simplified Chinese.

---

## File Map

**Task Spec and resolver contract**

- Modify `src/grf_ue_bridge/config/models.py`: add explicit `SimulationTaskConfig` / camera models, `DatasetTaskConfig.simulation`, and resolved-task camera payload fields while preserving existing `ue` configuration.
- Modify `src/grf_ue_bridge/config/resolver.py`: validate anchor IDs, profiles, mapping, and copy normalized simulation camera data into the resolved task.
- Modify `tests/test_task_config.py`: schema acceptance and rejection tests.
- Modify `tests/test_task_resolver.py`: resolved-task propagation and validation tests.
- Modify `tests/test_ue_resolved_task.py`: serialized resolved-task contract tests.

**UE integration and artifact**

- Modify `ue/run_task.py`: read the resolved camera contract, validate/prepare existing anchor bindings, and pass camera profiles/mapping to existing sequence and annotation functions.
- Modify `ue/import_grf_episode.py`: accept canonical camera entries while continuing to create existing Level Sequences, Camera Cut tracks, and transform tracks for mapped Actors.
- Modify `ue/annotation_exporter.py`: associate canonical camera IDs with actual calibration output and write the minimal actual camera-state artifact through a focused helper.
- Modify `ue/render_episode.py` only if the actual applied state must be finalized after the existing render setup; do not duplicate camera application logic here.
- Add `ue/camera_distribution.py` only if the existing files cannot contain focused pure/runtime helpers without mixing responsibilities. Its responsibility must remain camera profile validation/application/state serialization, not Actor management.

**Artifact contract and docs**

- Modify `docs/DATA_CONTRACT.md`: document `simulation.camera`, `ue.camera_mapping`, canonical IDs, and `camera_state.jsonl` after implementation behavior is verified.
- Modify `docs/VALIDATION_AND_LIMITATIONS.md`: record the actual anchor/C5 validation boundary and the fact that complex camera analysis is not implemented.

**UE asset boundary**

- Do not modify `.uasset`, `.umap`, Blueprint, plugin, or `Config/DefaultEngine.ini` in this phase. C5 availability is confirmed through live UE Editor/MCP only; if no C5 Actor exists, stop the UE implementation at the explicit diagnostic rather than creating one.

---

### Task 1: Add Camera Distribution Schema

**Files:**
- Modify: `src/grf_ue_bridge/config/models.py`
- Test: `tests/test_task_config.py`

**Interfaces:**
- Consumes: Existing `DatasetTaskConfig`, `UeProfile`, `ResolvedTask`, and Pydantic model validation.
- Produces: `DatasetTaskConfig.simulation.camera`, `SimulationTaskConfig`, `CameraDistributionConfig`, `CameraProfileConfig`, and `UeProfile.camera_mapping` with stable field names for the resolver.

- [ ] **Step 1: Write failing schema tests**

Add tests covering a valid five-anchor task and invalid anchor/profile data. Use this concrete fixture shape:

```python
"simulation": {
    "camera": {
        "distribution": {
            "system": "dual",
            "anchors": ["C1", "C2", "C3", "C4", "C5"],
            "static_surveillance_ratio": [0.70, 0.80],
            "dynamic_broadcast_ratio": [0.20, 0.30]
        },
        "profiles": {
            "C1": {
                "type": "static_surveillance",
                "coverage": "full_field",
                "resolution": [1920, 1080],
                "lens": {"horizontal_fov_deg": 72.0},
                "position_m": [1.0, -8.0, 8.0],
                "rotation_deg": [0.0, 45.0, 0.0],
                "height_m": 8.0,
                "distortion": None
            }
        }
    }
},
"ue": {
    "camera_mapping": {
        "C1": {"actor": "CineCam_01", "sequence": "LS_Cam_01"}
    }
}
```

Assert that valid data is loaded, `C1..C5` are represented as strings, invalid camera type is rejected, partial/broadcast profiles are rejected in this anchor-only phase, resolution dimensions must be positive, and a profile with neither focal length nor horizontal FOV is rejected.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
uv run pytest tests/test_task_config.py -q
```

Expected: the new tests fail because the explicit simulation/camera models and `camera_mapping` field do not yet exist.

- [ ] **Step 3: Implement the minimal schema**

Add focused Pydantic models in `models.py`:

```python
class CameraLensConfig(BaseModel):
    focal_length_mm: Optional[float] = Field(None, gt=0.0)
    horizontal_fov_deg: Optional[float] = Field(None, gt=0.0, lt=180.0)

    def validate_defined(self) -> None:
        if self.focal_length_mm is None and self.horizontal_fov_deg is None:
            raise ValueError("camera lens 必须提供 focal_length_mm 或 horizontal_fov_deg")


class CameraProfileConfig(BaseModel):
    type: Literal["static_surveillance"]
    coverage: Literal["full_field"]
    resolution: List[int]
    lens: CameraLensConfig
    position_m: List[float]
    rotation_deg: List[float]
    height_m: float = Field(..., gt=0.0)
    distortion: Optional[Dict] = None


class CameraDistributionConfig(BaseModel):
    system: Literal["dual"] = "dual"
    anchors: List[Literal["C1", "C2", "C3", "C4", "C5"]]
    static_surveillance_ratio: List[float]
    dynamic_broadcast_ratio: List[float]

    def validate_anchor_contract(self) -> None:
        required = ["C1", "C2", "C3", "C4", "C5"]
        if self.anchors != required:
            raise ValueError("anchor cameras 必须按 C1..C5 顺序完整提供")


class SimulationCameraConfig(BaseModel):
    distribution: CameraDistributionConfig
    profiles: Dict[str, CameraProfileConfig]

    def validate_camera_contract(self) -> None:
        self.distribution.validate_anchor_contract()
        required = ["C1", "C2", "C3", "C4", "C5"]
        if sorted(self.profiles) != required:
            raise ValueError("camera profiles 必须完整提供 C1..C5")
        for profile in self.profiles.values():
            if len(profile.resolution) != 2 or any(v <= 0 for v in profile.resolution):
                raise ValueError("camera resolution 必须为正数 [width, height]")
            if len(profile.position_m) != 3 or len(profile.rotation_deg) != 3:
                raise ValueError("camera position_m / rotation_deg 必须为三维向量")
            if abs(profile.position_m[2] - profile.height_m) > 1e-6:
                raise ValueError("camera height_m 必须与 position_m[2] 一致")
            profile.lens.validate_defined()


class CameraMappingConfig(BaseModel):
    actor: str
    sequence: str


class SimulationTaskConfig(BaseModel):
    camera: SimulationCameraConfig
```

Use model validators compatible with the repository’s installed Pydantic version, and invoke the explicit cross-field validation from the existing loader validation path. Add `camera_mapping: Dict[str, CameraMappingConfig]` to `UeProfile` with `extra="allow"` retained for unrelated fields.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run:

```powershell
uv run pytest tests/test_task_config.py -q
```

Expected: all existing task-config tests and the new camera schema tests pass.

- [ ] **Step 5: Run formatting/whitespace verification**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

---

### Task 2: Propagate and Validate Through ResolvedTask

**Files:**
- Modify: `src/grf_ue_bridge/config/models.py`
- Modify: `src/grf_ue_bridge/config/resolver.py`
- Modify: `tests/test_task_resolver.py`
- Modify: `tests/test_ue_resolved_task.py`

**Interfaces:**
- Consumes: `SimulationTaskConfig`, `CameraMappingConfig`, and `UeProfile.camera_mapping` from Task 1.
- Produces: `ResolvedTask.simulation` containing normalized `camera`, and `ResolvedTask.ue_profile.camera_mapping` containing explicit Canonical ID to UE Actor/Sequence mappings.

- [ ] **Step 1: Write failing propagation tests**

Extend the resolver fixture with all five profiles and mappings. Assert:

```python
resolved.simulation["camera"]["distribution"]["anchors"] == ["C1", "C2", "C3", "C4", "C5"]
resolved.ue_profile["camera_mapping"]["C5"]["actor"] == "CineCam_Main"
```

Add rejection tests for missing `C5`, a mapping key not present in the anchor list, a mapping missing `actor`, a mapping missing `sequence`, a duplicate UE Actor, and an invalid profile ID. Assert errors identify the Canonical ID or mapping field rather than failing with an opaque KeyError.

- [ ] **Step 2: Run focused resolver tests and verify failure**

Run:

```powershell
uv run pytest tests/test_task_resolver.py tests/test_ue_resolved_task.py -q
```

Expected: new assertions fail because `ResolvedTask` does not yet carry `simulation` and resolver validation does not yet enforce the mapping contract.

- [ ] **Step 3: Add resolved-task simulation field and resolver validation**

Add to `ResolvedTask`:

```python
simulation: Dict = Field(default_factory=dict, description="解析后的 simulation 参数")
```

In `validate_task()` and `resolve_task()`:

1. Call `task.simulation.camera.validate_camera_contract()`.
2. Require exactly `C1..C5` in the camera mapping for the anchor-only phase.
3. Require mapping entries to contain non-empty `actor` and `sequence` strings.
4. Reject duplicate mapped Actor names.
5. Copy `task.simulation.model_dump()` into `ResolvedTask.simulation`.
6. Copy normalized `camera_mapping` into `ue_profile` without converting Canonical IDs into UE names.
7. Keep the current `ue.sequences` behavior untouched until UE integration consumes the new mapping; do not silently synthesize sequence entries in the resolver.

Use explicit `ValueError` messages in Simplified Chinese containing the offending Canonical ID.

- [ ] **Step 4: Run focused resolver tests and verify they pass**

Run:

```powershell
uv run pytest tests/test_task_resolver.py tests/test_ue_resolved_task.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Run the full non-GRF Python suite**

Run:

```powershell
uv run pytest
```

Expected: existing tests remain green; no GRF integration tests are required for this schema-only task.

---

### Task 3: Create a Five-Anchor Task Fixture Without Assuming C5 Exists

**Files:**
- Create: `configs/camera_anchor_smoke_5cam.json`
- Modify: `tests/test_task_cli.py` only if fixture discovery requires it

**Interfaces:**
- Consumes: resolved schema and mapping contract from Tasks 1-2.
- Produces: a checked-in example task containing explicit `C1..C5` semantic profiles and UE mappings, with no random-camera fields.

- [ ] **Step 1: Add fixture validation test**

Add a test that loads the fixture and asserts:

```python
task.ue.camera_mapping.keys() == {"C1", "C2", "C3", "C4", "C5"}
task.simulation.camera.distribution.anchors == ["C1", "C2", "C3", "C4", "C5"]
```

The test must not load UE or assert that `CineCam_Main` exists.

- [ ] **Step 2: Write the fixture with explicit placeholder-independent paths**

Use `${FUTSALMOT_DATASET_ROOT}` and `${FUTSALMOT_UE_ROOT}`. Include five static full-field profiles with explicit resolution, FOV or focal length, position, rotation, and height. Include mappings such as the existing four names plus a clearly named C5 mapping candidate, but document in the fixture comment-free JSON metadata that the mapping requires live UE confirmation. Do not add `partial`, `broadcast`, random seed, or auto-create flags.

- [ ] **Step 3: Run fixture validation**

Run:

```powershell
uv run pytest tests/test_task_cli.py tests/test_task_config.py -q
```

Expected: fixture loads and validates as a Task Spec; no test claims that C5 is present in the map.

---

### Task 4: Confirm Live UE Anchor Bindings Before UE Code Changes

**Files:**
- No file changes.

**Interfaces:**
- Consumes: `configs/camera_anchor_smoke_5cam.json` and current UE Editor state.
- Produces: a verified mapping decision for all five slots, or a blocking diagnostic identifying the missing C5 Actor/Sequence.

- [ ] **Step 1: Check current Unreal MCP connection**

Use the Unreal MCP status/native tools to confirm the Editor is connected and query the current Level. The expected target is `/Game/FutsalMOT/Maps/L_FutsalCourt`; do not trust `DefaultEngine.ini`.

- [ ] **Step 2: Query existing camera Actors**

Use native Actor lookup/components tools or `FutsalMOTTools.run_python_code` for a short diagnostic to list Actors whose labels match the candidate mappings and confirm each has a `CineCameraComponent`. Record actual labels, transforms, component focal length/FOV, and resolution-relevant properties.

- [ ] **Step 3: Confirm existing Sequence assets and Camera Cut capability**

Query the four existing Sequence assets and verify the existing `LS_Cam_01..04` relationship. For C5, verify whether a pre-existing Actor and Sequence are available. Do not create either one automatically.

- [ ] **Step 4: Decide whether UE implementation can proceed**

Proceed only if five existing Actor/Sequence bindings are confirmed. If C5 is absent, stop before UE modifications and report the exact missing binding; do not alter the map or create an Actor. If C5 exists under a different name, update only the task fixture/mapping after user confirmation, then re-run Task 3 tests.

---

### Task 5: Apply Existing Anchor Camera Profiles in UE

**Files:**
- Modify: `ue/run_task.py`
- Modify: `ue/import_grf_episode.py`
- Add: `ue/camera_distribution.py` only if the helper boundary is needed
- Test: `tests/test_camera_distribution.py` for pure validation/serialization helpers

**Interfaces:**
- Consumes: `ResolvedTask.simulation.camera`, `ResolvedTask.ue_profile.camera_mapping`, and live UE bindings from Task 4.
- Produces: `resolve_anchor_bindings(resolved_camera, mapping) -> List[dict]`, `apply_anchor_profile(binding, profile) -> dict`, and `camera_state` rows containing actual Actor/component state.

- [ ] **Step 1: Write failing pure-helper tests**

Add tests for a pure `resolve_anchor_bindings()` helper using mocked mapping dictionaries. Assert it returns ordered `C1..C5` entries, preserves both `camera_id` and `ue_actor`, rejects missing C5, rejects duplicate Actor names, and does not create or mutate Actors.

Add tests for a pure state-normalization helper:

```python
row = normalize_camera_state(
    camera_id="C1",
    ue_actor="CineCam_01",
    sequence="LS_Cam_01",
    profile=profile_dict,
    actual={"resolution": [1920, 1080], "position_m": [1.0, -8.0, 8.0]},
    frame=0,
    source_step=0,
    time_seconds=0.0,
)
assert row["camera_id"] == "C1"
assert row["ue_actor"] == "CineCam_01"
```

- [ ] **Step 2: Run pure-helper tests and verify failure**

Run:

```powershell
uv run pytest tests/test_camera_distribution.py -q
```

Expected: FAIL because the helper module/functions do not yet exist.

- [ ] **Step 3: Implement binding resolution without Actor creation**

Implement only deterministic validation and application:

1. Load `C1..C5` in canonical order.
2. Resolve each mapping to an existing Actor with the current `find_actor()` mechanism.
3. If any Actor is missing, raise an error naming the Canonical ID and requested UE Actor.
4. Never call an Actor factory, spawn function, asset duplication function, or map-save operation to create a camera.
5. Require a CineCameraComponent.
6. Apply explicit resolution/FOV/focal length and transform to the existing Actor before Sequence creation, using UE cm for position and UE Rotator degrees for rotation.
7. Verify the applied state by reading it back from the Actor/component.
8. Reuse existing `_add_camera_cut()` and `_add_camera_transform_track()` for each mapped Sequence.

Keep camera application separate from player/ball scene application. Do not add a manager object or global service.

- [ ] **Step 4: Integrate with `run_task.py` after resolved-task loading**

Read `rt["simulation"]["camera"]` and `ue_profile["camera_mapping"]`. Before `create_sequence()` in `full` and `sequence` modes, resolve and apply the five existing bindings. Build the existing `sequences` list from explicit mappings only when the current pipeline requires it; preserve the current sequence package path and replace-existing behavior. In `render` mode, require that the existing Sequence and Actor bindings are present and validate state before MRQ submission.

- [ ] **Step 5: Run pure tests and static checks**

Run:

```powershell
uv run pytest tests/test_camera_distribution.py tests/test_task_resolver.py tests/test_ue_resolved_task.py -q
```

Expected: PASS with no whitespace errors.

- [ ] **Step 6: Execute the UE script through MCP**

Use `FutsalMOTTools.run_python_file` in the connected UE Editor to execute the existing `run_task.py` against the resolved anchor smoke task. Read the tool result and UE `LogPython`, `LogBlueprint`, and `LogMovieRenderPipeline` logs. If C5 is absent, treat the run as an expected blocking diagnostic and do not bypass it by creating an Actor.

---

### Task 6: Write Minimal Actual-State `camera_state.jsonl`

**Files:**
- Modify: `ue/annotation_exporter.py`
- Modify: `ue/run_task.py` if artifact orchestration belongs at the unified entry point
- Test: `tests/test_camera_distribution.py`
- Modify: `docs/DATA_CONTRACT.md`

**Interfaces:**
- Consumes: applied/read-back Actor and CineCameraComponent state from Task 5, canonical binding rows, and existing frame timing.
- Produces: `<episode>/camera_state.jsonl`, one actual-state row per camera per output frame, with `schema`, `version`, `frame`, `source_step`, `time_seconds`, `camera_id`, `ue_actor`, `sequence`, `type`, `coverage`, `resolution`, `focal_length_mm` or `horizontal_fov_deg`, `position_m`, `rotation_deg`, `height_m`, and `distortion`.

- [ ] **Step 1: Write failing artifact tests**

Test a pure writer with deterministic rows:

```python
write_camera_state_jsonl(path, rows)
lines = path.read_text(encoding="utf-8").splitlines()
assert json.loads(lines[0])["schema"] == "futsalmot_camera_state"
assert json.loads(lines[0])["version"] == 1
assert json.loads(lines[0])["camera_id"] == "C1"
```

Assert rows are ordered by `frame`, then canonical camera order; no overlap/topology analysis fields are required; and the writer rejects a row missing `camera_id`, `ue_actor`, resolution, or transform.

- [ ] **Step 2: Run focused artifact tests and verify failure**

Run:

```powershell
uv run pytest tests/test_camera_distribution.py -q
```

Expected: FAIL because the writer does not yet exist.

- [ ] **Step 3: Implement actual-state serialization**

Implement a small JSONL writer that records values read back from UE, not only requested profile values. For static anchors, repeated rows across frames are acceptable and expected. Use the existing frame convention: `frame` and `source_step` are 0-based, while `annotations.jsonl.frame_index` remains 1-based. Do not calculate coverage percentages, overlap, topology, visibility, or association metrics.

- [ ] **Step 4: Connect artifact generation to the existing annotation/full flow**

Generate `camera_state.jsonl` after all five camera states have been applied and read back, before or alongside existing `camera.json` generation. Ensure a failed/missing anchor does not leave a misleading complete artifact. Keep existing per-camera `camera.json` output unchanged except for an optional canonical `camera_id` field if the current contract can accept it without breaking validators; otherwise keep the canonical identity in `camera_state.jsonl` only.

- [ ] **Step 5: Document and test the artifact**

Update `DATA_CONTRACT.md` with exact schema/version, location, frame convention, and distinction from `camera.json`. Add tests for empty rows, malformed rows, ordering, and five-camera completeness.

Run:

```powershell
uv run pytest tests/test_camera_distribution.py tests/test_annotation_validator.py tests/test_task_audit.py -q
```

Expected: existing annotation behavior remains green and the new artifact tests pass.

---

### Task 7: Anchor Smoke Validation and Limitations Documentation

**Files:**
- Modify: `docs/DATA_CONTRACT.md`
- Modify: `docs/VALIDATION_AND_LIMITATIONS.md`
- Test: existing full test suite

**Interfaces:**
- Consumes: schema/resolver, UE application, and artifact behavior from Tasks 1-6.
- Produces: documented acceptance evidence and explicit limitations for C5 and actual UE verification.

- [ ] **Step 1: Run Dataset-side validation**

Run:

```powershell
uv run grf-ue task validate configs/camera_anchor_smoke_5cam.json
uv run grf-ue task resolve configs/camera_anchor_smoke_5cam.json
uv run pytest
```

Expected: task validation and resolution pass only when all five mappings are present in the Task Spec; Python tests pass without requiring GRF integration.

- [ ] **Step 2: Run live UE smoke only after C5 confirmation**

Through Unreal MCP, execute the resolved task in the real Editor Python environment. Confirm in logs and native queries:

- current level is `L_FutsalCourt`;
- five mapped Actors exist;
- each has a CineCameraComponent;
- each Sequence has the intended Camera Cut;
- read-back resolution, FOV/focal length, position, rotation, and height match the resolved profile;
- `camera_state.jsonl` contains five camera IDs for each expected frame;
- no UE Actor was automatically created.

- [ ] **Step 3: Run artifact and render checks**

After the asynchronous MRQ completion, run:

```powershell
uv run grf-ue task postprocess configs/camera_anchor_smoke_5cam.json
uv run grf-ue task audit configs/camera_anchor_smoke_5cam.json --validation-level quick
```

Do not change audit schemas in this phase. Record `camera_state.jsonl` as an additional artifact and record that current audit coverage for it is structural/manual unless a narrow existing check can be reused without changing ValidationResult semantics.

- [ ] **Step 4: Document limitations**

Add to `VALIDATION_AND_LIMITATIONS.md`:

- C5 must be confirmed in live UE Editor/MCP and is not inferred from code or binary asset names.
- This phase uses existing Actors only and never auto-creates cameras.
- Only static C1-C5 anchors are implemented.
- `camera_state.jsonl` records actual state and does not perform camera-network analysis.
- Partial, broadcast, random distribution, overlap graph, and distortion behavior are not implemented.

- [ ] **Step 5: Final verification and worktree review**

Run:

```powershell
uv run pytest
```

Expected: full non-GRF tests pass, diff check passes, only intended inner-repository files are modified, and no UE binary or plugin files are changed.

---

## Spec Coverage Review

- Current pipeline analysis: Tasks 1-2 and the existing code evidence in the design spec.
- Architecture boundary: Global Constraints and Tasks 1-2.
- Canonical IDs and mapping: Tasks 1-4.
- Fixed five anchors: Tasks 1-3 and Task 5.
- No random camera: Global Constraints and all task scopes.
- Explicit resolution/lens/FOV/position/rotation/height: Tasks 1 and 5.
- Actual `camera_state.jsonl`: Task 6.
- Existing Sequence/Camera Cut/MRQ reuse: Tasks 4-5.
- C5 live UE/MCP confirmation: Task 4 and Task 7.
- No automatic Actor creation: Global Constraints and Task 5.
- Existing validation boundaries: Task 7.
- Partial, broadcast, relationship analysis, and distortion: explicitly deferred and documented as not implemented.

## Self-Review

- No `TODO`, `TBD`, or vague implementation placeholder remains in the plan.
- Later tasks use the exact interfaces introduced by earlier tasks: `ResolvedTask.simulation`, `ue_profile.camera_mapping`, `resolve_anchor_bindings`, `normalize_camera_state`, and `write_camera_state_jsonl`.
- The plan does not require changing ValidationResult, Pipeline State, Run Manifest, Blueprint, MCP, or UE binary assets.
- C5 is a runtime prerequisite, not a code assumption or an automatically created asset.
- The plan preserves the existing `ue.sequences` pipeline while adding explicit Canonical ID mapping.
- Artifact state is actual read-back state and intentionally excludes complex relationship/coverage analysis.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-06-camera-distribution-anchors.md`.

Execution options:

1. **Subagent-Driven**: dispatch a fresh implementation/review agent per task with checkpoints.
2. **Inline Execution**: execute this plan in the current session with task-by-task verification checkpoints.
