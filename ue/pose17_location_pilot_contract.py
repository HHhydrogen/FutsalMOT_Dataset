"""明确四seed位置监督试产范围；只授权采集，不提升可见性或训练准入。"""
import re

PILOT_WINDOWS = {1042: (100, 220, "train"), 1044: (307, 427, "train"),
                 1048: (100, 220, "train"), 1102: (100, 220, "synthetic_dev")}


def validate_location_pilot_contract(plan):
    """普通小样返回false；出现任一pilot字段即要求完整、精确的显式契约。"""
    if "location_pilot_contract" not in plan and "location_pilot" not in plan:
        return False
    pilot = plan.get("location_pilot")
    if plan.get("location_pilot_contract") is not True or not isinstance(pilot, dict):
        raise ValueError("位置试产需要显式opt-in及独立契约")
    seed = pilot.get("seed")
    if type(seed) is not int or seed not in PILOT_WINDOWS:
        raise ValueError("位置试产仅当前冻结四seed")
    start, end, split = PILOT_WINDOWS[seed]
    expected = dict(schema="futsalmot_pose17_location_pilot_v1", batch_id="location_pilot_4seed_a01",
                    seed=seed, split=split, source_frame_range=[start, end],
                    location_supervision=True, visibility_supervision=False, formal_pose_release=False)
    if pilot != expected or any(type(pilot.get(k)) is not bool for k in
                              ("location_supervision", "visibility_supervision", "formal_pose_release")):
        raise ValueError("位置试产范围/split/监督边界不符合冻结契约")
    motion = plan.get("source_motion", {})
    if (plan.get("diagnostic_scope") != "pose17_gpu_render_cpu_lbs_joint_diagnostic"
            or plan.get("frame_count") != 120 or type(plan.get("frame_count")) is not int
            or plan.get("start_frame") != start or type(plan.get("start_frame")) is not int
            or plan.get("review_frame_indices") != [60]
            or any(type(x) is not int for x in plan["review_frame_indices"])
            or motion.get("root_seed") != seed or type(motion.get("root_seed")) is not int
            or motion.get("fps") != 10 or type(motion.get("fps")) is not int
            or motion.get("selected_source_frame_range") != [start, end]
            or motion.get("context_source_frame_range") != [0, end]
            or motion.get("sequence_frame_equals_source_step") is not True
            or "location_pilot" not in plan.get("source_scope", "")
            or "diagnostic" not in plan.get("source_scope", "")):
        raise ValueError("位置试产必须绑定真实连续来源、120帧及同源时基")
    for key in ("source_frames_sha256", "source_meta_sha256"):
        if not isinstance(motion.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", motion[key]) is None:
            raise ValueError("位置试产缺冻结源SHA")
    return True
