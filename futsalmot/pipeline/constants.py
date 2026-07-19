from __future__ import annotations


TEMPLATE_NAMES = {
    1: "solo_dribble_shot_4v4",
    2: "dribble_pass_receive_4v4",
    3: "pass_receive_dribble_shot_4v4",
}

WINDOWS_PIPELINE_SCRIPTS = {
    "generate_episode": "11_generate_random_episode.py",
    "validate_episode": "10_validate_episode.py",
    "compile_trajectory": "12_compile_trajectory.py",
    "enhance_trajectory": "13_enhance_trajectory.py",
    "validate_trajectory": "14_validate_trajectory.py",
    "event_annotations": "31_generate_event_annotations.py",
}

UE_PIPELINE_SCRIPTS = {
    "setup_8_players": "23_ue_setup_8_players.py",
    "preflight": "21_preflight.py",
    "build_sequences": "20_build_sequences.py",
    "scan_animations": "22_scan_animations.py",
}
