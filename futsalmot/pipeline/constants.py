from __future__ import annotations


TEMPLATE_NAMES = {
    1: "solo_dribble_shot_4v4",
    2: "dribble_pass_receive_4v4",
    3: "pass_receive_dribble_shot_4v4",
}

WINDOWS_PIPELINE_SCRIPTS = {
    "generate_episode": "futsalmot/scripts/generate_random_episode.py",
    "validate_episode": "futsalmot/scripts/validate_episode.py",
    "compile_trajectory": "futsalmot/scripts/compile_trajectory.py",
    "enhance_trajectory": "futsalmot/scripts/enhance_trajectory.py",
    "validate_trajectory": "futsalmot/scripts/validate_trajectory.py",
    "event_annotations": "futsalmot/scripts/generate_event_annotations.py",
}

UE_PIPELINE_SCRIPTS = {
    "setup_8_players": "futsalmot/scripts/ue_setup_8_players.py",
    "preflight": "futsalmot/scripts/ue_preflight.py",
    "build_sequences": "futsalmot/scripts/ue_build_sequences.py",
    "scan_animations": "futsalmot/scripts/ue_scan_animations.py",
}
