"""已真实读取生产角色身份的精确路径白名单；不按名字或prefix猜测资产。"""

ORIGINAL_VISUAL_MESH = "/Game/FutsalMOT/Characters/MPFB/Players/Player_003/MPFB_Player_003.MPFB_Player_003"
CALIBRATION_VISUAL_MESHES = {
    4: "/Game/FutsalMOT/Characters/MPFB/Players/Player_004/Mesh/MPFB_Player_004.MPFB_Player_004",
    5: "/Game/FutsalMOT/Characters/MPFB/Players/Player_005/Mesh/MPFB_Player_005.MPFB_Player_005",
    6: "/Game/FutsalMOT/Characters/MPFB/Players/Player_006/Mesh/MPFB_Player_006.MPFB_Player_006",
    7: "/Game/FutsalMOT/Characters/MPFB/Players/Player_007/Mesh/MPFB_Player_007.MPFB_Player_007",
    8: "/Game/FutsalMOT/Characters/MPFB/Players/Player_008/Mesh/MPFB_Player_008.MPFB_Player_008",
    9: "/Game/FutsalMOT/Characters/MPFB/Players/Player_009/Mesh/MPFB_Player_009.MPFB_Player_009",
}
JOINT_VISUAL_IDS = {"L0":3,"L1":4,"L2":5,"L3":6,"L4":7,"R0":8,"R1":9,"R2":3,"R3":4,"R4":5}


def production_visual_mesh(player_id):
    if type(player_id) is int and player_id==3:
        return ORIGINAL_VISUAL_MESH
    return calibration_visual_mesh(player_id)


def calibration_visual_mesh(player_id):
    if type(player_id) is not int or player_id not in CALIBRATION_VISUAL_MESHES:
        raise ValueError("calibration_player_id必须为显式整数4..9")
    return CALIBRATION_VISUAL_MESHES[player_id]
