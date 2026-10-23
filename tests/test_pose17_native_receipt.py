"""验证native落盘数据读取，不创建或解引用任何Unreal对象。"""

import pytest

from pose17_native_receipt import normalize_native_receipt, vector


def test_native_ustruct_names_and_vectors_preserve_unknown():
    raw = {"schema": "futsalmot_pose17_native_receipt_v1", "records": [{"entityId": "L0", "bSampleSucceeded": True,
        "cameraWorldLocationM": {"x": 0, "y": 1, "z": 2}, "imageSize": "X=3840 Y=2160", "sample": {
          "worldPointsM": ["X=1.5 Y=2.5 Z=3.5"] * 17, "keypointNames": [str(index) for index in range(17)],
          "bMRQSynchronizationVerified": False}}]}
    record = normalize_native_receipt(raw)["records"][0]
    assert record["entity_id"] == "L0"
    assert record["world_m"][0] == [1.5, 2.5, 3.5]
    assert record["visibility_state"] == ["unknown"] * 17
    assert record["image_size"] == [3840, 2160]


@pytest.mark.parametrize("value", ["X=1 Y=2", [1, float("nan"), 0], "X=bad Y=2 Z=3"])
def test_invalid_vector_rejected(value):
    with pytest.raises(ValueError):
        vector(value)


def test_does_not_accept_sync_claim_or_twelve_point_success():
    for count, verified in ((12, False), (17, True)):
        raw = {"schema": "futsalmot_pose17_native_receipt_v1", "records": [{"sampleSucceeded": True, "sample": {
          "worldPointsM": [[1, 2, 3]] * count, "keypointNames": ["point"] * count, "mrqSynchronizationVerified": verified}}]}
        with pytest.raises(ValueError):
            normalize_native_receipt(raw)
