"""dataset_export 序列化测试。"""

from dataset_export import (
    build_mot_gt,
    build_seqinfo,
    format_mot_line,
    mot_int_bbox,
)


def _obj(track_id, cls, in_frame, xyxy, raw_xywh=None, clipped_xywh=None):
    return {
        "in_frame": in_frame,
        "class": cls,
        "track_id": track_id,
        "bbox_xyxy": xyxy,
        "raw_bbox_xywh": raw_xywh if raw_xywh is not None else xyxy,
        "bbox_xywh": clipped_xywh if clipped_xywh is not None else xyxy,
    }


class TestMotIntBbox:
    def test_basic(self):
        assert mot_int_bbox(0.4, 0.6, 2.4, 3.6, 1920, 1080) == (0, 0, 3, 4)

    def test_clipped_to_image(self):
        # 越界的输入先被裁剪到图像内
        assert mot_int_bbox(-5.0, -5.0, 1918.5, 1078.5, 1920, 1080) == (0, 0, 1919, 1079)

    def test_minimum_size(self):
        assert mot_int_bbox(100.0, 100.0, 100.4, 100.4, 1920, 1080) == (100, 100, 1, 1)

    def test_at_right_edge(self):
        assert mot_int_bbox(1918.5, 100.0, 1920.0, 200.0, 1920, 1080) == (1918, 100, 2, 100)


class TestMotSerialization:
    def test_format_mot_line(self):
        line = format_mot_line(1, 3, 100, 200, 30, 80, 1, 1, 1.0)
        assert line == "1,3,100,200,30,80,1,1,1.00"

    def test_build_mot_gt_exact(self):
        objects = [
            [
                _obj(1, "player", True, [10.0, 20.0, 110.0, 220.0]),
                _obj(2, "player", True, [300.0, 400.0, 350.0, 480.0]),
            ],
            [
                _obj(1, "player", True, [11.0, 21.0, 111.0, 221.0]),
                _obj(2, "player", False, None),  # 不在画面：不输出到 MOT
            ],
        ]
        rows = build_mot_gt(objects, 1920, 1080, include_ball=False)
        assert rows == [
            "1,1,10,20,100,200,1,1,1.00",
            "1,2,300,400,50,80,1,1,1.00",
            "2,1,11,21,100,200,1,1,1.00",
        ]

    def test_include_ball(self):
        objects = [[
            _obj(1, "player", True, [10.0, 20.0, 110.0, 220.0]),
            _obj(100, "ball", True, [500.0, 540.0, 520.0, 560.0]),
        ]]
        rows_no_ball = build_mot_gt(objects, 1920, 1080, include_ball=False)
        assert len(rows_no_ball) == 1
        rows_with_ball = build_mot_gt(objects, 1920, 1080, include_ball=True)
        assert rows_with_ball == [
            "1,1,10,20,100,200,1,1,1.00",
            "1,100,500,540,20,20,1,100,1.00",
        ]

    def test_truncation_visibility(self):
        objects = [[
            # 裁剪后面积 50*100=5000，原始面积 100*100=10000 → 0.50
            _obj(
                1, "player", True,
                xyxy=[0.0, 0.0, 50.0, 100.0],
                raw_xywh=[-50.0, 0.0, 100.0, 100.0],
                clipped_xywh=[0.0, 0.0, 50.0, 100.0],
            ),
        ]]
        rows = build_mot_gt(
            objects, 1920, 1080, include_ball=False, visibility_mode="truncation"
        )
        assert rows[0] == "1,1,0,0,50,100,1,1,0.50"

    def test_visibility_unoccluded_default(self):
        objects = [[
            _obj(1, "player", True, [10.0, 20.0, 110.0, 220.0]),
        ]]
        rows = build_mot_gt(objects, 1920, 1080, include_ball=False)
        assert rows[0].endswith("1,1.00")


class TestSeqinfo:
    def test_seqinfo_exact(self):
        text = build_seqinfo("episode_0001", "img1", 30, 300, 1920, 1080)
        assert text == (
            "[Sequence]\n"
            "name=episode_0001\n"
            "imDir=img1\n"
            "frameRate=30\n"
            "seqLength=300\n"
            "imWidth=1920\n"
            "imHeight=1080\n"
            "imExt=.png\n"
        )
