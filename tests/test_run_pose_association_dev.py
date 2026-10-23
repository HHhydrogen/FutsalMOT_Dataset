"""同schema自绘数值检测的CPU接线；不运行真实缓存/模型推理或TrackEval。"""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments/sim2real"));sys.path.insert(0,str(ROOT/"src"))
SPEC=importlib.util.spec_from_file_location("association_dev_test",ROOT/"experiments/sim2real/run_pose_association_dev.py")
TOOL=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(TOOL)


def test_lambda0_cli_core_writes_three_complete_mot_sequences(tmp_path,monkeypatch):
    pytest.importorskip("lap");pytest.importorskip("ultralytics")
    frames=[]
    for sequence in TOOL.SEQUENCES:
        for index in range(150):
            key="%s:%06d"%(sequence,index+1)
            frames.append(dict(sequence_id=sequence,camera_id="broadcast",frame_index=index,frame_number=index+1,
                width=1920,height=1080,detections=[dict(detection_id=key+":000000",bbox_xywh_px=[100.+index*.1,100.,40.,80.],score=.9,class_id=0)]))
    manifest={"source_manifest":{"path":"toy_source.json","sha256":"c"*64}}
    monkeypatch.setattr(TOOL,"load_fixed_cache",lambda *a:(manifest,frames))
    monkeypatch.setattr(TOOL,"output_path",lambda p:Path(p))
    config=ROOT/"experiments/sim2real/bytetrack_fixed.yaml"
    protocol=dict(schema="futsalmot_pose_association_protocol_candidate_v1",source_manifest=manifest["source_manifest"],test_split_used=False,
        tracking={"tracker_config":{"path":str(config),"sha256":TOOL.sha(config)}},pose_cost_candidate={"lambda_candidates":[0.,.1,.25]})
    path=tmp_path/"protocol.json";path.write_text(json.dumps(protocol),encoding="utf-8")
    output=tmp_path/"candidate"
    result=TOOL.run(tmp_path/"cache.json","a"*64,path,TOOL.sha(path),output)
    assert result["gpu_used"] is False and result["GT_read"] is False and result["metrics_evaluated"] is False
    assert (output/"receipt.json").exists()
    for sequence in TOOL.SEQUENCES:
        current=result["sequences"][sequence]
        assert current["frames"]==current["tracked_rows"]==150 and current["lambda0_equivalence"]["passed"]
        lines=(output/"data"/(sequence+".txt")).read_text().splitlines()
        assert len(lines)==150 and lines[0].startswith("1,1,") and lines[-1].startswith("150,1,")
    with pytest.raises(ValueError):TOOL.run(tmp_path/"cache.json","a"*64,path,"0"*64,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()
