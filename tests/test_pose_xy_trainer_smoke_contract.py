"""CPU兼容smoke的toy夹具边界测试；不运行训练。"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('smoke_pose_xy_trainer_cpu',Path(__file__).resolve().parents[1]/'tools/smoke_pose_xy_trainer_cpu.py')
TOOL=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class ToySmokeContractTests(unittest.TestCase):
    def test_fixture_has_four_images_and_xy_only_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'toy'
            config,hashes=TOOL.make_fixture(root)
            self.assertEqual(len(hashes),9)
            self.assertIn('kpt_shape: [17, 2]',config.read_text(encoding='utf-8'))
            for split in ('train','val'):
                self.assertEqual(len(list((root/'images'/split).glob('*.png'))),2)
                for label in (root/'labels'/split).glob('*.txt'):
                    row=TOOL.validate_toy_label(label.read_text())
                    self.assertEqual(len(row),39)
                    self.assertEqual(row[5:],list(n for p in TOOL.TOY_POINTS for n in p))
            for relative,digest in hashes.items():self.assertEqual(TOOL.sha(root/relative),digest)

    def test_fixture_never_overwrites_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):TOOL.make_fixture(Path(temporary))

    def test_rejects_visibility_triplets_nonfinite_and_outside(self):
        valid='0 0.5 0.5 0.8 0.8 '+' '.join(str(v) for p in TOOL.TOY_POINTS for v in p)
        self.assertEqual(len(TOOL.validate_toy_label(valid)),39)
        with self.assertRaises(ValueError):TOOL.validate_toy_label(valid+' 2')
        for bad in ('nan','inf','0','1','-0.1'):
            row=valid.split();row[-1]=bad
            with self.assertRaises(ValueError):TOOL.validate_toy_label(' '.join(row))


if __name__=='__main__':unittest.main()
