"""串行剖析真实 EXR，逐帧核对已验收 golden；只写独立性能证据目录。"""
import argparse
import cProfile
import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pstats
import time

import numpy as np
import OpenEXR
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module-source', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.module_source:
        spec = importlib.util.spec_from_file_location('profiled_cryptomatte', args.module_source)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    else:
        from grf_ue_bridge import cryptomatte as module
    source = Path(module.__file__)
    mapping_path = Path(__file__).resolve().parents[1] / 'ue/actor_mapping.example.json'
    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    plan = module._resolve_actor_plan(mapping)
    args.output.mkdir(parents=True, exist_ok=False)
    profile = cProfile.Profile()
    records = []
    frames = sorted(args.episode.glob('CineCam_*/render_mask/*.exr'))
    if args.limit:frames = frames[:args.limit]
    started = time.perf_counter()
    profile.enable()
    for path in frames:
        camera = path.parent.parent
        frame_index = int(path.stem) + 1
        golden_path = camera / 'mask' / ('{:06d}.png'.format(frame_index))
        start = time.perf_counter()
        file = OpenEXR.File(str(path))
        read_at = time.perf_counter()
        manifest = next(json.loads(value) for key,value in file.header().items()
                        if key.startswith('cryptomatte/') and key.endswith('/manifest'))
        layers = file.channels()
        ids,layer_name = module.select_cryptomatte_id_channel(manifest,layers)
        select_at = time.perf_counter()
        mask,counts = module.build_mask(manifest,ids,mapping,plan)
        build_at = time.perf_counter()
        output_path = args.output / camera.name / golden_path.name
        module.save_mask_png(mask,output_path)
        save_at = time.perf_counter()
        with Image.open(golden_path) as image:golden = np.array(image)
        with Image.open(output_path) as image:roundtrip = np.array(image)
        mismatch = int(np.count_nonzero(mask != golden))
        assert mismatch == 0 and np.array_equal(mask,roundtrip)
        record = {'camera':camera.name,'frame_index':frame_index,'exr':str(path),
                  'shape':list(mask.shape),'manifest_entries':len(manifest),'layers':list(layers),
                  'selected_layer':layer_name,'openexr_seconds':read_at-start,
                  'select_seconds':select_at-read_at,'build_seconds':build_at-select_at,
                  'save_seconds':save_at-build_at,'decode_seconds':save_at-start,
                  'different_pixels':mismatch,'pixel_counts':counts,
                  'mask_pixel_sha256':hashlib.sha256(mask.tobytes()).hexdigest()}
        records.append(record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
        del file,layers,ids,mask,golden,roundtrip
        gc.collect()
    profile.disable()
    profile.dump_stats(str(args.output/'profile.pstats'))
    stream = io.StringIO()
    pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(35)
    (args.output/'profile.txt').write_text(stream.getvalue(),encoding='utf-8')
    result = {'module':str(source),'module_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'numpy_version':np.__version__,'openexr_version':OpenEXR.__version__,
              'frames':records,'total_decode_seconds':sum(row['decode_seconds'] for row in records),
              'wall_seconds':time.perf_counter()-started,'all_pixels_equal':True}
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PROFILE_COMPLETE',result['total_decode_seconds'],len(records),flush=True)


if __name__ == '__main__':main()
