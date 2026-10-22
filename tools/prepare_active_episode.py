"""命令行：从已归档完整轨迹准备活跃 episode，不运行 GRF 或 UE。"""

import argparse
import json
from pathlib import Path

from grf_ue_bridge.active_episode import prepare_active_episode
from grf_ue_bridge.config.resolver import load_resolved_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolved', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--selection-json', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--window-index', type=int, default=0)
    parser.add_argument('--output-json', type=Path)
    args = parser.parse_args()
    selection = json.loads(args.selection_json.read_text(encoding='utf-8'))
    if isinstance(selection, list):
        matches = [entry for entry in selection if entry.get('root_seed') == args.seed]
        if len(matches) != 1:
            parser.error('列表报告要求 --seed 且恰好匹配一项')
        selection = matches[0]
    entry = prepare_active_episode(load_resolved_task(args.resolved), args.source, selection, args.window_index)
    text = json.dumps(entry, ensure_ascii=False, indent=2) + '\n'
    if args.output_json:
        args.output_json.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
