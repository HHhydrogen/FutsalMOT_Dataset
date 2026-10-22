"""目录搬移后修复可执行数据清单；历史运行 receipt、指标与 provenance 保持原字节。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

OLD = 'G:/FutsalMOT_Research'
ROOT = Path('D:/projects/FutsalMOT_Research')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_path(value, new_root):
    """兼容历史清单中的正反斜杠，只替换完整路径前缀。"""
    normalized=value.replace('\\','/')
    if normalized==OLD or normalized.startswith(OLD+'/'):
        return new_root.as_posix()+normalized[len(OLD):]
    return value


def rewrite_config(value, root):
    if isinstance(value,str):return replace_path(value,root)
    if isinstance(value,list):return [rewrite_config(item,root) for item in value]
    if isinstance(value,dict):return {key:rewrite_config(item,root) for key,item in value.items()}
    return value


def plan(root):
    changes=[];lists=[]
    for path in sorted((root/'datasets').rglob('dataset.yaml')):
        original=path.read_bytes()
        config=json.loads(original.decode('utf-8'))
        updated=rewrite_config(config,root)
        if config!=updated:
            changes.append((path,original,(json.dumps(updated,ensure_ascii=False,indent=2)+'\n').encode('utf-8')))
        base=Path(updated['path'])
        if base.resolve()!=path.parent.resolve():raise ValueError('dataset.yaml.path 与实际目录不一致')
        for split in ('train','val','test'):
            reference=updated.get(split)
            if not isinstance(reference,str) or not reference.endswith('.txt'):continue
            listing=Path(reference) if Path(reference).is_absolute() else base/reference
            listing.resolve().relative_to(root.resolve())
            if listing not in lists:lists.append(listing)
    for path in sorted(lists):
        original=path.read_bytes()
        old_lines=original.decode('utf-8').splitlines()
        new_lines=[replace_path(line,root) for line in old_lines]
        for line in new_lines:
            if not line.strip():continue
            image=Path(line)
            if not image.is_file():raise ValueError('迁移后图片不存在：{}'.format(image))
            # YOLO 从 images 推导 labels，路径必须同时有效。
            parts=list(image.parts)
            if 'images' not in parts:raise ValueError('图片路径不符合 YOLO images/labels 约定')
            parts[parts.index('images')]='labels'
            label=Path(*parts).with_suffix('.txt')
            if not label.is_file():raise ValueError('迁移后标签不存在：{}'.format(label))
        if new_lines!=old_lines:
            changes.append((path,original,('\n'.join(new_lines)+'\n').encode('utf-8')))
    caches=[]
    for path in sorted((root/'datasets').rglob('*.cache')):
        contents=path.read_bytes()
        if OLD.encode('utf-8') in contents or OLD.replace('/','\\').encode('utf-8') in contents:
            caches.append(path)
    return changes,caches,lists


def migrate(root, apply=False):
    root=Path(root).resolve()
    changes,caches,lists=plan(root)
    result={'root':str(root),'apply':apply,'changes':[{'path':path.relative_to(root).as_posix(),'before_sha256':sha(old),'after_sha256':sha(new)} for path,old,new in changes],
            'cache_quarantine':[path.relative_to(root).as_posix() for path in caches],
            'validated_lists':[path.relative_to(root).as_posix() for path in lists],
            'history_policy':'不改 runtime/download receipts、provenance、指标、训练/评测历史参数和已有哈希。'}
    if apply and (changes or caches):
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        archive=root/'path_migrations'/stamp
        archive.mkdir(parents=True,exist_ok=False)
        for path,old,new in changes:
            backup=archive/'backup'/path.relative_to(root);backup.parent.mkdir(parents=True,exist_ok=True)
            backup.write_bytes(old)
            temporary=path.with_name(path.name+'.rebase_tmp')
            with temporary.open('xb') as stream:stream.write(new)
            temporary.replace(path)
        for path in caches:
            backup=archive/'label_caches'/path.relative_to(root);backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(path),str(backup))
        result['archive']=str(archive)
        (archive/'migration_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        # 重读全部生效配置与清单，确认不再含旧路径且图片和标签存在。
        if plan(root)[0]:raise RuntimeError('迁移后仍存在待替换数据路径')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    print(json.dumps(migrate(args.root,args.apply),ensure_ascii=False,indent=2))
