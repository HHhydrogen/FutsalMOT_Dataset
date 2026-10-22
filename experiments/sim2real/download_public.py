# -*- coding: utf-8 -*-
"""从官方 SoccerNet ZIP 读取目录并按序列下载，避免下载整个大包。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath

REPO = "SoccerNet/SN-Tracking-2023"
REVISION = "3777f456d908570e10b19e7354744640e77c351e"
SIZES = {"train": 9575523385, "test": 8711601474}


class HTTPRangeReader(io.RawIOBase):
    """仅接收正确的 206 range 响应；不允许服务器静默返回整个 ZIP。"""

    def __init__(self, url, size, budget=8_000_000_000):
        self.url, self.size, self.budget = url, size, budget
        self.position = self.downloaded = self.requests = 0
        self.cache_start, self.cache = 0, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if not 0 <= self.position <= self.size:
            raise ValueError("ZIP seek 越界")
        return self.position

    def read(self, length=-1):
        if length < 0:
            length = self.size - self.position
        length = min(length, self.size - self.position)
        if not length:
            return b""
        start = self.position
        if self.cache_start <= start and start + length <= self.cache_start + len(self.cache):
            offset = start - self.cache_start
            self.position += length
            return self.cache[offset:offset + length]
        end = min(self.size - 1, start + max(length, 65536) - 1)
        if self.downloaded + end - start + 1 > self.budget:
            raise RuntimeError("已达到下载预算，停止且保留已下载数据")
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=90) as response:
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError(f"服务器未正确支持 HTTP range: {response.status} / {response.headers.get('Content-Range')}")
            data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise RuntimeError("HTTP range 长度不匹配")
        self.downloaded += len(data)
        self.requests += 1
        self.cache_start, self.cache = start, data
        self.position += length
        return data[:length]


def open_archive(split, budget):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{split}.zip?download=true"
    reader = HTTPRangeReader(url, SIZES[split], budget)
    return reader, zipfile.ZipFile(reader)


def sequence_of(name):
    match = re.search(r"(?:^|/)(SNMOT-\d+)(?:/|$)", name)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=list(SIZES), default="train")
    parser.add_argument("--output", type=Path, default=Path("D:/projects/FutsalMOT_Research/public/soccernet_tracking"))
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--max-frame", type=int, default=150)
    parser.add_argument("--budget", type=int, default=2_000_000_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reader, archive = open_archive(args.split, args.budget)
    infos = archive.infolist()
    sequences = sorted({sequence_of(x.filename) for x in infos} - {None})
    listing = [{"name": x.filename, "size": x.file_size, "compressed_size": x.compress_size, "crc32": f"{x.CRC:08x}"} for x in infos]
    (args.output / f"{args.split}_zip_index.json").write_text(json.dumps({"repo": REPO, "revision": REVISION, "archive_size": SIZES[args.split], "sequences": sequences, "members": listing}, indent=2), encoding="utf-8")
    print("ARCHIVE_INDEX", json.dumps({"split": args.split, "sequences": sequences, "members": len(infos), "downloaded": reader.downloaded}), flush=True)
    if args.list_only:
        return
    if not args.sequences:
        raise ValueError("下载前必须显式指定 --sequences，禁止默认整包")
    if not set(args.sequences) <= set(sequences):
        raise ValueError("请求的序列不在官方 split 中")
    records = []
    selected=[]
    for info in infos:
        if info.is_dir() or sequence_of(info.filename) not in args.sequences:
            continue
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("ZIP 中包含不安全路径")
        frame = re.search(r"/img1/(\d+)\.(?:jpg|png)$", info.filename)
        if frame and int(frame.group(1)) > args.max_frame:
            continue
        selected.append(info)
    for sequence in args.sequences:
        group=[x for x in selected if sequence_of(x.filename)==sequence]
        start=min(x.header_offset for x in group)
        end=max(x.header_offset+30+len(x.filename.encode('utf-8'))+len(x.extra)+x.compress_size for x in group)
        reader.seek(start)
        reader.read(end-start)
        print('SEQUENCE_PREFETCH',sequence,end-start,flush=True)
        for info in group:
            destination = args.output / Path(*PurePosixPath(info.filename).parts)
            destination.resolve().relative_to(args.output.resolve())
            if destination.is_file() and destination.stat().st_size == info.file_size:
                data = destination.read_bytes()
            else:
                data = archive.read(info)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            if zlib.crc32(data)&0xffffffff != info.CRC:
                raise RuntimeError(f'已下载内容的 CRC 与官方 ZIP 不一致：{info.filename}')
            records.append({"path": info.filename, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            if len(records) % 50 == 0:
                print("DOWNLOAD_PROGRESS", len(records), reader.downloaded, flush=True)
    receipt = {"repo": REPO, "revision": REVISION, "official_split": args.split, "sequences": args.sequences, "frame_limit": args.max_frame, "downloaded_bytes": reader.downloaded, "http_requests": reader.requests, "timestamp_unix": time.time(), "files": records}
    (args.output / f"{args.split}_download_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("DOWNLOAD_COMPLETE", len(records), reader.downloaded, flush=True)


if __name__ == "__main__":
    main()
