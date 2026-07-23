#!/usr/bin/env python3
"""扫描当前目录下所有 YYYY-MM-DD.html 周报，提取其 <head> 中的
report-meta 元数据，按日期倒序写入 reports.json。

新增一份周报后运行：python3 build_index.py
"""
import json
import re
from pathlib import Path

FILE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.html$")
META_PATTERN = re.compile(
    r'<script[^>]*id=["\']report-meta["\'][^>]*>([\s\S]*?)</script>', re.I
)


def main() -> None:
    here = Path(__file__).resolve().parent
    metas = []
    for f in sorted(here.glob("*.html")):
        if not FILE_PATTERN.match(f.name):
            continue
        html = f.read_text(encoding="utf-8")
        m = META_PATTERN.search(html)
        if not m:
            print(f"跳过 {f.name}：未找到 report-meta")
            continue
        meta = json.loads(m.group(1).strip())
        meta["file"] = f.name
        meta.setdefault("date", f.stem)
        metas.append(meta)

    metas.sort(key=lambda x: x["date"], reverse=True)
    out = here / "reports.json"
    out.write_text(
        json.dumps(metas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已写入 {out.name}，共 {len(metas)} 份报告。")


if __name__ == "__main__":
    main()
