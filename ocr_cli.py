"""Command-line entry point for OCR keyword annotation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ocr_core import PROJECT_ROOT, create_ocr, recognize_and_mark, write_image


SAMPLE_NAME = "识别并标记出“对象”.jpg"
DEFAULT_IMAGE = next(
    (
        PROJECT_ROOT / directory / SAMPLE_NAME
        for directory in ("target_img", "examples")
        if (PROJECT_ROOT / directory / SAMPLE_NAME).is_file()
    ),
    PROJECT_ROOT / "target_img" / SAMPLE_NAME,
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "对象_标记结果.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR 识别并标记图片中的关键词。")
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE, help="待识别图片路径")
    parser.add_argument("query", nargs="?", default="对象", help="要查找的文字，默认：对象")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="标记图片的输出路径")
    parser.add_argument("--report", type=Path, help="可选：将识别匹配详情写入 JSON")
    parser.add_argument("--threshold", type=int, default=85, help="模糊匹配阈值（0-100，默认 85）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        annotated, matches, line_count = recognize_and_mark(
            create_ocr(), args.image, args.query, args.threshold
        )
        if not matches:
            print(f"未找到“{args.query}”。已识别 {line_count} 行文字。")
            return 1
        write_image(args.output, annotated)
        print(f"找到 {len(matches)} 处“{args.query}”，已保存标记图片：{args.output}")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps({"query": args.query, "matches": matches}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"识别详情已保存：{args.report}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
