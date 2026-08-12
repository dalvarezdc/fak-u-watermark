"""Command-line interface for quick testing.

Examples:
  faku text analyze "Hello world" --preset kirchenbauer_default
  faku text highlight sample.txt -o out.html
  faku image exif photo.jpg
  faku image strip photo.jpg -o clean.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGES = _ROOT / "packages"
for p in (_PACKAGES, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def cmd_text_analyze(args: argparse.Namespace) -> int:
    from watermark_core.analyzer import WatermarkAnalyzer
    from watermark_core.visualization import tokens_to_html

    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text:
        print("No text provided.", file=sys.stderr)
        return 1

    analyzer = WatermarkAnalyzer(
        scheme=args.scheme,
        gamma=args.gamma,
        key=args.key,
        tokenizer_name=args.tokenizer,
        threshold=args.threshold,
        preset=args.preset,
    )
    result = analyzer.analyze(text)
    stats = result.statistics.to_dict()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"Verdict:  {stats['verdict_label']}")
        print(f"Z-score:  {stats['z_score']:.4f}")
        print(f"Green:    {stats['green_count']}/{stats['total_tokens']} ({stats['green_fraction']:.2%})")
        print(f"P-value:  {stats['p_value']:.4e}")
        print(f"Scheme:   {stats['scheme']}  gamma={stats['gamma']}")
        if args.html:
            html_out = tokens_to_html(result.tokens, show_highlights=True)
            out = Path(args.html)
            out.write_text(html_out, encoding="utf-8")
            print(f"HTML written to {out}")
    return 0


def cmd_text_highlight(args: argparse.Namespace) -> int:
    from watermark_core.analyzer import WatermarkAnalyzer
    from watermark_core.visualization import tokens_to_annotated_document

    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if not text:
        print("No text provided.", file=sys.stderr)
        return 1

    analyzer = WatermarkAnalyzer(
        scheme=args.scheme,
        gamma=args.gamma,
        key=args.key,
        tokenizer_name=args.tokenizer,
        preset=args.preset,
    )
    result = analyzer.analyze(text)
    doc = tokens_to_annotated_document(
        result.tokens,
        statistics=result.statistics.to_dict(),
    )
    out = Path(args.output or "highlighted.html")
    out.write_text(doc, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def cmd_image_exif(args: argparse.Namespace) -> int:
    from image_tools.exif import read_exif

    meta = read_exif(args.file)
    print(json.dumps(meta, indent=2, default=str, ensure_ascii=False))
    return 0


def cmd_image_strip(args: argparse.Namespace) -> int:
    from image_tools.exif import write_image_without_exif

    dest = args.output or str(Path(args.file).with_stem(Path(args.file).stem + "_stripped"))
    path = write_image_without_exif(args.file, dest)
    print(f"Wrote cleaned image to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="faku",
        description="fak-u-watermark — detect, highlight, neutralize",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # text
    text = sub.add_parser("text", help="Text watermark tools")
    text_sub = text.add_subparsers(dest="text_cmd", required=True)

    analyze = text_sub.add_parser("analyze", help="Analyze text for watermark signal")
    analyze.add_argument("text", nargs="?", default="", help="Text to analyze")
    analyze.add_argument("-f", "--file", help="Read text from file")
    analyze.add_argument("--scheme", default="kgw")
    analyze.add_argument("--gamma", type=float, default=0.25)
    analyze.add_argument("--key", default=None)
    analyze.add_argument("--tokenizer", default="gpt2")
    analyze.add_argument("--threshold", type=float, default=4.0)
    analyze.add_argument("--preset", default=None)
    analyze.add_argument("--json", action="store_true")
    analyze.add_argument("--html", help="Write highlighted HTML to path")
    analyze.set_defaults(func=cmd_text_analyze)

    hl = text_sub.add_parser("highlight", help="Export annotated HTML")
    hl.add_argument("text", nargs="?", default="")
    hl.add_argument("-f", "--file")
    hl.add_argument("-o", "--output", default="highlighted.html")
    hl.add_argument("--scheme", default="kgw")
    hl.add_argument("--gamma", type=float, default=0.25)
    hl.add_argument("--key", default=None)
    hl.add_argument("--tokenizer", default="gpt2")
    hl.add_argument("--preset", default="kirchenbauer_default")
    hl.set_defaults(func=cmd_text_highlight)

    # image
    image = sub.add_parser("image", help="Image EXIF tools")
    image_sub = image.add_subparsers(dest="image_cmd", required=True)

    exif = image_sub.add_parser("exif", help="Print EXIF / metadata")
    exif.add_argument("file")
    exif.set_defaults(func=cmd_image_exif)

    strip = image_sub.add_parser("strip", help="Strip all metadata")
    strip.add_argument("file")
    strip.add_argument("-o", "--output")
    strip.set_defaults(func=cmd_image_strip)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
