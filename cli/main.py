"""Command-line interface for fak-u-watermark.

Examples:
  faku text analyze "Hello world" --preset kirchenbauer_default
  faku text batch ./docs/*.txt --json
  faku text neutralize -f in.txt --style subtle
  faku text targeted -f in.txt --key 15485863
  faku text adaptive -f in.txt --max-rounds 3
  faku settings show
  faku settings set --api-key sk-... --base-url https://api.openai.com/v1 --model gpt-4o-mini
  faku image exif photo.jpg
  faku image c2pa photo.jpg
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


def _read_text(args: argparse.Namespace) -> str:
    if getattr(args, "file", None):
        return Path(args.file).read_text(encoding="utf-8")
    return getattr(args, "text", "") or ""


def _analyzer_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        scheme=getattr(args, "scheme", None),
        gamma=getattr(args, "gamma", None),
        key=getattr(args, "key", None),
        tokenizer_name=getattr(args, "tokenizer", None),
        threshold=getattr(args, "threshold", None),
        preset=getattr(args, "preset", None),
    )


def cmd_text_analyze(args: argparse.Namespace) -> int:
    from watermark_core.analyzer import WatermarkAnalyzer
    from watermark_core.density import density_summary, density_to_html, sliding_window_density
    from watermark_core.visualization import tokens_to_html

    text = _read_text(args)
    if not text:
        print("No text provided.", file=sys.stderr)
        return 1

    analyzer = WatermarkAnalyzer(**_analyzer_kwargs(args))
    result = analyzer.analyze(text)
    stats = result.statistics.to_dict()
    points = sliding_window_density(result.tokens, window=args.density_window, gamma=result.gamma)

    if args.json:
        out = result.to_dict()
        out["density_summary"] = density_summary(points)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"Verdict:  {stats['verdict_label']}")
        print(f"Z-score:  {stats['z_score']:.4f}")
        print(f"Green:    {stats['green_count']}/{stats['total_tokens']} ({stats['green_fraction']:.2%})")
        print(f"P-value:  {stats['p_value']:.4e}")
        print(f"Scheme:   {stats['scheme']}  gamma={stats['gamma']}")
        ds = density_summary(points)
        print(f"Density:  mean={ds['mean_fraction']:.2%} max={ds['max_fraction']:.2%} hot={ds['hot_spans']}")
        if args.html:
            html_out = tokens_to_html(result.tokens, show_highlights=True)
            heat = density_to_html(points)
            Path(args.html).write_text(html_out + "\n<hr/>\n" + heat, encoding="utf-8")
            print(f"HTML written to {args.html}")
    return 0


def cmd_text_highlight(args: argparse.Namespace) -> int:
    from watermark_core.analyzer import WatermarkAnalyzer
    from watermark_core.density import density_to_html, sliding_window_density
    from watermark_core.visualization import tokens_to_annotated_document

    text = _read_text(args)
    if not text:
        print("No text provided.", file=sys.stderr)
        return 1

    analyzer = WatermarkAnalyzer(**_analyzer_kwargs(args))
    result = analyzer.analyze(text)
    points = sliding_window_density(result.tokens, gamma=result.gamma)
    doc = tokens_to_annotated_document(
        result.tokens,
        statistics=result.statistics.to_dict(),
    )
    # Append heatmap section
    heat = density_to_html(points)
    doc = doc.replace("</body>", f"<h2>Density heatmap</h2>{heat}</body>")
    out = Path(args.output or "highlighted.html")
    out.write_text(doc, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def cmd_text_batch(args: argparse.Namespace) -> int:
    from watermark_core.batch import analyze_batch_files, batch_to_markdown

    paths = list(args.files or [])
    if not paths:
        print("Provide one or more files.", file=sys.stderr)
        return 1
    results = analyze_batch_files(paths, **_analyzer_kwargs(args))
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
    else:
        print(batch_to_markdown(results))
    return 0


def cmd_text_neutralize(args: argparse.Namespace) -> int:
    from watermark_core.neutralize import NeutralizeConfig, neutralize_sync

    text = _read_text(args)
    if not text:
        print("No text provided.", file=sys.stderr)
        return 1
    config = NeutralizeConfig.from_env(style=args.style)
    if args.api_key:
        config.api_key = args.api_key
    if args.base_url:
        config.base_url = args.base_url
    if args.model:
        config.model = args.model
    result = neutralize_sync(text, config)
    if not result.success:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(result.cleaned, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(result.cleaned)
    return 0


def cmd_text_targeted(args: argparse.Namespace) -> int:
    from watermark_core.analyzer import WatermarkAnalyzer
    from watermark_core.targeted import neutralize_targeted

    text = _read_text(args)
    if not text:
        print("No text provided.", file=sys.stderr)
        return 1
    analyzer = WatermarkAnalyzer(**_analyzer_kwargs(args))
    result = neutralize_targeted(text, analyzer=analyzer)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.notes)
        if args.output:
            Path(args.output).write_text(result.cleaned, encoding="utf-8")
            print(f"Wrote {args.output}")
        else:
            print(result.cleaned)
    return 0


def cmd_text_adaptive(args: argparse.Namespace) -> int:
    from watermark_core.adaptive import neutralize_adaptive
    from watermark_core.analyzer import WatermarkAnalyzer
    from watermark_core.neutralize import NeutralizeConfig

    text = _read_text(args)
    if not text:
        print("No text provided.", file=sys.stderr)
        return 1
    analyzer = WatermarkAnalyzer(**_analyzer_kwargs(args))
    config = NeutralizeConfig.from_env(style=args.style)
    if args.api_key:
        config.api_key = args.api_key
    if getattr(args, "base_url", None):
        config.base_url = args.base_url
    if getattr(args, "model", None):
        config.model = args.model
    result = neutralize_adaptive(
        text,
        analyzer=analyzer,
        config=config,
        max_rounds=args.max_rounds,
        target_z=args.target_z,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"rounds={result.rounds} z_scores={result.z_scores} success={result.success}")
        if result.error:
            print(f"note: {result.error}", file=sys.stderr)
        if args.output:
            Path(args.output).write_text(result.cleaned, encoding="utf-8")
            print(f"Wrote {args.output}")
        else:
            print(result.cleaned)
    return 0 if result.success else 1


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


def cmd_image_c2pa(args: argparse.Namespace) -> int:
    from image_tools.c2pa_tools import detect_c2pa, strip_c2pa

    if args.strip:
        cleaned, report = strip_c2pa(args.file)
        dest = args.output or str(Path(args.file).with_stem(Path(args.file).stem + "_noc2pa"))
        Path(dest).write_bytes(cleaned)
        print(json.dumps(report.to_dict(), indent=2))
        print(f"Wrote {dest}")
    else:
        print(json.dumps(detect_c2pa(args.file).to_dict(), indent=2))
    return 0


def cmd_settings_show(args: argparse.Namespace) -> int:
    from watermark_core.settings import load_settings, settings_path

    s = load_settings()
    data = s.masked_dict() if not args.reveal else s.to_dict()
    data["_path"] = str(settings_path())
    print(json.dumps(data, indent=2))
    return 0


def cmd_settings_set(args: argparse.Namespace) -> int:
    from watermark_core.settings import (
        apply_provider_preset,
        clear_api_key,
        load_settings,
        save_settings,
        settings_path,
    )

    if args.clear_key:
        clear_api_key()
        print("API key cleared.")
    if args.provider:
        apply_provider_preset(args.provider, keep_key=True)
        print(f"Provider preset: {args.provider}")

    s = load_settings()
    if args.api_key:
        s.api_key = args.api_key
    if args.base_url:
        s.base_url = args.base_url
    if args.model:
        s.model = args.model
    if args.inpaint_model:
        s.inpaint_model = args.inpaint_model
    save_settings(s)
    print(f"Saved settings → {settings_path()}")
    print(json.dumps(s.masked_dict(), indent=2))
    return 0


def _add_wm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scheme", default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--key", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--preset", default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="faku",
        description="fak-u-watermark — detect, highlight, neutralize",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── text ──────────────────────────────────────────────────────────────
    text = sub.add_parser("text", help="Text watermark tools")
    text_sub = text.add_subparsers(dest="text_cmd", required=True)

    analyze = text_sub.add_parser("analyze", help="Analyze text for watermark signal")
    analyze.add_argument("text", nargs="?", default="")
    analyze.add_argument("-f", "--file")
    _add_wm_args(analyze)
    analyze.add_argument("--density-window", type=int, default=20)
    analyze.add_argument("--json", action="store_true")
    analyze.add_argument("--html", help="Write highlighted HTML + heatmap")
    analyze.set_defaults(func=cmd_text_analyze)

    hl = text_sub.add_parser("highlight", help="Export annotated HTML + heatmap")
    hl.add_argument("text", nargs="?", default="")
    hl.add_argument("-f", "--file")
    hl.add_argument("-o", "--output", default="highlighted.html")
    _add_wm_args(hl)
    hl.set_defaults(func=cmd_text_highlight, preset="kirchenbauer_default")

    batch = text_sub.add_parser("batch", help="Analyze multiple files")
    batch.add_argument("files", nargs="+")
    _add_wm_args(batch)
    batch.add_argument("--json", action="store_true")
    batch.set_defaults(func=cmd_text_batch)

    neu = text_sub.add_parser("neutralize", help="LLM paraphrase neutralize")
    neu.add_argument("text", nargs="?", default="")
    neu.add_argument("-f", "--file")
    neu.add_argument("-o", "--output")
    neu.add_argument("--style", choices=["subtle", "strong"], default="subtle")
    neu.add_argument("--api-key", default=None)
    neu.add_argument("--base-url", default=None)
    neu.add_argument("--model", default=None)
    neu.set_defaults(func=cmd_text_neutralize)

    tgt = text_sub.add_parser("targeted", help="Offline green→red synonym neutralize")
    tgt.add_argument("text", nargs="?", default="")
    tgt.add_argument("-f", "--file")
    tgt.add_argument("-o", "--output")
    _add_wm_args(tgt)
    tgt.add_argument("--json", action="store_true")
    tgt.set_defaults(func=cmd_text_targeted)

    adp = text_sub.add_parser("adaptive", help="Adaptive paraphrase until z drops")
    adp.add_argument("text", nargs="?", default="")
    adp.add_argument("-f", "--file")
    adp.add_argument("-o", "--output")
    adp.add_argument("--style", choices=["subtle", "strong"], default="subtle")
    adp.add_argument("--max-rounds", type=int, default=3)
    adp.add_argument("--target-z", type=float, default=4.0)
    adp.add_argument("--api-key", default=None)
    adp.add_argument("--base-url", default=None)
    adp.add_argument("--model", default=None)
    _add_wm_args(adp)
    adp.add_argument("--json", action="store_true")
    adp.set_defaults(func=cmd_text_adaptive)

    # ── image ─────────────────────────────────────────────────────────────
    image = sub.add_parser("image", help="Image tools")
    image_sub = image.add_subparsers(dest="image_cmd", required=True)

    exif = image_sub.add_parser("exif", help="Print EXIF / metadata")
    exif.add_argument("file")
    exif.set_defaults(func=cmd_image_exif)

    strip = image_sub.add_parser("strip", help="Strip all metadata")
    strip.add_argument("file")
    strip.add_argument("-o", "--output")
    strip.set_defaults(func=cmd_image_strip)

    c2pa = image_sub.add_parser("c2pa", help="Detect / strip C2PA markers")
    c2pa.add_argument("file")
    c2pa.add_argument("--strip", action="store_true")
    c2pa.add_argument("-o", "--output")
    c2pa.set_defaults(func=cmd_image_c2pa)

    # ── settings ──────────────────────────────────────────────────────────
    settings = sub.add_parser("settings", help="API keys & provider config")
    settings_sub = settings.add_subparsers(dest="settings_cmd", required=True)

    show = settings_sub.add_parser("show", help="Show saved settings")
    show.add_argument("--reveal", action="store_true", help="Show full API key")
    show.set_defaults(func=cmd_settings_show)

    st = settings_sub.add_parser("set", help="Save API key / provider manually")
    st.add_argument("--api-key", default=None)
    st.add_argument("--base-url", default=None)
    st.add_argument("--model", default=None)
    st.add_argument("--inpaint-model", default=None)
    st.add_argument(
        "--provider",
        choices=["openai", "deepseek", "xai", "custom"],
        default=None,
    )
    st.add_argument("--clear-key", action="store_true")
    st.set_defaults(func=cmd_settings_set)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
