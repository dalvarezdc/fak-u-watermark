"""Targeted green→red neutralization when the watermark key is known.

Offline: replace green-list tokens with red-list synonyms / near-synonyms
while preserving surrounding text as much as possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .analyzer import WatermarkAnalyzer
from .tokenizer import encode_with_offsets, load_tokenizer, vocab_size

# Small built-in synonym map (word → alternatives). Expand as needed.
_SYNONYMS: dict[str, list[str]] = {
    "quick": ["fast", "swift", "rapid"],
    "fast": ["quick", "rapid", "swift"],
    "big": ["large", "great", "huge"],
    "large": ["big", "great", "huge"],
    "small": ["little", "tiny", "minor"],
    "good": ["fine", "great", "solid"],
    "bad": ["poor", "weak", "rough"],
    "happy": ["glad", "pleased", "joyful"],
    "said": ["stated", "noted", "told"],
    "show": ["display", "reveal", "present"],
    "important": ["key", "major", "critical"],
    "help": ["aid", "assist", "support"],
    "use": ["utilize", "employ", "apply"],
    "make": ["create", "build", "form"],
    "get": ["obtain", "gain", "fetch"],
    "think": ["believe", "consider", "reckon"],
    "know": ["understand", "recognize"],
    "want": ["wish", "desire", "need"],
    "need": ["require", "want"],
    "people": ["persons", "folks", "humans"],
    "start": ["begin", "commence"],
    "end": ["finish", "conclude", "close"],
    "new": ["fresh", "novel", "recent"],
    "old": ["aged", "prior", "former"],
    "high": ["elevated", "tall", "upper"],
    "low": ["lower", "small", "modest"],
    "very": ["quite", "highly", "truly"],
    "also": ["too", "additionally", "likewise"],
    "however": ["yet", "still", "though"],
    "because": ["since", "as"],
    "about": ["regarding", "around", "concerning"],
    "before": ["prior", "earlier"],
    "after": ["following", "later"],
    "under": ["beneath", "below"],
    "over": ["above", "across"],
    "problem": ["issue", "challenge", "matter"],
    "result": ["outcome", "finding"],
    "method": ["approach", "technique", "way"],
    "system": ["framework", "setup"],
    "data": ["information", "figures"],
    "model": ["system", "network"],
    "generate": ["produce", "create"],
    "generated": ["produced", "created"],
    "text": ["content", "writing", "prose"],
    "watermark": ["mark", "signal", "trace"],
}


@dataclass
class TargetedResult:
    original: str
    cleaned: str
    replacements: list[dict] = field(default_factory=list)
    green_before: int = 0
    green_after: int = 0
    z_before: float = 0.0
    z_after: float = 0.0
    success: bool = True
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "cleaned": self.cleaned,
            "replacements": self.replacements,
            "green_before": self.green_before,
            "green_after": self.green_after,
            "z_before": self.z_before,
            "z_after": self.z_after,
            "success": self.success,
            "notes": self.notes,
            "method": "targeted_synonym",
        }


def neutralize_targeted(
    text: str,
    analyzer: WatermarkAnalyzer | None = None,
    *,
    scheme: str = "kgw",
    gamma: float = 0.25,
    key: str | int | None = None,
    tokenizer_name: str = "gpt2",
    max_replacements: int = 500,
) -> TargetedResult:
    """
    Replace green-list token spans with red-list alternatives.

    Strategy per green token (left→right on original text):
    1. Look up synonyms for the stripped word form
    2. Prefer candidates whose first token is on the red list given previous context
    3. Fallback: leave token unchanged
    """
    if analyzer is None:
        analyzer = WatermarkAnalyzer(
            scheme=scheme,
            gamma=gamma,
            key=key,
            tokenizer_name=tokenizer_name,
        )
    before = analyzer.analyze(text)
    if not text.strip():
        return TargetedResult(original=text, cleaned=text, notes="Empty text.")

    tok = load_tokenizer(analyzer.tokenizer_name)
    vsize = vocab_size(analyzer.tokenizer_name)
    token_ids, token_strings, offsets = encode_with_offsets(text, analyzer.tokenizer_name)

    # Rebuild text from left to right with optional span replacements
    pieces: list[str] = []
    cursor = 0
    replacements: list[dict] = []
    # Track reconstructed token stream for context (approx)
    rebuilt_ids: list[int] = []
    window = max(1, int(getattr(analyzer, "window", 1)))

    for i, (tid, tstr, (start, end)) in enumerate(zip(token_ids, token_strings, offsets)):
        # Preserve any characters between previous end and this start
        if start > cursor:
            pieces.append(text[cursor:start])
            cursor = start

        is_green = False
        if not (i == 0 and analyzer.scheme_name == "kgw"):
            src = rebuilt_ids if rebuilt_ids else token_ids[:i]
            prev = src[-window:]
            is_green = analyzer.scheme.score_token(tid, prev, vsize)

        replacement_text = None
        if is_green and len(replacements) < max_replacements:
            src = rebuilt_ids if rebuilt_ids else token_ids[:i]
            replacement_text = _pick_red_replacement(
                token_text=tstr,
                prev_ids=src[-window:],
                analyzer=analyzer,
                tokenizer=tok,
                vocab_size=vsize,
            )

        if replacement_text is not None and replacement_text != tstr:
            pieces.append(replacement_text)
            replacements.append(
                {
                    "index": i,
                    "original": tstr,
                    "replacement": replacement_text,
                    "start": start,
                    "end": end,
                }
            )
            # Update rebuilt_ids with encoding of replacement
            new_ids = tok.encode(replacement_text, add_special_tokens=False)
            rebuilt_ids.extend(new_ids if new_ids else [tid])
        else:
            pieces.append(text[start:end] if end > start else tstr)
            rebuilt_ids.append(tid)

        cursor = max(cursor, end)

    if cursor < len(text):
        pieces.append(text[cursor:])

    cleaned = "".join(pieces)

    after = analyzer.analyze(cleaned)
    return TargetedResult(
        original=text,
        cleaned=cleaned,
        replacements=replacements,
        green_before=before.statistics.green_count,
        green_after=after.statistics.green_count,
        z_before=before.statistics.z_score,
        z_after=after.statistics.z_score,
        success=True,
        notes=(
            f"Replaced {len(replacements)} green-list token(s). "
            f"z: {before.statistics.z_score:.2f} → {after.statistics.z_score:.2f}"
        ),
    )


def _pick_red_replacement(
    *,
    token_text: str,
    prev_ids: list[int],
    analyzer: WatermarkAnalyzer,
    tokenizer,
    vocab_size: int,
) -> str | None:
    word = token_text
    # GPT-2 often has a leading space (decoded) or Ġ marker
    leading_space = word.startswith(" ") or word.startswith("\u0120") or word.startswith("Ġ")
    core = word.lstrip(" \t\n\rĠ")
    if not core or not re.search(r"[A-Za-z]", core):
        return None

    def _case_like(src: str, template: str) -> str:
        if template.isupper():
            return src.upper()
        if template[:1].isupper() and template[1:].islower():
            return src[:1].upper() + src[1:].lower()
        if template.islower():
            return src.lower()
        return src

    candidates = list(_SYNONYMS.get(core.lower(), []))

    for cand in candidates:
        shaped = _case_like(cand, core)
        # Keep the original token's leading space so words are not glued together.
        form = (" " + shaped) if leading_space else shaped
        ids = tokenizer.encode(form, add_special_tokens=False)
        if not ids:
            continue
        if analyzer.scheme.score_token(ids[0], prev_ids, vocab_size):
            continue
        if len(ids) == 1:
            return form
        if all(
            not analyzer.scheme.score_token(mid, prev_ids + list(ids[:j]), vocab_size)
            for j, mid in enumerate(ids)
        ):
            return form
    return None
