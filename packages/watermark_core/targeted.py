"""Targeted green→red neutralization when the watermark key is known.

Offline: replace green-list tokens with red-list synonyms / near-synonyms
while preserving surrounding text as much as possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .analyzer import AnalysisResult, WatermarkAnalyzer
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

    for i, (tid, tstr, (start, end)) in enumerate(zip(token_ids, token_strings, offsets)):
        # Preserve any characters between previous end and this start
        if start > cursor:
            pieces.append(text[cursor:start])
            cursor = start

        is_green = False
        if not (i == 0 and analyzer.scheme_name == "kgw"):
            prev = rebuilt_ids if rebuilt_ids else token_ids[:i]
            is_green = analyzer.scheme.score_token(tid, prev, vsize)

        replacement_text = None
        if is_green and len(replacements) < max_replacements:
            replacement_text = _pick_red_replacement(
                token_text=tstr,
                prev_ids=rebuilt_ids if rebuilt_ids else token_ids[:i],
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
    # Normalize whitespace runs introduced by awkward token edges
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

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
    # GPT-2 often has leading space Ġ
    leading_space = word.startswith(" ") or word.startswith("\u0120") or word.startswith("Ġ")
    # transformers decode usually gives real space
    core = word.lstrip(" \t\n\rĠ")
    if not core or not re.search(r"[A-Za-z]", core):
        return None

    # Preserve simple casing
    def _case_like(src: str, template: str) -> str:
        if template.isupper():
            return src.upper()
        if template[:1].isupper() and template[1:].islower():
            return src[:1].upper() + src[1:].lower()
        if template.islower():
            return src.lower()
        return src

    lower = core.lower()
    candidates = list(_SYNONYMS.get(lower, []))
    # Mild morphological hacks
    if lower.endswith("ing") and lower[:-3] in _SYNONYMS:
        candidates.extend(s + "ing" for s in _SYNONYMS[lower[:-3]])
    if lower.endswith("ed") and lower[:-2] in _SYNONYMS:
        candidates.extend(s + "ed" for s in _SYNONYMS[lower[:-2]])

    green = analyzer.scheme.get_green_list(prev_ids, vocab_size)

    for cand in candidates:
        shaped = _case_like(cand, core)
        # Prefer space-prefixed form if original had leading space
        trial = (" " + shaped) if (word.startswith(" ") or leading_space) else shaped
        # Also try without forcing space
        for form in (trial, shaped, " " + shaped):
            ids = tokenizer.encode(form, add_special_tokens=False)
            if not ids:
                continue
            # Accept if first token is red (not green)
            if ids[0] not in green:
                # Prefer single-token replacements
                if len(ids) == 1:
                    return form if form.startswith(" ") or not word.startswith(" ") else form
                # multi-token ok if all red-ish
                if all(
                    not analyzer.scheme.score_token(
                        mid, prev_ids + list(ids[:j]), vocab_size
                    )
                    for j, mid in enumerate(ids)
                ):
                    return form
    return None
