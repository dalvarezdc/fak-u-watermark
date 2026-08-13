"""Adaptive neutralization: paraphrase → re-score until z-score drops."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .analyzer import WatermarkAnalyzer
from .chunking import DEFAULT_MAX_CHUNK_CHARS, TextChunk, join_chunks, split_document
from .neutralize import NeutralizeConfig, NeutralizeResult, neutralize_sync


@dataclass
class AdaptiveResult:
    original: str
    cleaned: str
    rounds: int
    z_scores: list[float] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    style: str = "subtle"
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "cleaned": self.cleaned,
            "rounds": self.rounds,
            "z_scores": [round(z, 4) for z in self.z_scores],
            "success": self.success,
            "error": self.error,
            "style": self.style,
            "model": self.model,
            "method": "adaptive_paraphrase",
            "final_z": round(min(self.z_scores), 4) if self.z_scores else None,
        }


def neutralize_adaptive(
    text: str,
    *,
    analyzer: WatermarkAnalyzer | None = None,
    config: NeutralizeConfig | None = None,
    max_rounds: int = 3,
    target_z: float = 4.0,
    scheme: str = "kgw",
    gamma: float = 0.25,
    key: str | int | None = None,
    tokenizer_name: str = "gpt2",
    escalate_style: bool = True,
) -> AdaptiveResult:
    """
    Repeatedly paraphrase until z-score < target_z or max_rounds exhausted.

    Long documents are split; only high-z sections are rewritten each round.
    """
    if analyzer is None:
        analyzer = WatermarkAnalyzer(
            scheme=scheme,
            gamma=gamma,
            key=key,
            tokenizer_name=tokenizer_name,
            threshold=target_z,
        )
    config = replace(config or NeutralizeConfig.from_env())
    max_chars = int(config.max_chunk_chars or DEFAULT_MAX_CHUNK_CHARS)
    chunks = split_document(text, max_chars=max_chars)
    body_idxs = [i for i, c in enumerate(chunks) if c.kind == "body" and c.text.strip()]
    if len(body_idxs) > 1:
        return _adaptive_chunked(
            text,
            chunks=chunks,
            body_idxs=body_idxs,
            analyzer=analyzer,
            config=config,
            max_rounds=max_rounds,
            target_z=target_z,
            escalate_style=escalate_style,
        )
    return _adaptive_whole(
        text,
        analyzer=analyzer,
        config=config,
        max_rounds=max_rounds,
        target_z=target_z,
        escalate_style=escalate_style,
    )


def _adaptive_whole(
    text: str,
    *,
    analyzer: WatermarkAnalyzer,
    config: NeutralizeConfig,
    max_rounds: int,
    target_z: float,
    escalate_style: bool,
) -> AdaptiveResult:
    current = text
    best_text = text
    z_scores: list[float] = []
    best_z = analyzer.analyze(current).statistics.z_score
    z_scores.append(best_z)
    last_style = config.style
    last_model = config.model

    if best_z < target_z:
        return AdaptiveResult(
            original=text,
            cleaned=text,
            rounds=0,
            z_scores=z_scores,
            success=True,
            style=last_style,
            model=last_model,
        )

    for r in range(1, max_rounds + 1):
        last_style = "strong" if escalate_style and r >= 2 else config.style
        result: NeutralizeResult = neutralize_sync(
            current, replace(config, style=last_style)
        )
        last_model = result.model or last_model
        if not result.success:
            return AdaptiveResult(
                original=text,
                cleaned=best_text,
                rounds=r - 1,
                z_scores=z_scores,
                success=False,
                error=result.error,
                style=last_style,
                model=last_model,
            )
        if result.cleaned == current:
            return AdaptiveResult(
                original=text,
                cleaned=best_text,
                rounds=r,
                z_scores=z_scores,
                success=False,
                error=(
                    f"Paraphrase unchanged after round {r}; "
                    f"best z={best_z:.2f} (target < {target_z})."
                ),
                style=last_style,
                model=last_model,
            )

        current = result.cleaned
        z = analyzer.analyze(current).statistics.z_score
        z_scores.append(z)
        if z < best_z:
            best_z = z
            best_text = current
        if z < target_z:
            return AdaptiveResult(
                original=text,
                cleaned=current,
                rounds=r,
                z_scores=z_scores,
                success=True,
                style=last_style,
                model=last_model,
            )

    return AdaptiveResult(
        original=text,
        cleaned=best_text,
        rounds=max_rounds,
        z_scores=z_scores,
        success=False,
        error=(
            f"Stopped after {max_rounds} rounds; best z={best_z:.2f} "
            f"(target < {target_z})."
        ),
        style=last_style,
        model=last_model,
    )


def _adaptive_chunked(
    text: str,
    *,
    chunks: list[TextChunk],
    body_idxs: list[int],
    analyzer: WatermarkAnalyzer,
    config: NeutralizeConfig,
    max_rounds: int,
    target_z: float,
    escalate_style: bool,
) -> AdaptiveResult:
    current = list(chunks)
    best_text = text
    z_scores: list[float] = []
    best_z = analyzer.analyze(text).statistics.z_score
    z_scores.append(best_z)
    last_style = config.style
    last_model = config.model

    if best_z < target_z:
        return AdaptiveResult(
            original=text,
            cleaned=text,
            rounds=0,
            z_scores=z_scores,
            success=True,
            style=last_style,
            model=last_model,
        )

    for r in range(1, max_rounds + 1):
        last_style = "strong" if escalate_style and r >= 2 else config.style
        round_cfg = replace(config, style=last_style)
        changed = False
        last_error = None
        for i in body_idxs:
            piece = current[i].text
            local_z = analyzer.analyze(piece).statistics.z_score
            if local_z < target_z:
                continue
            result = neutralize_sync(piece, round_cfg)
            last_model = result.model or last_model
            if not result.success:
                last_error = result.error
                continue
            if result.cleaned != piece:
                current[i] = TextChunk(result.cleaned, current[i].kind)
                changed = True
        joined = join_chunks(current)
        z = analyzer.analyze(joined).statistics.z_score
        z_scores.append(z)
        if z < best_z:
            best_z = z
            best_text = joined
        if z < target_z:
            return AdaptiveResult(
                original=text,
                cleaned=joined,
                rounds=r,
                z_scores=z_scores,
                success=True,
                style=last_style,
                model=last_model,
            )
        if not changed:
            return AdaptiveResult(
                original=text,
                cleaned=best_text,
                rounds=r,
                z_scores=z_scores,
                success=False,
                error=last_error
                or (
                    f"Paraphrase unchanged after round {r}; "
                    f"best z={best_z:.2f} (target < {target_z})."
                ),
                style=last_style,
                model=last_model,
            )

    return AdaptiveResult(
        original=text,
        cleaned=best_text,
        rounds=max_rounds,
        z_scores=z_scores,
        success=False,
        error=(
            f"Stopped after {max_rounds} rounds; best z={best_z:.2f} "
            f"(target < {target_z})."
        ),
        style=last_style,
        model=last_model,
    )
