"""Adaptive neutralization: paraphrase → re-score until z-score drops."""

from __future__ import annotations

from dataclasses import dataclass, field

from .analyzer import WatermarkAnalyzer
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
            "final_z": round(self.z_scores[-1], 4) if self.z_scores else None,
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

    Uses the existing OpenAI-compatible API (not a local LLM).
    Optionally escalates subtle → strong after round 1 if still high.
    """
    if analyzer is None:
        analyzer = WatermarkAnalyzer(
            scheme=scheme,
            gamma=gamma,
            key=key,
            tokenizer_name=tokenizer_name,
            threshold=target_z,
        )
    config = config or NeutralizeConfig.from_env()

    current = text
    z_scores: list[float] = []
    stats0 = analyzer.analyze(current).statistics
    z_scores.append(stats0.z_score)

    if stats0.z_score < target_z:
        return AdaptiveResult(
            original=text,
            cleaned=text,
            rounds=0,
            z_scores=z_scores,
            success=True,
            style=config.style,
            model=config.model,
            error=None,
        )

    last_error = None
    for r in range(1, max_rounds + 1):
        if escalate_style and r >= 2:
            config.style = "strong"
        result: NeutralizeResult = neutralize_sync(current, config)
        if not result.success:
            last_error = result.error
            break
        current = result.cleaned
        z = analyzer.analyze(current).statistics.z_score
        z_scores.append(z)
        if z < target_z:
            return AdaptiveResult(
                original=text,
                cleaned=current,
                rounds=r,
                z_scores=z_scores,
                success=True,
                style=config.style,
                model=config.model,
            )

    return AdaptiveResult(
        original=text,
        cleaned=current,
        rounds=len(z_scores) - 1,
        z_scores=z_scores,
        success=last_error is None,
        error=last_error
        or (
            f"Stopped after {max_rounds} rounds; final z={z_scores[-1]:.2f} "
            f"(target < {target_z})."
            if z_scores[-1] >= target_z
            else None
        ),
        style=config.style,
        model=config.model,
    )
