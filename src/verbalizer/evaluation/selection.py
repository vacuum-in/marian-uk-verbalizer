"""Safety-gated composite checkpoint selection."""

from __future__ import annotations

from typing import Any


def composite_score(metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gate_failures = {
        key: {"actual": float(metrics.get(key, 0.0)), "maximum": float(maximum)}
        for key, maximum in config["safety_gates"].items()
        if float(metrics.get(key, 0.0)) > float(maximum)
    }
    components: dict[str, float] = {}
    score = 0.0
    for key, raw_weight in config["weights"].items():
        weight = float(raw_weight)
        if key.startswith("inverse_"):
            metric_name = key.removeprefix("inverse_")
            value = 1.0 - min(1.0, max(0.0, float(metrics[metric_name])))
        else:
            value = float(metrics[key])
        components[key] = value
        score += weight * value
    return {
        "eligible": not gate_failures,
        "score": round(score, 9),
        "gate_failures": gate_failures,
        "components": components,
    }
