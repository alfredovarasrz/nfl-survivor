"""Probability utilities shared by the Survivor optimizers.

The functions in this module are deliberately independent of files, APIs, and
optimizer libraries.  That makes the basic probability calculations easy to
test against hand-worked examples.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence


class ProbabilityError(ValueError):
    """Raised when odds or probabilities are invalid."""


def american_odds_to_implied(odds: float | int) -> float:
    """Convert American odds to the sportsbook's raw implied probability."""

    try:
        value = float(odds)
    except (TypeError, ValueError) as exc:
        raise ProbabilityError(f"American odds must be numeric, received {odds!r}") from exc
    if not math.isfinite(value) or value == 0:
        raise ProbabilityError("American odds must be finite and non-zero")
    if value > 0:
        return 100.0 / (value + 100.0)
    return -value / (-value + 100.0)


def validate_probability(probability: float, *, name: str = "probability") -> float:
    """Return a finite probability strictly between zero and one."""

    try:
        value = float(probability)
    except (TypeError, ValueError) as exc:
        raise ProbabilityError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ProbabilityError(f"{name} must be strictly between 0 and 1")
    return value


def remove_vig(
    odds_a: float | int,
    odds_b: float | int,
) -> tuple[float, float]:
    """Return proportional no-vig probabilities from a two-sided moneyline."""

    raw_a = american_odds_to_implied(odds_a)
    raw_b = american_odds_to_implied(odds_b)
    total = raw_a + raw_b
    return raw_a / total, raw_b / total


def normalize_probability_pair(
    probability_a: float,
    probability_b: float,
) -> tuple[float, float]:
    """Normalize two positive probability estimates so that they sum to one."""

    a = validate_probability(probability_a, name="probability_a")
    b = validate_probability(probability_b, name="probability_b")
    total = a + b
    return a / total, b / total


def consensus_fair_probability(
    moneyline_pairs: Iterable[tuple[float | int, float | int]],
) -> tuple[float, float]:
    """Return the median of per-book no-vig probabilities.

    Vig is removed within each sportsbook before combining books.  Taking the
    median prevents one stale or anomalous book from dominating the consensus.
    """

    fair_a = [remove_vig(a, b)[0] for a, b in moneyline_pairs]
    if not fair_a:
        raise ProbabilityError("At least one complete two-sided moneyline is required")
    probability_a = statistics.median(fair_a)
    return probability_a, 1.0 - probability_a


def survival_curve(probabilities: Sequence[float]) -> list[float]:
    """Return P(survive through week k) for every supplied week."""

    curve: list[float] = []
    cumulative = 1.0
    for week, probability in enumerate(probabilities, start=1):
        cumulative *= validate_probability(probability, name=f"week {week} probability")
        curve.append(cumulative)
    return curve


def expected_weeks_survived(probabilities: Sequence[float]) -> float:
    """Return the exact expected number of weeks survived."""

    return math.fsum(survival_curve(probabilities))
