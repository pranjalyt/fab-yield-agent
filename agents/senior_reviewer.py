"""
SeniorEngineerReviewer — rule-based multi-agent reviewer.

This agent reviews submitted recipes against episode-varying qualification
constraints. It is NOT an LLM — it's deterministic and fast.

Bonus track coverage:
  - Snorkel AI: constraints change each episode (simulated expert preferences)
  - Fleet AI: one agent evaluating another agent's outputs
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple

from environment.rsm_simulator import PARAM_RANGES, PHASE_PARAMS


def sample_episode_constraints(
    active_params: List[str],
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Sample episode-varying qualified process ranges.

    Each episode, the 'senior engineer' has slightly different constraints
    for what counts as a 'production-qualified' recipe. This prevents
    the agent from memorizing a single recipe and forces it to read feedback.
    """
    if rng is None:
        rng = np.random.default_rng()

    constraints = {}
    for param in active_params:
        lo, hi = PARAM_RANGES[param]
        span = hi - lo
        # Qualified window is a random 50-70% slice of the full range
        window_fraction = rng.uniform(0.50, 0.70)
        window = span * window_fraction
        # Offset the window randomly within [lo, hi]
        offset = rng.uniform(0, span - window)
        q_lo = lo + offset
        q_hi = q_lo + window
        constraints[param] = (round(q_lo, 4), round(q_hi, 4))

    return constraints


class SeniorEngineerReviewer:
    """
    Rule-based reviewer agent. Reviews proposed recipes against
    episode-specific qualified process ranges.

    The agent (LLM process engineer) must read rejection feedback
    and adapt its recipe — this is the multi-agent coordination signal.
    """

    def __init__(self, episode_constraints: Dict[str, Tuple[float, float]]):
        self.qualified_ranges = episode_constraints
        self.revision_budget = 2   # extra experiments if rejected

    def review(self, proposed_recipe: Dict[str, float]) -> Dict:
        """
        Check recipe against qualified ranges.

        Returns:
            approved: bool
            feedback: str — human-readable rejection reasons
            revision_budget: int — extra experiments granted if rejected
            violations: list — machine-readable violation list
        """
        violations = []
        for param, value in proposed_recipe.items():
            if param not in self.qualified_ranges:
                continue
            q_lo, q_hi = self.qualified_ranges[param]
            if not (q_lo <= value <= q_hi):
                violations.append({
                    "param": param,
                    "submitted": round(value, 4),
                    "qualified_range": [round(q_lo, 4), round(q_hi, 4)],
                    "delta": round(min(abs(value - q_lo), abs(value - q_hi)), 4),
                })

        if violations:
            feedback_parts = [
                f"{v['param']}={v['submitted']} outside [{v['qualified_range'][0]}, {v['qualified_range'][1]}]"
                for v in violations
            ]
            return {
                "approved": False,
                "feedback": (
                    f"Recipe REJECTED ({len(violations)} violation(s)). "
                    f"Revise: {'; '.join(feedback_parts)}. "
                    f"You have {self.revision_budget} additional experiment(s) to fix this."
                ),
                "violations": violations,
                "revision_budget": self.revision_budget,
            }

        return {
            "approved": True,
            "feedback": "Recipe APPROVED for production qualification. All parameters within certified ranges.",
            "violations": [],
            "revision_budget": 0,
        }