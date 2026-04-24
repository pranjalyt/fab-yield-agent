"""
agents/senior_reviewer.py — SeniorEngineerReviewer

Rule-based (NOT an LLM). Fast, deterministic.
Episode-varying qualification constraints = Snorkel bonus track.

API matches env.py exactly:
  constraints = sample_episode_constraints(active_params, rng)
  reviewer = SeniorEngineerReviewer(constraints)
  result = reviewer.review(params_dict)   ← plain dict, not FabAction
"""
import numpy as np
from typing import Dict, Optional, Tuple

from environment.rsm_simulator import PARAM_RANGES


def sample_episode_constraints(
    active_params: list,
    rng=None,
    tightness_lo: float = 0.50,
    tightness_hi: float = 0.70,
) -> Dict[str, Tuple[float, float]]:
    """
    Sample episode-varying qualified ranges for each active parameter.
    Returns a dict of param → (q_lo, q_hi).

    Called once per episode in env.reset(). Changes every episode (Snorkel bonus).
    The agent doesn't see these values until it submits a recipe.
    """
    if rng is None:
        rng = np.random.default_rng()

    constraints = {}
    for param in active_params:
        lo, hi = PARAM_RANGES[param]
        span = hi - lo
        window = span * rng.uniform(tightness_lo, tightness_hi)
        offset = rng.uniform(0, span - window)
        constraints[param] = (
            round(lo + offset, 4),
            round(lo + offset + window, 4),
        )
    return constraints


class SeniorEngineerReviewer:
    """
    Reviews submitted process recipes against episode-varying constraints.

    Multi-agent bonus track:
    - Snorkel AI: constraints change every episode
    - Fleet AI: one agent monitors/evaluates another agent's outputs

    Called from env.step() when action.submit=True.
    review() takes a plain dict of {param: value} — same as env._clamp_params output.
    """

    def __init__(self, episode_constraints: Dict[str, Tuple[float, float]]):
        self.qualified_ranges = episode_constraints
        self.revision_budget = 2
        self.reviews_done = 0

    def review(self, recipe: Dict[str, float]) -> Dict:
        """
        Check each param in the recipe against its qualified range.

        Args:
            recipe: plain dict {param_name: float_value}
                    — exactly what env._clamp_params() returns

        Returns:
            dict with keys: approved, feedback, violations, revision_budget_remaining
        """
        self.reviews_done += 1
        violations = []

        for param, value in recipe.items():
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

        remaining = max(0, self.revision_budget - self.reviews_done)

        if violations:
            parts = [
                f"{v['param']}={v['submitted']} outside "
                f"[{v['qualified_range'][0]}, {v['qualified_range'][1]}]"
                for v in violations
            ]
            return {
                "approved": False,
                "feedback": (
                    f"Recipe REJECTED ({len(violations)} violation(s)). "
                    f"Revise: {'; '.join(parts)}. "
                    f"Revision budget remaining: {remaining} experiments."
                ),
                "violations": violations,
                "revision_budget_remaining": remaining,
            }

        return {
            "approved": True,
            "feedback": "Recipe APPROVED. Qualified for production.",
            "violations": [],
            "revision_budget_remaining": remaining,
        }

    def get_constraint_hint(self) -> str:
        """
        Vague hint shown in agent prompt after step 8.
        Forces agent to hedge toward center-of-range values.
        """
        return (
            "NOTE: Your final recipe will be reviewed against production "
            "qualification constraints. Parameters far from their nominal "
            "process window are at risk of rejection."
        )