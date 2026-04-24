"""
reviewer.py — SeniorEngineerReviewer

Rule-based (NOT an LLM). Fast, deterministic, no extra training needed.
Generates episode-varying qualification constraints — this is the Snorkel
"changing preferences" bonus track.

Each episode a new set of tighter-than-optimum qualified ranges is sampled.
The agent's submitted recipe must satisfy ALL of them to be approved.
"""
import numpy as np
from typing import Dict, Optional
from environment.models import ReviewerConstraints, FabAction
from rsm_simulator import PARAM_RANGES


def generate_episode_constraints(
    active_params: list,
    optimum_params: Dict[str, float],
    seed: int,
    tightness: float = 0.35,
) -> ReviewerConstraints:
    """
    Sample episode-varying qualified ranges around (but not always centered on)
    the true optimum. The agent doesn't know these until it submits.

    tightness: fraction of full range that is "qualified"
    """
    rng = np.random.default_rng(seed + 9999)
    qualified_ranges = {}

    for p in active_params:
        lo_full, hi_full = PARAM_RANGES[p]
        full_width = hi_full - lo_full
        window = tightness * full_width

        # Center the window near optimum but with some random offset
        opt_val = optimum_params[p]
        offset = rng.uniform(-0.15 * full_width, 0.15 * full_width)
        center = np.clip(opt_val + offset, lo_full + window / 2, hi_full - window / 2)

        q_lo = round(max(lo_full, center - window / 2), 4)
        q_hi = round(min(hi_full, center + window / 2), 4)
        qualified_ranges[p] = [q_lo, q_hi]

    return ReviewerConstraints(qualified_ranges=qualified_ranges)


class SeniorEngineerReviewer:
    """
    Reviews the agent's submitted recipe against episode-varying
    qualification constraints. Returns approval or specific feedback.

    This creates the multi-agent coordination dynamic:
    the Process Engineer agent must read reviewer feedback and adapt.
    """

    def __init__(self, constraints: ReviewerConstraints):
        self.constraints = constraints
        self.revision_budget = constraints.revision_budget
        self.reviews_done = 0

    def review(self, action: FabAction) -> Dict:
        """
        Check each active parameter against its qualified range.
        Returns approval status + specific violation feedback.
        """
        self.reviews_done += 1
        violations = []

        for param, value in action.params.items():
            if param not in self.constraints.qualified_ranges:
                continue
            lo, hi = self.constraints.qualified_ranges[param]
            if not (lo <= value <= hi):
                violations.append(
                    f"{param}={value:.4g} is outside qualified range "
                    f"[{lo:.4g}, {hi:.4g}]"
                )

        if violations:
            remaining = max(0, self.revision_budget - self.reviews_done)
            return {
                "approved": False,
                "feedback": (
                    f"Recipe REJECTED by Senior Engineer. "
                    f"Violations: {'; '.join(violations)}. "
                    f"Revision budget remaining: {remaining} experiments."
                ),
                "violations": violations,
                "revision_budget_remaining": remaining,
            }

        return {
            "approved": True,
            "feedback": "Recipe APPROVED. Qualified for production.",
            "violations": [],
            "revision_budget_remaining": self.revision_budget - self.reviews_done,
        }

    def get_constraint_hint(self) -> str:
        """
        Returns a vague hint visible in the agent prompt after step 8.
        The agent knows constraints EXIST but not exact values until submission.
        This forces the agent to hedge toward center-of-range values.
        """
        return (
            "NOTE: Your final recipe will be reviewed against production "
            "qualification constraints. Parameters far from their nominal "
            "process window are at risk of rejection."
        )