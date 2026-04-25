"""
FabYieldEnv — OpenEnv-compliant semiconductor yield optimization environment.

Mirrors the HFOrchestratorEnv pattern exactly:
  - Pydantic models for Observation and Action
  - .reset() returns Observation
  - .step(action) returns (Observation, reward_dict, done, info)
  - server.py imports this and mounts it on FastAPI
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

from environment.rsm_simulator import RSMSimulator, PARAM_RANGES, PHASE_PARAMS, ALL_PARAMS
from agents.senior_reviewer import SeniorEngineerReviewer, sample_episode_constraints


# ─── Pydantic Models (OpenEnv-style, same pattern as HFOrchestratorEnv) ──────

class ExperimentRecord(BaseModel):
    step: int
    params: Dict[str, float]
    yield_pct: float
    defect: str
    primary_bottleneck_guess: str = ""
    reasoning: str = ""


class FabObservation(BaseModel):
    """What the agent sees each step. Serializable for FastAPI response."""
    step: int
    phase: str
    budget_remaining: int
    target_yield: float = 92.0
    current_best_yield: float = 0.0
    experiment_history: List[ExperimentRecord] = Field(default_factory=list)
    phase_hint: str = ""
    reviewer_feedback: str = ""          # filled if reviewer rejected last recipe
    active_params: List[str] = Field(default_factory=list)
    param_ranges: Dict[str, List[float]] = Field(default_factory=dict)
    done: bool = False
    episode_result: Optional[str] = None  # "success" | "budget_exhausted" | None


class FabAction(BaseModel):
    """Parsed from agent's XML output. Also accepted directly via API."""
    params: Dict[str, float]
    primary_bottleneck: str = ""
    reasoning: str = ""
    submit: bool = False


# ─── Phase Config ─────────────────────────────────────────────────────────────

EPISODE_PHASES = {
    "exploration":  (1, 4),
    "hypothesis":   (5, 8),
    "exploitation": (9, 11),
    "submission":   (12, 12),
}

PHASE_HINTS = {
    "exploration":  "Vary parameters broadly to map the response surface. Cover diverse combinations.",
    "hypothesis":   "You have data now. Form a hypothesis about the primary bottleneck and test it directly.",
    "exploitation": "Converge on the optimum. Fine-tune 1-2 params near your current best point.",
    "submission":   "Final step. Submit your recipe and state your causal diagnosis of the yield bottleneck.",
}

MAX_STEPS = 12
TARGET_YIELD = 92.0


def _get_phase(step: int) -> str:
    for phase, (lo, hi) in EPISODE_PHASES.items():
        if lo <= step <= hi:
            return phase
    return "submission"


# ─── Main Environment ─────────────────────────────────────────────────────────

class FabYieldEnv:
    """
    Semiconductor Yield Optimization RL Environment.

    Episode flow:
      obs = env.reset(difficulty=1)          # start new wafer lot
      obs, rewards, done, info = env.step(action)   # run one experiment
      ...
      obs, rewards, done, info = env.step(FabAction(params={...}, submit=True))

    difficulty: 1 = 5 params (easy), 2 = 10 params (medium), 3 = 15 params (hard)
    """

    def __init__(self, difficulty: int = 1):
        self.difficulty = difficulty
        self.rsm: Optional[RSMSimulator] = None
        self.reviewer: Optional[SeniorEngineerReviewer] = None
        self._obs: Optional[FabObservation] = None
        self._done: bool = False
        self._last_submitted_params: Optional[Dict] = None

    def reset(self, seed: int = None, difficulty: int = None) -> FabObservation:
        """Start a fresh episode. Returns initial observation."""
        if difficulty is not None:
            self.difficulty = difficulty

        self.rsm = RSMSimulator(seed=seed, difficulty=self.difficulty)
        constraints = sample_episode_constraints(
            active_params=PHASE_PARAMS[self.difficulty],
            rng=np.random.default_rng(seed)
        )
        self.reviewer = SeniorEngineerReviewer(constraints)
        self._done = False
        self._last_submitted_params = None

        active = PHASE_PARAMS[self.difficulty]
        param_ranges = {p: list(PARAM_RANGES[p]) for p in active}

        self._obs = FabObservation(
            step=1,
            phase="exploration",
            budget_remaining=MAX_STEPS,
            target_yield=TARGET_YIELD,
            current_best_yield=0.0,
            experiment_history=[],
            phase_hint=PHASE_HINTS["exploration"],
            reviewer_feedback="",
            active_params=active,
            param_ranges=param_ranges,
            done=False,
        )
        return self._obs

    def step(self, action: FabAction) -> Tuple[FabObservation, Dict[str, float], bool, Dict]:
        """
        Run one experiment (or final submission).

        Returns:
            obs:          updated FabObservation
            reward_dict:  breakdown of all reward components
            done:         episode over?
            info:         causal ground truth (for training, not shown to agent)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # Clamp params to valid ranges
        params = self._clamp_params(action.params)

        # Run the RSM simulation
        yield_pct, defect = self.rsm.predict(params)

        # Record the experiment
        record = ExperimentRecord(
            step=self._obs.step,
            params=params,
            yield_pct=yield_pct,
            defect=defect,
            primary_bottleneck_guess=action.primary_bottleneck,
            reasoning=action.reasoning,
        )
        self._obs.experiment_history.append(record)

        # Update best yield
        best = max(r.yield_pct for r in self._obs.experiment_history)
        self._obs.current_best_yield = best

        # Advance state
        self._obs.step += 1
        self._obs.budget_remaining -= 1
        self._obs.phase = _get_phase(self._obs.step)
        self._obs.phase_hint = PHASE_HINTS.get(self._obs.phase, "")

        # Check reviewer if agent submitted
        reviewer_result = {"approved": True, "feedback": ""}
        if action.submit:
            reviewer_result = self.reviewer.review(params)
            self._obs.reviewer_feedback = reviewer_result["feedback"]
            self._last_submitted_params = params

        # Terminal condition
        done = (
            action.submit
            or self._obs.budget_remaining <= 0
            or self._obs.step > MAX_STEPS
        )
        self._done = done
        self._obs.done = done
        if done:
            self._obs.episode_result = (
                "success" if best >= TARGET_YIELD else "budget_exhausted"
            )

        # Compute rewards
        reward_dict = self._compute_reward(
            action=action,
            yield_pct=yield_pct,
            done=done,
            submitted_params=params if action.submit else None,
            reviewer_approved=reviewer_result["approved"],
        )

        # Info carries ground truth for training (never in agent prompt)
        info = {
            "causal_structure": self.rsm.causal_structure(),
            "reviewer_approved": reviewer_result["approved"],
            "true_optimum": self.rsm.optimum_params,
            "defect": defect,
        }

        return self._obs, reward_dict, done, info

    def _compute_reward(
        self,
        action: FabAction,
        yield_pct: float,
        done: bool,
        submitted_params: Optional[Dict],
        reviewer_approved: bool,
    ) -> Dict[str, float]:
        rewards = {}

        # 1. YIELD REWARD — Make it a steep hill, not a freebie
        # 0 points for anything under 50% yield. 1.0 points for hitting target.
        yield_floor = 50.0
        if yield_pct <= yield_floor:
            rewards["yield"] = 0.0
        else:
            rewards["yield"] = min((yield_pct - yield_floor) / (TARGET_YIELD - yield_floor), 1.0)

        # 2. CAUSAL ATTRIBUTION — Score this first
        causal = self.rsm.causal_structure()
        attributed = action.primary_bottleneck.strip().lower()
        true_primary = causal["primary_param"].lower()
        true_secondary = [p.lower() for p in causal["secondary_params"]]

        if attributed == true_primary: rewards["causal"] = 1.0
        elif attributed in true_secondary: rewards["causal"] = 0.4
        else: rewards["causal"] = 0.0

        # 3. EFFICIENCY REWARD — Gated behind competence
        # You only get the speed bonus if you ACTUALLY fixed the yield AND found the root cause
        if yield_pct >= TARGET_YIELD and rewards["causal"] > 0:
            rewards["efficiency"] = max(0.0, (MAX_STEPS - self._obs.step + 1) / MAX_STEPS)
        else:
            rewards["efficiency"] = 0.0

        # 4. STABILITY REWARD — Penalize 1-step blind submissions
        if done and submitted_params:
            if self._obs.step < 3: # Punish guessing without experimenting
                rewards["stability"] = 0.0
            else:
                lot_std = self.rsm.lot_variance(submitted_params, n_lots=20)
                rewards["stability"] = max(0.0, 1.0 - lot_std / 5.0)
                if not reviewer_approved:
                    rewards["stability"] *= 0.3
        else:
            rewards["stability"] = 0.0

        # Composite (Weights remain the same)
        rewards["total"] = (
            0.50 * rewards["yield"] +
            0.20 * rewards["efficiency"] +
            0.15 * rewards["causal"] +
            0.15 * rewards["stability"]
        )

        return {k: round(float(v), 4) for k, v in rewards.items()}

    def _clamp_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Clamp all params to their valid ranges. Fills missing active params with midpoint."""
        result = {}
        for p in PHASE_PARAMS[self.difficulty]:
            lo, hi = PARAM_RANGES[p]
            val = params.get(p, (lo + hi) / 2.0)  # default to midpoint if missing
            result[p] = float(np.clip(val, lo, hi))
        return result

    @property
    def observation_space(self) -> Dict:
        return {
            "type": "dict",
            "fields": [
                "step", "phase", "budget_remaining", "target_yield",
                "current_best_yield", "experiment_history",
                "phase_hint", "reviewer_feedback", "active_params", "param_ranges",
            ]
        }

    @property
    def action_space(self) -> Dict:
        return {
            "type": "dict",
            "fields": {
                "params": "Dict[str, float] — experiment parameter values",
                "primary_bottleneck": "str — agent's causal hypothesis",
                "reasoning": "str — one sentence explanation",
                "submit": "bool — True to end episode and submit recipe",
            }
        }