"""
FabYieldEnv — OpenEnv-compliant semiconductor yield optimization environment.
Features: Multi-Agent Financial Controller, TPU/GPU Task Binning, and Test-Time Compute.
"""

from __future__ import annotations

import random
import numpy as np
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple

# 🛑 1. STRICT OPENENV COMPLIANCE
from openenv import Environment 

from environment.rsm_simulator import RSMSimulator, PARAM_RANGES, PHASE_PARAMS
from agents.senior_reviewer import SeniorEngineerReviewer, sample_episode_constraints

# ─── Pydantic Models ─────────────────────────────────────────────────────────

class ExperimentRecord(BaseModel):
    step: int
    params: Dict[str, float]
    yield_pct: float
    defect: str
    primary_bottleneck_guess: str = ""
    reasoning: str = ""

class FabObservation(BaseModel):
    step: int
    phase: str
    task_target: str       # NEW: i9 GPU vs Edge TPU Binning
    budget_remaining: int
    target_yield: float = 92.0
    current_best_yield: float = 0.0
    experiment_history: List[ExperimentRecord] = Field(default_factory=list)
    phase_hint: str = ""
    reviewer_feedback: str = ""
    financial_feedback: str = "" # NEW: Multi-agent cost feedback
    active_params: List[str] = Field(default_factory=list)
    param_ranges: Dict[str, List[float]] = Field(default_factory=dict)
    done: bool = False
    episode_result: Optional[str] = None

class FabAction(BaseModel):
    think: str = "" # NEW: Claude-style Test-Time Compute block
    params: Dict[str, float]
    primary_bottleneck: str = ""
    reasoning: str = ""
    submit: bool = False

# ─── Config & Task Definitions ────────────────────────────────────────────────

EPISODE_PHASES = {
    "exploration":  (1, 4),
    "hypothesis":   (5, 8),
    "exploitation": (9, 11),
    "submission":   (12, 12),
}

MAX_STEPS = 12

# NEW: The GPU/TPU Binning Tasks
TASKS = [
    {
        "name": "Data Center GPU (H200X Flagship)",
        "target": 95.0,
        "hint": "Maximize performance. Budget is high, but the yield surface is highly volatile."
    },
    {
        "name": "Edge TPU (Mobile Neural Engine)",
        "target": 90.0,
        "hint": "Thermal efficiency is critical. The Financial Controller will reject high temp/power recipes."
    }
]

def _get_phase(step: int) -> str:
    for phase, (lo, hi) in EPISODE_PHASES.items():
        if lo <= step <= hi: return phase
    return "submission"

# ─── Main Environment ─────────────────────────────────────────────────────────

class FabYieldEnv(Environment): # 🛑 Inherits from OpenEnv
    
    def __init__(self, difficulty: int = 1):
        super().__init__()
        self.difficulty = difficulty
        self.rsm = None
        self.reviewer = None
        self._obs = None
        self._done = False
        self._current_task = None

    # ─── METHOD 1: RESET ───────────────────────────────────────
    def reset(self, seed: int = None, difficulty: int = None) -> dict:
        if difficulty is not None:
            self.difficulty = difficulty

        self.rsm = RSMSimulator(seed=seed, difficulty=self.difficulty)
        rng = np.random.default_rng(seed)
        
        constraints = sample_episode_constraints(PHASE_PARAMS[self.difficulty], rng)
        self.reviewer = SeniorEngineerReviewer(constraints)
        
        # Randomly assign GPU or TPU task
        self._current_task = random.choice(TASKS)
        self._done = False

        active = PHASE_PARAMS[self.difficulty]
        param_ranges = {p: list(PARAM_RANGES[p]) for p in active}

        self._obs = FabObservation(
            step=1,
            phase="exploration",
            task_target=f"{self._current_task['name']} — {self._current_task['hint']}",
            budget_remaining=MAX_STEPS,
            target_yield=self._current_task['target'],
            current_best_yield=0.0,
            experiment_history=[],
            active_params=active,
            param_ranges=param_ranges,
            done=False,
        )
        
        # 🛑 Return dict for OpenEnv serialization
        return self._obs.model_dump()

    # ─── METHOD 2: STEP ────────────────────────────────────────
    def step(self, action: dict) -> Tuple[dict, Dict[str, float], bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        act = FabAction(**action)
        params = self._clamp_params(act.params)

        # 🛑 FINANCIAL CONTROLLER LOGIC (Multi-Agent)
        fin_approved = True
        fin_feedback = ""
        if "Edge TPU" in self._current_task["name"]:
            if params.get("temp", 0) > 180 or params.get("rf_power", 0) > 300:
                fin_approved = False
                fin_feedback = "❌ FINANCIAL CONTROLLER REJECTED: Edge TPU requires thermal efficiency. Temp must be <=180 and RF Power <=300."

        # If financially rejected, simulate a failed lot without checking physics
        if not fin_approved:
            yield_pct = 0.0
            defect = "budget_overrun"
            self._obs.financial_feedback = fin_feedback
        else:
            yield_pct, defect = self.rsm.predict(params)
            self._obs.financial_feedback = "✅ Financial Controller Approved."

        record = ExperimentRecord(
            step=self._obs.step,
            params=params,
            yield_pct=yield_pct,
            defect=defect,
            primary_bottleneck_guess=act.primary_bottleneck,
            reasoning=act.reasoning,
        )
        self._obs.experiment_history.append(record)
        self._obs.current_best_yield = max(r.yield_pct for r in self._obs.experiment_history)

        self._obs.step += 1
        self._obs.budget_remaining -= 1
        self._obs.phase = _get_phase(self._obs.step)

        reviewer_approved = True
        if act.submit:
            rev_res = self.reviewer.review(params)
            reviewer_approved = rev_res["approved"]
            self._obs.reviewer_feedback = rev_res["feedback"]

        done = (act.submit or self._obs.budget_remaining <= 0 or self._obs.step > MAX_STEPS)
        self._done = done
        self._obs.done = done

        # 🛑 REWARD CALCULATION
        reward_dict = self._compute_reward(act, yield_pct, done, params if act.submit else None, reviewer_approved, fin_approved)

        info = {
            "causal_structure": self.rsm.causal_structure(),
            "defect": defect,
            "financial_approved": fin_approved
        }

        return self._obs.model_dump(), reward_dict, done, info

    # ─── METHOD 3: STATE ───────────────────────────────────────
    def state(self) -> dict:
        """OpenEnv read-only state method."""
        if self._obs is None:
            raise RuntimeError("Environment not initialized.")
        return self._obs.model_dump()

    # ─── REWARDS ───────────────────────────────────────────────
    def _compute_reward(self, action: FabAction, yield_pct: float, done: bool, submitted_params: dict, rev_ok: bool, fin_ok: bool) -> dict:
        rewards = {"yield": 0.0, "causal": 0.0, "reasoning": 0.0, "financial": 0.0, "total": 0.0}

        # 1. Yield Reward
        target = self._current_task["target"]
        if yield_pct > 50.0:
            rewards["yield"] = min((yield_pct - 50.0) / (target - 50.0), 1.0)

        # 2. Causal Attribution
        causal = self.rsm.causal_structure()
        if action.primary_bottleneck.strip().lower() == causal["primary_param"].lower():
            rewards["causal"] = 1.0

        # 3. TEST-TIME COMPUTE REWARD (Claude-style)
        think_len = len(action.think.split())
        if think_len > 30: 
            rewards["reasoning"] = 0.5  # Good thought process
        elif think_len == 0:
            rewards["reasoning"] = -0.5 # Penalty for skipping thought

        # 4. Financial Controller Penalty
        if not fin_ok:
            rewards["financial"] = -1.0
            rewards["yield"] = 0.0 # Nullify yield if it's too expensive

        # Total Composite
        rewards["total"] = (0.40 * rewards["yield"]) + (0.20 * rewards["causal"]) + (0.20 * rewards["reasoning"]) + (0.20 * rewards["financial"])
        return {k: round(float(v), 4) for k, v in rewards.items()}

    def _clamp_params(self, params: dict) -> dict:
        res = {}
        for p in PHASE_PARAMS[self.difficulty]:
            lo, hi = PARAM_RANGES[p]
            res[p] = float(np.clip(params.get(p, (lo + hi) / 2.0), lo, hi))
        return res