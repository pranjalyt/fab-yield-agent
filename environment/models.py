"""
models.py — Shared dataclasses for the Fab Yield environment.
FabAction, FabObservation, FabState. These are the interface contracts.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FabAction:
    """
    What the agent sends each step.
    Parsed from XML tags in the LLM's output.
    """
    params: Dict[str, float]          # parameter values to try
    primary_bottleneck: str           # agent's causal diagnosis
    reasoning: str                    # one-sentence justification
    submit: bool = False              # True on step 12 or early submission


@dataclass
class ExperimentRecord:
    """One row in the agent's experiment history."""
    step: int
    params: Dict[str, float]
    yield_pct: float
    defect: str
    bottleneck_guess: str
    reasoning: str


@dataclass
class FabObservation:
    """
    Everything the agent sees at each step.
    This is what gets serialized and sent in the API response.
    """
    step: int                              # 1–12
    phase: str                             # exploration/hypothesis/exploitation/submission
    budget_remaining: int
    current_best_yield: float
    target_yield: float                    # 80 / 88 / 92 depending on difficulty
    active_params: List[str]               # which params are active this episode
    param_ranges: Dict[str, List[float]]   # [lo, hi] for each active param
    experiment_history: List[ExperimentRecord]
    current_hypothesis: str               # agent's own running text (env stores it)
    reviewer_feedback: Optional[str]      # None until submission attempted
    done: bool = False


@dataclass
class ReviewerConstraints:
    """Episode-varying qualification ranges for the SeniorEngineerReviewer."""
    qualified_ranges: Dict[str, List[float]]   # param → [lo, hi]
    revision_budget: int = 2                   # how many extra experiments allowed after rejection


@dataclass
class RewardBreakdown:
    """All 4 reward components — returned with every step for monitoring."""
    yield_reward: float       # 0–1, dense, every step
    efficiency_reward: float  # 0–1, sparse (only if target hit)
    causal_reward: float      # 0 or 1 (or 0.4 partial), every step
    stability_reward: float   # 0–1, terminal only
    total: float              # weighted composite


@dataclass
class StepResult:
    """Full return from env.step() — maps to the /step API response."""
    observation: FabObservation
    rewards: RewardBreakdown
    done: bool
    info: Dict                 # debug info: true_bottleneck revealed at episode end