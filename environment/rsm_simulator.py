"""
RSM Simulator — Response Surface Methodology engine.

This is the hidden ground truth of the fab environment. Every episode
a new RSM is sampled (different coefficients = different optimization problem).
The agent NEVER sees this object directly — only yield% and defect type.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ─── Parameter Space ──────────────────────────────────────────────────────────
# Phase 1 (easy) uses first 5 params, Phase 2 uses first 10, Phase 3 all 15.
PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "temp":         (160.0, 220.0),   # °C         — primary yield driver
    "etch_time":    (30.0,  90.0),    # seconds    — interacts with pressure
    "pressure":     (1.5,   3.0),     # torr       — plasma density
    "dopant":       (8e14,  2e15),    # atoms/cm³  — transistor threshold
    "spin_speed":   (1500.0,3500.0),  # RPM        — resist uniformity
    "anneal_time":  (30.0,  120.0),   # seconds    — dopant activation
    "anneal_temp":  (900.0, 1100.0),  # °C         — crystal recovery
    "dep_rate":     (10.0,  50.0),    # nm/min     — film thickness
    "rf_power":     (100.0, 500.0),   # watts      — etch selectivity
    "gas_ar":       (20.0,  100.0),   # sccm       — plasma chemistry
    "gas_cf4":      (5.0,   30.0),    # sccm       — fluorine density
    "wafer_rot":    (0.0,   30.0),    # RPM        — uniformity correction
    "chuck_temp":   (15.0,  40.0),    # °C         — wafer stress
    "cmp_pressure": (1.0,   5.0),     # psi        — planarization
    "cmp_velocity": (30.0,  90.0),    # RPM        — removal rate
}

ALL_PARAMS = list(PARAM_RANGES.keys())

# Curriculum difficulty: how many params are active per phase
PHASE_PARAMS = {
    1: ALL_PARAMS[:5],   # easy   — 5 params
    2: ALL_PARAMS[:10],  # medium — 10 params
    3: ALL_PARAMS,       # hard   — all 15 params
}


def normalize(params: Dict, active_params: List[str]) -> np.ndarray:
    """Map each active parameter value to [-1, 1] for RSM computation."""
    x = np.zeros(len(active_params))
    for i, p in enumerate(active_params):
        lo, hi = PARAM_RANGES[p]
        x[i] = 2.0 * (params[p] - lo) / (hi - lo) - 1.0
    return x


@dataclass
class RSMSimulator:
    """
    Second-order Response Surface Model.
    Instantiated fresh each episode with new random coefficients.

    yield = base + β·x + α·x² + Σ γ_ij·x_i·x_j + noise

    The primary and secondary causal parameters are determined by the
    magnitude of beta coefficients — the agent must discover these.
    """
    seed: int = None
    difficulty: int = 3  # 1, 2, or 3 — controls active params

    # Set by __post_init__
    active_params: List[str] = field(default_factory=list, init=False)
    beta: np.ndarray = field(default=None, init=False)        # linear
    alpha: np.ndarray = field(default=None, init=False)       # quadratic
    interactions: Dict = field(default_factory=dict, init=False)
    primary_param: str = field(default="", init=False)
    secondary_params: List[str] = field(default_factory=list, init=False)
    base_yield: float = field(default=0.93, init=False)
    optimum_params: Dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.active_params = PHASE_PARAMS[self.difficulty]
        n = len(self.active_params)

        # Linear coefficients: drive the main yield gradient
        self.beta = rng.normal(0, 0.18, n)

        # The primary bottleneck = param with highest |beta|
        primary_idx = int(np.argmax(np.abs(self.beta)))
        self.primary_param = self.active_params[primary_idx]

        # Secondary params: above-threshold but not primary
        threshold = 0.07
        self.secondary_params = [
            self.active_params[i] for i in range(n)
            if abs(self.beta[i]) > threshold and i != primary_idx
        ]

        # Quadratic coefficients (curvature — ensures a finite optimum)
        # Always negative on diagonal so surface has a maximum, not minimum
        self.alpha = -np.abs(rng.normal(0.06, 0.02, n))

        # Sparse interaction terms (only 4-6 pairs to keep it tractable)
        n_interactions = min(5, n * (n - 1) // 2)
        pairs = []
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        chosen = rng.choice(len(all_pairs), size=n_interactions, replace=False)
        for idx in chosen:
            i, j = all_pairs[idx]
            self.interactions[(i, j)] = rng.normal(0, 0.04)

        # Compute theoretical optimum: x* = -beta / (2*alpha) in normalized space
        opt_x = np.clip(-self.beta / (2 * self.alpha), -1.0, 1.0)
        self.optimum_params = {}
        for i, p in enumerate(self.active_params):
            lo, hi = PARAM_RANGES[p]
            # Denormalize back to real units
            self.optimum_params[p] = lo + (opt_x[i] + 1.0) / 2.0 * (hi - lo)

        self.base_yield = rng.uniform(0.91, 0.96)

    def predict(self, params: Dict) -> Tuple[float, str]:
        """
        Run a virtual wafer experiment.

        Returns:
            yield_pct: float in [0, 100]
            defect: one of 'edge_ring', 'center_spot', 'random_scatter', 'none'
        """
        x = normalize(params, self.active_params)
        n = len(self.active_params)

        y = self.base_yield
        y += float(np.dot(self.beta, x))            # linear contribution
        y += float(np.dot(self.alpha, x ** 2))      # quadratic (curvature)
        for (i, j), coef in self.interactions.items():
            if i < n and j < n:
                y += coef * x[i] * x[j]             # interaction term

        # Process noise: lot-to-lot variation (~1.5% std)
        noise = np.random.normal(0.0, 0.015)
        y = float(np.clip(y + noise, 0.0, 1.0))

        defect = self._classify_defect(x, y, params)
        return round(y * 100, 2), defect

    def _classify_defect(self, x: np.ndarray, yield_val: float, params: Dict) -> str:
        """
        Map parameter deviations to physically-motivated defect signatures.
        These are interpretable clues for the agent's causal reasoning.
        """
        if yield_val > 0.88:
            return "none"

        # Pressure + gas flow deviation → edge plasma non-uniformity
        p_idx = self.active_params.index("pressure") if "pressure" in self.active_params else -1
        g_idx = self.active_params.index("gas_ar") if "gas_ar" in self.active_params else -1
        if p_idx >= 0 and g_idx >= 0:
            if abs(x[p_idx]) + abs(x[g_idx]) > 1.3:
                return "edge_ring"

        # Temperature + RF power deviation → center hotspot
        t_idx = self.active_params.index("temp") if "temp" in self.active_params else -1
        r_idx = self.active_params.index("rf_power") if "rf_power" in self.active_params else -1
        if t_idx >= 0:
            rf_contrib = abs(x[r_idx]) if r_idx >= 0 else 0
            if abs(x[t_idx]) + rf_contrib > 1.2:
                return "center_spot"

        # Dopant deviation → contamination-like random scatter
        d_idx = self.active_params.index("dopant") if "dopant" in self.active_params else -1
        if d_idx >= 0 and abs(x[d_idx]) > 0.85:
            return "random_scatter"

        return "none"

    def lot_variance(self, params: Dict, n_lots: int = 20) -> float:
        """Simulate lot-to-lot yield variance for the stability reward."""
        yields = [self.predict(params)[0] for _ in range(n_lots)]
        return float(np.std(yields))

    def causal_structure(self) -> Dict:
        """Return ground truth for reward scoring. Never expose to agent prompt."""
        return {
            "primary_param": self.primary_param,
            "secondary_params": self.secondary_params,
        }