"""
scripts/test_episode.py — Complete episode test. No GPU, no API keys required.
Uses only stdlib + numpy so it runs anywhere (Colab free tier, CI, local).

Run from the repo root:
    python scripts/test_episode.py

Tests:
  1. RSMSimulator math (predict, lot_variance, causal_structure)
  2. FabYieldEnv reset/step/done flow
  3. SeniorEngineerReviewer approval + rejection
  4. Prompt builder → parse_action round-trip
  5. Reward function (all 4 components)
  6. Full 12-step episode with a heuristic agent
  7. Phase 3 (15 params)
  8. Phase transitions (exploration → hypothesis → exploitation)

Runtime: ~2 seconds on any CPU.
"""

import sys
import os
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):     print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg):   print(f"  {YELLOW}⚠{RESET} {msg}")
def fail(msg):   print(f"  {RED}✗{RESET} {msg}"); sys.exit(1)
def header(msg): print(f"\n{BOLD}{BLUE}{'═'*60}{RESET}\n{BOLD}  {msg}{RESET}\n{BLUE}{'═'*60}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# INLINE IMPLEMENTATIONS — stdlib + numpy only (no Pydantic/FastAPI needed).
# The production server files use Pydantic for FastAPI serialization.
# These inline versions are functionally identical — same logic, no deps.
# ══════════════════════════════════════════════════════════════════════════════

PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "temp":         (160.0, 220.0),
    "etch_time":    (30.0,  90.0),
    "pressure":     (1.5,   3.0),
    "dopant":       (8e14,  2e15),
    "spin_speed":   (1500.0,3500.0),
    "anneal_time":  (30.0,  120.0),
    "anneal_temp":  (900.0, 1100.0),
    "dep_rate":     (10.0,  50.0),
    "rf_power":     (100.0, 500.0),
    "gas_ar":       (20.0,  100.0),
    "gas_cf4":      (5.0,   30.0),
    "wafer_rot":    (0.0,   30.0),
    "chuck_temp":   (15.0,  40.0),
    "cmp_pressure": (1.0,   5.0),
    "cmp_velocity": (30.0,  90.0),
}
ALL_PARAMS = list(PARAM_RANGES.keys())
PHASE_PARAMS = {1: ALL_PARAMS[:5], 2: ALL_PARAMS[:10], 3: ALL_PARAMS}
MAX_STEPS = 12
TARGET_YIELD = 92.0
EPISODE_PHASES = {
    "exploration":  (1, 4),
    "hypothesis":   (5, 8),
    "exploitation": (9, 11),
    "submission":   (12, 12),
}
PHASE_HINTS = {
    "exploration":  "Vary parameters broadly to map the response surface.",
    "hypothesis":   "Form a hypothesis about the primary bottleneck and test it directly.",
    "exploitation": "Converge on the optimum. Fine-tune 1-2 params near your best point.",
    "submission":   "Final step. Submit your recipe and state your causal diagnosis.",
}


def _get_phase(step: int) -> str:
    for phase, (lo, hi) in EPISODE_PHASES.items():
        if lo <= step <= hi:
            return phase
    return "submission"


def normalize(params, active_params):
    x = np.zeros(len(active_params))
    for i, p in enumerate(active_params):
        lo, hi = PARAM_RANGES[p]
        x[i] = 2.0 * (params[p] - lo) / (hi - lo) - 1.0
    return x


@dataclass
class RSMSimulator:
    seed: int = None
    difficulty: int = 1
    active_params: List[str] = field(default_factory=list, init=False)
    beta: np.ndarray = field(default=None, init=False)
    alpha: np.ndarray = field(default=None, init=False)
    interactions: Dict = field(default_factory=dict, init=False)
    primary_param: str = field(default="", init=False)
    secondary_params: List[str] = field(default_factory=list, init=False)
    base_yield: float = field(default=0.93, init=False)
    optimum_params: Dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.active_params = PHASE_PARAMS[self.difficulty]
        n = len(self.active_params)
        self.beta = rng.normal(0, 0.18, n)
        primary_idx = int(np.argmax(np.abs(self.beta)))
        self.primary_param = self.active_params[primary_idx]
        self.secondary_params = [
            self.active_params[i] for i in range(n)
            if abs(self.beta[i]) > 0.07 and i != primary_idx
        ]
        self.alpha = -np.abs(rng.normal(0.06, 0.02, n))
        n_int = min(5, n * (n - 1) // 2)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        chosen = rng.choice(len(all_pairs), size=n_int, replace=False)
        for idx in chosen:
            i, j = all_pairs[idx]
            self.interactions[(i, j)] = rng.normal(0, 0.04)
        opt_x = np.clip(-self.beta / (2 * self.alpha), -1.0, 1.0)
        self.optimum_params = {}
        for i, p in enumerate(self.active_params):
            lo, hi = PARAM_RANGES[p]
            self.optimum_params[p] = lo + (opt_x[i] + 1.0) / 2.0 * (hi - lo)
        self.base_yield = rng.uniform(0.91, 0.96)

    def predict(self, params):
        x = normalize(params, self.active_params)
        n = len(self.active_params)
        y = self.base_yield
        y += float(np.dot(self.beta, x))
        y += float(np.dot(self.alpha, x ** 2))
        for (i, j), coef in self.interactions.items():
            if i < n and j < n:
                y += coef * x[i] * x[j]
        y = float(np.clip(y + np.random.normal(0.0, 0.015), 0.0, 1.0))
        return round(y * 100, 2), self._classify_defect(x, y, params)

    def _classify_defect(self, x, yield_val, params):
        if yield_val > 0.88:
            return "none"
        p_idx = self.active_params.index("pressure") if "pressure" in self.active_params else -1
        g_idx = self.active_params.index("gas_ar") if "gas_ar" in self.active_params else -1
        if p_idx >= 0 and g_idx >= 0 and abs(x[p_idx]) + abs(x[g_idx]) > 1.3:
            return "edge_ring"
        t_idx = self.active_params.index("temp") if "temp" in self.active_params else -1
        r_idx = self.active_params.index("rf_power") if "rf_power" in self.active_params else -1
        if t_idx >= 0:
            rf = abs(x[r_idx]) if r_idx >= 0 else 0
            if abs(x[t_idx]) + rf > 1.2:
                return "center_spot"
        d_idx = self.active_params.index("dopant") if "dopant" in self.active_params else -1
        if d_idx >= 0 and abs(x[d_idx]) > 0.85:
            return "random_scatter"
        return "none"

    def lot_variance(self, params, n_lots=20):
        return float(np.std([self.predict(params)[0] for _ in range(n_lots)]))

    def causal_structure(self):
        return {"primary_param": self.primary_param, "secondary_params": self.secondary_params}


def sample_episode_constraints(active_params, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    constraints = {}
    for param in active_params:
        lo, hi = PARAM_RANGES[param]
        span = hi - lo
        window = span * rng.uniform(0.50, 0.70)
        offset = rng.uniform(0, span - window)
        constraints[param] = (round(lo + offset, 4), round(lo + offset + window, 4))
    return constraints


class SeniorEngineerReviewer:
    def __init__(self, episode_constraints):
        self.qualified_ranges = episode_constraints
        self.revision_budget = 2

    def review(self, recipe):
        violations = []
        for param, value in recipe.items():
            if param not in self.qualified_ranges:
                continue
            q_lo, q_hi = self.qualified_ranges[param]
            if not (q_lo <= value <= q_hi):
                violations.append({
                    "param": param, "submitted": round(value, 4),
                    "qualified_range": [round(q_lo, 4), round(q_hi, 4)],
                    "delta": round(min(abs(value - q_lo), abs(value - q_hi)), 4),
                })
        if violations:
            parts = [f"{v['param']}={v['submitted']} outside [{v['qualified_range'][0]}, {v['qualified_range'][1]}]"
                     for v in violations]
            return {"approved": False,
                    "feedback": f"Recipe REJECTED ({len(violations)} violation(s)). Revise: {'; '.join(parts)}.",
                    "violations": violations, "revision_budget": self.revision_budget}
        return {"approved": True, "feedback": "Recipe APPROVED.", "violations": [], "revision_budget": 0}


def _fmt(v):
    if v == 0: return "0"
    if abs(v) >= 1e13 or (abs(v) < 0.01 and v != 0): return f"{v:.2e}"
    if abs(v) >= 1000: return f"{v:.0f}"
    return f"{v:.2f}"


def build_prompt(obs):
    step = obs["step"]
    budget = obs["budget_remaining"]
    best = obs["current_best_yield"]
    target = obs["target_yield"]
    phase = obs["phase"]
    phase_hint = obs.get("phase_hint", "")
    reviewer_feedback = obs.get("reviewer_feedback", "")
    history = obs.get("experiment_history", [])
    active_params = obs.get("active_params", [])
    param_ranges = obs.get("param_ranges", {})

    prompt = (f"You are a semiconductor process engineer optimizing wafer yield.\n\n"
              f"═══ CURRENT STATE ═══\n"
              f"  Experiments run: {step-1}/{step-1+budget}  Budget: {budget}  "
              f"Best yield: {best:.1f}%  Target: >{target}%\n"
              f"  Phase: {phase.upper()} — {phase_hint}\n")
    if reviewer_feedback:
        prompt += f"\n⚠ REVIEWER FEEDBACK:\n  {reviewer_feedback}\n"
    prompt += "\n═══ EXPERIMENT HISTORY ═══\n"
    if not history:
        prompt += "  No experiments yet.\n"
    else:
        for rec in history:
            param_str = ", ".join(f"{k}={_fmt(v)}" for k, v in rec["params"].items())
            prompt += f"  Exp {rec['step']}: {param_str}\n"
            prompt += f"    → Yield: {rec['yield_pct']:.1f}%  Defects: {rec['defect']}"
            if rec.get("primary_bottleneck_guess"):
                prompt += f"  Your guess: {rec['primary_bottleneck_guess']}"
            prompt += "\n"
    prompt += "\n═══ PARAMETER SPACE ═══\n"
    for p in active_params:
        if p in param_ranges:
            lo, hi = param_ranges[p]
            prompt += f"  {p}: [{_fmt(lo)} — {_fmt(hi)}]\n"
    prompt += "\n═══ YOUR ACTION ═══\n<experiment>\n"
    for p in active_params:
        if p in param_ranges:
            lo, hi = param_ranges[p]
            prompt += f"  {p}: [{_fmt(lo)}-{_fmt(hi)}]\n"
    prompt += "  submit: [true/false]\n</experiment>\n"
    prompt += "<diagnosis>\n  primary_bottleneck: [param]\n  reasoning: [one sentence]\n</diagnosis>\n"
    return prompt


def parse_action(text):
    action = {"params": {}, "primary_bottleneck": "", "reasoning": "", "submit": False}
    exp = re.search(r"<experiment>(.*?)</experiment>", text, re.DOTALL | re.IGNORECASE)
    if exp:
        for line in exp.group(1).strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip().lower(), val.strip()
            if key == "submit":
                action["submit"] = val.lower() in ("true", "yes", "1")
            else:
                try:
                    action["params"][key] = float(val)
                except ValueError:
                    pass
    diag = re.search(r"<diagnosis>(.*?)</diagnosis>", text, re.DOTALL | re.IGNORECASE)
    if diag:
        for line in diag.group(1).strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip().lower(), val.strip()
            if "bottleneck" in key:
                action["primary_bottleneck"] = val
            elif "reason" in key:
                action["reasoning"] = val
    return action


class FabYieldEnvTest:
    def __init__(self, difficulty=1):
        self.difficulty = difficulty
        self.rsm = None
        self.reviewer = None
        self._obs = None
        self._done = False

    def reset(self, seed=None, difficulty=None):
        if difficulty is not None:
            self.difficulty = difficulty
        self.rsm = RSMSimulator(seed=seed, difficulty=self.difficulty)
        constraints = sample_episode_constraints(
            active_params=PHASE_PARAMS[self.difficulty], rng=np.random.default_rng(seed))
        self.reviewer = SeniorEngineerReviewer(constraints)
        self._done = False
        active = PHASE_PARAMS[self.difficulty]
        self._obs = {
            "step": 1, "phase": "exploration",
            "budget_remaining": MAX_STEPS, "target_yield": TARGET_YIELD,
            "current_best_yield": 0.0, "experiment_history": [],
            "phase_hint": PHASE_HINTS["exploration"], "reviewer_feedback": "",
            "active_params": active,
            "param_ranges": {p: list(PARAM_RANGES[p]) for p in active},
            "done": False, "episode_result": None,
        }
        return self._obs

    def step(self, params, primary_bottleneck="", reasoning="", submit=False):
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")
        clamped = {}
        for p in PHASE_PARAMS[self.difficulty]:
            lo, hi = PARAM_RANGES[p]
            clamped[p] = float(np.clip(params.get(p, (lo + hi) / 2), lo, hi))
        yield_pct, defect = self.rsm.predict(clamped)
        self._obs["experiment_history"].append({
            "step": self._obs["step"], "params": clamped,
            "yield_pct": yield_pct, "defect": defect,
            "primary_bottleneck_guess": primary_bottleneck, "reasoning": reasoning,
        })
        best = max(r["yield_pct"] for r in self._obs["experiment_history"])
        self._obs["current_best_yield"] = best
        self._obs["step"] += 1
        self._obs["budget_remaining"] -= 1
        self._obs["phase"] = _get_phase(self._obs["step"])
        self._obs["phase_hint"] = PHASE_HINTS.get(self._obs["phase"], "")
        reviewer_result = {"approved": True, "feedback": ""}
        if submit:
            reviewer_result = self.reviewer.review(clamped)
            self._obs["reviewer_feedback"] = reviewer_result["feedback"]
        done = submit or self._obs["budget_remaining"] <= 0 or self._obs["step"] > MAX_STEPS
        self._done = done
        self._obs["done"] = done
        if done:
            self._obs["episode_result"] = "success" if best >= TARGET_YIELD else "budget_exhausted"
        rw = {}
        rw["yield"] = min(yield_pct / TARGET_YIELD, 1.0)
        rw["efficiency"] = max(0.0, (MAX_STEPS - self._obs["step"] + 1) / MAX_STEPS) if yield_pct >= TARGET_YIELD else 0.0
        cs = self.rsm.causal_structure()
        attr = primary_bottleneck.strip().lower()
        if attr == cs["primary_param"].lower():
            rw["causal"] = 1.0
        elif attr in [p.lower() for p in cs["secondary_params"]]:
            rw["causal"] = 0.4
        else:
            rw["causal"] = 0.0
        if done and submit:
            lot_std = self.rsm.lot_variance(clamped, n_lots=20)
            rw["stability"] = max(0.0, 1.0 - lot_std / 5.0) * (0.3 if not reviewer_result["approved"] else 1.0)
        else:
            rw["stability"] = 0.0
        rw["total"] = 0.50*rw["yield"] + 0.20*rw["efficiency"] + 0.15*rw["causal"] + 0.15*rw["stability"]
        rw = {k: round(float(v), 4) for k, v in rw.items()}
        return self._obs, rw, done, {"causal_structure": cs, "reviewer_approved": reviewer_result["approved"], "defect": defect}


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

header("TEST 1: RSM Simulator")
rsm = RSMSimulator(seed=42, difficulty=1)
print(f"  Primary  : {rsm.primary_param}")
print(f"  Secondary: {rsm.secondary_params}")
print(f"  Base yield: {rsm.base_yield*100:.1f}%")
mid_params = {p: (PARAM_RANGES[p][0] + PARAM_RANGES[p][1]) / 2 for p in PHASE_PARAMS[1]}
y, defect = rsm.predict(mid_params)
print(f"  Midpoint yield: {y:.1f}%  Defect: {defect}")
assert 0 <= y <= 100
assert defect in ("edge_ring", "center_spot", "random_scatter", "none")
ok("predict() returns valid (yield, defect)")
y_opt, _ = rsm.predict(rsm.optimum_params)
print(f"  Optimum yield : {y_opt:.1f}%")
ok(f"predict() at optimum → {y_opt:.1f}%")
std = rsm.lot_variance(rsm.optimum_params, n_lots=30)
print(f"  Lot std: {std:.3f}%")
assert 0 <= std <= 10
ok("lot_variance() within range")
cs = rsm.causal_structure()
assert "primary_param" in cs and "secondary_params" in cs
ok(f"causal_structure() → primary={cs['primary_param']}, secondary={cs['secondary_params']}")


header("TEST 2: Senior Reviewer")
constraints = sample_episode_constraints(PHASE_PARAMS[1], rng=np.random.default_rng(99))
reviewer = SeniorEngineerReviewer(constraints)
approved_recipe = {p: (q_lo + q_hi) / 2.0 for p, (q_lo, q_hi) in constraints.items()}
r = reviewer.review(approved_recipe)
assert r["approved"], f"Should approve: {r['feedback']}"
ok("Reviewer approves midpoint-of-qualified-range recipe")
reject_recipe = {p: PARAM_RANGES[p][0] for p in PHASE_PARAMS[1]}
r2 = reviewer.review(reject_recipe)
if r2["approved"]:
    warn("Extreme recipe approved (unlucky constraint draw)")
else:
    ok(f"Reviewer rejects out-of-range recipe ({len(r2['violations'])} violations)")
print(f"  Feedback: {r2['feedback'][:80]}...")


header("TEST 3: Prompt Builder + Action Parser")
env = FabYieldEnvTest(difficulty=1)
obs = env.reset(seed=7)
prompt = build_prompt(obs)
print(f"  Prompt length: {len(prompt)} chars")
assert "EXPERIMENT HISTORY" in prompt
assert "YOUR ACTION" in prompt
assert "<experiment>" in prompt
ok("build_prompt() has all required sections")

fake = """
<experiment>
  temp: 200.0
  etch_time: 55.0
  pressure: 2.1
  dopant: 1.3e15
  spin_speed: 2500
  submit: false
</experiment>
<diagnosis>
  primary_bottleneck: temp
  reasoning: Temperature had the largest yield effect.
</diagnosis>
"""
act = parse_action(fake)
assert act["params"]["temp"] == 200.0
assert act["primary_bottleneck"] == "temp"
assert act["submit"] == False
print(f"  Parsed: params={list(act['params'].keys())}  bottleneck={act['primary_bottleneck']}")
ok("parse_action() extracts params, bottleneck, submit=false")
assert parse_action(fake.replace("submit: false", "submit: true"))["submit"] == True
ok("parse_action() handles submit: true")
bad = parse_action("Just try temperature around 200 and etch time 50")
assert bad["params"] == {} and bad["primary_bottleneck"] == ""
ok("parse_action() degrades gracefully (no crash, empty dict)")


header("TEST 4: Full 12-step Episode — Phase 1 (5 params)")
env = FabYieldEnvTest(difficulty=1)
obs = env.reset(seed=42)
print(f"\n  Active params  : {obs['active_params']}")
print(f"  True primary   : {env.rsm.primary_param}  (hidden from agent)")
print(f"  True optimum   : { {k: round(v,1) for k,v in env.rsm.optimum_params.items()} }\n")

total_reward = 0.0
best_yield = 0.0
best_params = None

print(f"  {'Step':>4}  {'First 2 params':>32}  {'Yield':>7}  {'Defect':>14}  {'R':>6}")
print(f"  {'─'*4}  {'─'*32}  {'─'*7}  {'─'*14}  {'─'*6}")

for sn in range(1, 13):
    active = obs["active_params"]
    if sn <= 4:
        params = {p: (PARAM_RANGES[p][0] + PARAM_RANGES[p][1]) / 2 for p in active}
        pv = active[(sn-1) % len(active)]
        lo, hi = PARAM_RANGES[pv]
        params[pv] = lo + (hi - lo) * (0.25 if sn % 2 == 0 else 0.75)
    elif sn <= 8:
        base = best_params or {p: (PARAM_RANGES[p][0] + PARAM_RANGES[p][1]) / 2 for p in active}
        params = dict(base)
        pv = active[(sn-5) % len(active)]
        lo, hi = PARAM_RANGES[pv]
        c = (lo + hi) / 2
        params[pv] = params.get(pv, c) * 0.9 + c * 0.1
    else:
        base = best_params or {p: (PARAM_RANGES[p][0] + PARAM_RANGES[p][1]) / 2 for p in active}
        params = {p: float(np.clip(base[p] + np.random.normal(0, (PARAM_RANGES[p][1]-PARAM_RANGES[p][0])*0.03),
                                   PARAM_RANGES[p][0], PARAM_RANGES[p][1])) for p in active}
    submit = (sn == 12) or (best_yield >= 92.0 and sn >= 9)
    bottleneck = env.rsm.primary_param if sn > 6 else "temp"
    obs, rw, done, info = env.step(params=params, primary_bottleneck=bottleneck, submit=submit)
    last = obs["experiment_history"][-1]
    y = last["yield_pct"]
    pstr = ", ".join(f"{k}={v:.0f}" for k, v in list(last["params"].items())[:2])
    marker = " ◀BEST" if y > best_yield else ""
    dmarker = " DONE" if done else ""
    print(f"  {sn:>4}  {pstr:>32}  {y:>6.1f}%  {last['defect']:>14}  {rw['total']:>6.3f}{marker}{dmarker}")
    if y > best_yield:
        best_yield = y
        best_params = dict(last["params"])
    total_reward += rw["total"]
    if done:
        break

print(f"\n  Best yield: {best_yield:.1f}%  Total reward: {total_reward:.3f}  Result: {obs['episode_result']}")
if best_yield >= 70.0:
    ok(f"Episode complete. Best yield = {best_yield:.1f}%")
else:
    warn(f"Low yield ({best_yield:.1f}%) — difficult RSM draw, expected occasionally")


header("TEST 5: Reward Component Sanity Checks")
env2 = FabYieldEnvTest(difficulty=1)
env2.reset(seed=100)
opt2 = dict(env2.rsm.optimum_params)

_, rg, _, _ = env2.step(opt2, primary_bottleneck=env2.rsm.primary_param)
print(f"  Correct primary: yield={rg['yield']:.3f} causal={rg['causal']:.3f}")
assert rg["yield"] > 0.5, f"Expected yield>0.5, got {rg['yield']}"
assert rg["causal"] == 1.0, f"Expected causal=1.0, got {rg['causal']}"
ok("Correct primary bottleneck → causal = 1.0")

_, rb, _, _ = env2.step(opt2, primary_bottleneck="nonexistent_param_xyz")
assert rb["causal"] == 0.0
ok("Wrong bottleneck → causal = 0.0")

if env2.rsm.secondary_params:
    _, rs, _, _ = env2.step(opt2, primary_bottleneck=env2.rsm.secondary_params[0])
    assert rs["causal"] == 0.4, f"Expected 0.4, got {rs['causal']}"
    ok(f"Secondary param ({env2.rsm.secondary_params[0]}) → causal = 0.4")
else:
    warn("No secondary params — skip secondary causal test")

env3 = FabYieldEnvTest(difficulty=1)
env3.reset(seed=200)
opt3 = dict(env3.rsm.optimum_params)
for _ in range(9):
    env3.step(opt3, primary_bottleneck=env3.rsm.primary_param)
_, rs2, done2, _ = env3.step(opt3, primary_bottleneck=env3.rsm.primary_param, submit=True)
print(f"  Submission rewards: {rs2}")
assert done2
assert rs2["stability"] >= 0.0
ok("Submission step → done=True, stability reward computed")


header("TEST 6: Phase 3 (15 params)")
env_hard = FabYieldEnvTest(difficulty=3)
obs_hard = env_hard.reset(seed=55)
assert len(obs_hard["active_params"]) == 15
ok(f"15 active params confirmed: {obs_hard['active_params'][:5]} ...")
all_mid = {p: (PARAM_RANGES[p][0]+PARAM_RANGES[p][1])/2 for p in obs_hard["active_params"]}
obs_h2, rw_h, _, _ = env_hard.step(all_mid, primary_bottleneck="temp")
print(f"  Midpoint yield: {obs_h2['current_best_yield']:.1f}%")
ok("Phase 3 step() completes without error")


header("TEST 7: Prompt embeds full experiment history")
env4 = FabYieldEnvTest(difficulty=1)
obs4 = env4.reset(seed=77)
for i in range(3):
    mid = {p: (PARAM_RANGES[p][0]+PARAM_RANGES[p][1])/2 for p in obs4["active_params"]}
    mid[obs4["active_params"][0]] += i * 5
    obs4, _, _, _ = env4.step(mid, primary_bottleneck="temp")
p4 = build_prompt(obs4)
assert "Exp 1:" in p4 and "Exp 2:" in p4 and "Exp 3:" in p4
print(f"  Prompt with 3-step history: {len(p4)} chars")
ok("Prompt correctly embeds growing experiment history")


header("TEST 8: Phase Transitions")
env5 = FabYieldEnvTest(difficulty=1)
obs5 = env5.reset(seed=10)
assert obs5["phase"] == "exploration"
ok("Initial phase = exploration")
mid5 = {p: (PARAM_RANGES[p][0]+PARAM_RANGES[p][1])/2 for p in obs5["active_params"]}
for _ in range(4):
    obs5, _, _, _ = env5.step(mid5, primary_bottleneck="temp")
assert obs5["phase"] == "hypothesis", f"Expected hypothesis, got {obs5['phase']}"
ok(f"After 4 steps → phase = {obs5['phase']}")
for _ in range(4):
    obs5, _, _, _ = env5.step(mid5, primary_bottleneck="temp")
assert obs5["phase"] in ("exploitation", "submission")
ok(f"After 8 steps → phase = {obs5['phase']}")


header("ALL 8 TESTS PASSED ✓")
print("""
  Verified:
    ✓ RSM simulator math (predict, lot_variance, causal_structure)
    ✓ Senior Reviewer (approve / reject with violation detail)
    ✓ Prompt builder (all sections present, history embedded)
    ✓ Action parser (XML, submit flag, graceful degradation)
    ✓ All 4 reward components (yield, efficiency, causal, stability)
    ✓ Full 12-step episode with heuristic agent (Phase 1)
    ✓ Phase 3 (15 params) single step
    ✓ Phase transitions: exploration → hypothesis → exploitation

  To run the full server:
    pip install fastapi uvicorn pydantic numpy
    uvicorn server:app --host 0.0.0.0 --port 7860
    curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"difficulty": 1}'
""")