"""
prompt_builder.py — Build text prompts from FabObservation.

This is the agent's entire view of the world. The quality of this
prompt directly affects training — structured, information-dense prompts
lead to better GRPO trajectories.
"""

from typing import Dict, Any


def build_prompt(obs: Dict[str, Any]) -> str:
    """
    Build the agent's text prompt from a FabObservation dict.

    The prompt includes:
    - Episode meta (step, budget, target)
    - Full experiment history (params + yield + defect)
    - Phase hint (guides long-horizon planning)
    - Reviewer feedback (if recipe was rejected last step)
    - Parameter ranges (so agent knows the valid space)
    - Output format instructions (XML tags for structured parsing)
    """
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

    # ─── Header ──────────────────────────────────────────────────────────────
    prompt = f"""You are a semiconductor process engineer optimizing wafer yield at a chip fab.

═══ CURRENT STATE ═══════════════════════════════════════════
  Experiments run:    {step - 1}/{step - 1 + budget}
  Budget remaining:   {budget} experiments
  Current best yield: {best:.1f}%
  Target yield:       >{target}%
  Phase:              {phase.upper()} — {phase_hint}
"""

    # ─── Reviewer feedback (if any) ──────────────────────────────────────────
    if reviewer_feedback:
        prompt += f"""
⚠ REVIEWER FEEDBACK FROM LAST SUBMISSION:
  {reviewer_feedback}
"""

    # ─── Experiment history ───────────────────────────────────────────────────
    prompt += "\n═══ EXPERIMENT HISTORY ══════════════════════════════════════\n"
    if not history:
        prompt += "  No experiments yet.\n"
    else:
        for rec in history:
            param_str = ", ".join(
                f"{k}={_fmt(v)}" for k, v in rec["params"].items()
            )
            prompt += f"  Exp {rec['step']}: {param_str}\n"
            prompt += f"    → Yield: {rec['yield_pct']:.1f}%  |  Defects: {rec['defect']}"
            if rec.get("primary_bottleneck_guess"):
                prompt += f"  |  Your guess: {rec['primary_bottleneck_guess']}"
            prompt += "\n"

    # ─── Parameter space ──────────────────────────────────────────────────────
    prompt += "\n═══ PARAMETER SPACE ══════════════════════════════════════════\n"
    for param in active_params:
        if param in param_ranges:
            lo, hi = param_ranges[param]
            prompt += f"  {param}: [{_fmt(lo)} — {_fmt(hi)}]\n"

    # ─── Action instructions ──────────────────────────────────────────────────
    submit_hint = " (set submit: true to end episode)" if step >= 9 else ""

    prompt += f"""
═══ YOUR ACTION ══════════════════════════════════════════════
Choose your next experiment parameters{submit_hint}.
State which parameter you believe is the PRIMARY yield bottleneck and why.

Defect guide:
  edge_ring      → pressure or gas flow is off (etch non-uniformity)
  center_spot    → temperature or RF power issue (hotspot)
  random_scatter → dopant concentration problem
  none           → you are in a good region of parameter space

Output your response in this EXACT format (required for parsing):
<experiment>
"""
    for param in active_params:
        if param in param_ranges:
            lo, hi = param_ranges[param]
            prompt += f"  {param}: [{_fmt(lo)}-{_fmt(hi)}]\n"
    prompt += "  submit: [true/false]\n"
    prompt += """</experiment>
<diagnosis>
  primary_bottleneck: [parameter name]
  reasoning: [one sentence — what evidence supports your choice]
</diagnosis>
"""

    return prompt


def _fmt(v: float) -> str:
    """Format a float cleanly (scientific for very small/large, else decimal)."""
    if v == 0:
        return "0"
    if abs(v) >= 1e13 or (abs(v) < 0.01 and v != 0):
        return f"{v:.2e}"
    if abs(v) >= 1000:
        return f"{v:.0f}"
    return f"{v:.2f}"


def parse_action(text: str) -> Dict[str, Any]:
    """
    Parse XML-tagged agent output into a structured action dict.

    Handles common LLM formatting errors gracefully:
    - Missing tags → empty dict (env uses param midpoints)
    - Non-numeric values → skipped
    - Extra whitespace/newlines → stripped
    """
    import re

    action = {
        "params": {},
        "primary_bottleneck": "",
        "reasoning": "",
        "submit": False,
    }

    # Extract <experiment> block
    exp_match = re.search(r"<experiment>(.*?)</experiment>", text, re.DOTALL | re.IGNORECASE)
    if exp_match:
        for line in exp_match.group(1).strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()
            if key == "submit":
                action["submit"] = val.lower() in ("true", "yes", "1")
            else:
                try:
                    action["params"][key] = float(val)
                except ValueError:
                    pass  # skip malformed values

    # Extract <diagnosis> block
    diag_match = re.search(r"<diagnosis>(.*?)</diagnosis>", text, re.DOTALL | re.IGNORECASE)
    if diag_match:
        for line in diag_match.group(1).strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()
            if "bottleneck" in key:
                action["primary_bottleneck"] = val
            elif "reasoning" in key or "reason" in key:
                action["reasoning"] = val

    return action