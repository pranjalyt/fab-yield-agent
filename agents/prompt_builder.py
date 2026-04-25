"""
agents/prompt_builder.py — Builds the structured text prompt for the LLM agent.

Imports from environment.env (Pydantic models) — NOT environment.models.
Field names match env.py's FabObservation exactly.
"""
import re
from environment.env import FabObservation, FabAction

def build_prompt(obs: FabObservation) -> str:
    """
    Build the full text prompt from a FabObservation (Pydantic model from env.py).
    This is what gets passed to the LLM at each training step.
    """
    lines = []

    # 🛑 NEW: 1. Task Target / Objective Binning
    lines.append("═══ OBJECTIVE ═══")
    lines.append(f"  Task: {obs.task_target}")
    lines.append("")

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("═══ CURRENT STATE ═══")
    lines.append(f"- Step: {obs.step}/{12}")
    lines.append(f"- Phase: {obs.phase.upper()} — {obs.phase_hint}")
    lines.append(f"- Budget remaining: {obs.budget_remaining} experiments")
    lines.append(f"- Current best yield: {obs.current_best_yield:.1f}%")
    lines.append(f"- Target yield: >{obs.target_yield:.0f}%")
    lines.append("")

    # ── Active parameters and ranges ─────────────────────────────────────────
    lines.append("═══ ACTIVE PARAMETERS (with allowed ranges) ═══")
    for p in obs.active_params:
        lo, hi = obs.param_ranges[p]
        lines.append(f"  {p}: [{_fmt(lo)} — {_fmt(hi)}]")
    lines.append("")

    # ── Experiment history ────────────────────────────────────────────────────
    lines.append("═══ EXPERIMENT HISTORY ═══")
    if not obs.experiment_history:
        lines.append("  No experiments yet.")
    else:
        for r in obs.experiment_history:
            param_str = ", ".join(f"{k}={_fmt(v)}" for k, v in r.params.items())
            guess_str = f"  Your guess: {r.primary_bottleneck_guess}" \
                        if r.primary_bottleneck_guess else ""
            lines.append(
                f"  Exp {r.step}: {param_str}\n"
                f"    → Yield: {r.yield_pct:.1f}%  Defect: {r.defect}{guess_str}"
            )
    lines.append("")

    # 🛑 NEW: 2. Multi-Agent Feedback (Reviewer & Financial Controller)
    feedback_present = False
    if obs.reviewer_feedback:
        lines.append(f"⚠ SENIOR REVIEWER FEEDBACK: {obs.reviewer_feedback}")
        feedback_present = True
    
    if obs.financial_feedback:
        lines.append(f"⚠ FINANCIAL CONTROLLER FEEDBACK: {obs.financial_feedback}")
        feedback_present = True
        
    if feedback_present:
        lines.append("")

    # ── Phase instruction ─────────────────────────────────────────────────────
    phase_instructions = {
        "exploration": (
            "PHASE: EXPLORATION — Vary parameters broadly. "
            "Goal: understand which parameters matter most. "
            "Try experiments that differ significantly from each other."
        ),
        "hypothesis": (
            "PHASE: HYPOTHESIS — You have data. Test your causal theory. "
            "Isolate your suspected primary bottleneck by changing it while "
            "holding others near their best-so-far values."
        ),
        "exploitation": (
            "PHASE: EXPLOITATION — Converge on the optimum. "
            "Make fine adjustments around your best result. "
            "Don't explore — exploit."
        ),
        "submission": (
            "PHASE: SUBMISSION — Final experiment. Submit your best recipe. "
            "Set submit: true. The Senior Engineer will review against "
            "production qualification constraints."
        ),
    }
    lines.append(phase_instructions.get(obs.phase, ""))
    lines.append("")

    # 🛑 NEW: 3. XML action template with <think> tag for Test-Time Compute
    lines.append("═══ YOUR ACTION ═══")
    lines.append("You MUST think step-by-step before acting. Analyze the history and constraints.")
    lines.append("Output ONLY the XML below, no text outside tags:")
    lines.append("")
    lines.append("<think>")
    lines.append("  [Write your causal analysis and hypothesis testing strategy here]")
    lines.append("</think>")
    lines.append("<diagnosis>")
    lines.append(f"  primary_bottleneck: {obs.active_params[0]}")
    lines.append("  reasoning: [one sentence causal reasoning]")
    lines.append("</diagnosis>")
    lines.append("<experiment>")
    for p in obs.active_params:
        lo, hi = obs.param_ranges[p]
        default = (lo + hi) / 2
        lines.append(f"  {p}: {_fmt(default)}")
    lines.append("  submit: false")
    lines.append("</experiment>")

    return "\n".join(lines)


def parse_action(text: str, active_params: list) -> FabAction:
    """
    Parse LLM XML output → FabAction (Pydantic model from env.py).
    Robust to whitespace, minor formatting variation, missing tags.
    """
    params = {}
    primary_bottleneck = active_params[0] if active_params else "temp"
    reasoning = ""
    submit = False
    think = ""

    # 🛑 NEW: 4. Extract <think> block
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        think = think_match.group(1).strip()

    # Parse <experiment> block
    exp_match = re.search(r"<experiment>(.*?)</experiment>", text, re.DOTALL | re.IGNORECASE)
    if exp_match:
        for line in exp_match.group(1).strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip().lower(), val.strip()
            if key == "submit":
                submit = val.lower() in ("true", "yes", "1")
            elif key in active_params:
                try:
                    params[key] = float(val)
                except ValueError:
                    pass

    # Parse <diagnosis> block
    diag_match = re.search(r"<diagnosis>(.*?)</diagnosis>", text, re.DOTALL | re.IGNORECASE)
    if diag_match:
        for line in diag_match.group(1).strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key, val = key.strip().lower(), val.strip()
            if "bottleneck" in key:
                primary_bottleneck = val.lower()
            elif "reason" in key:
                reasoning = val

    return FabAction(
        think=think, # Passed to the environment for scoring
        params=params,
        primary_bottleneck=primary_bottleneck,
        reasoning=reasoning,
        submit=submit,
    )


def _fmt(v: float) -> str:
    """Format a float cleanly for display in prompts."""
    if v == 0:
        return "0"
    if abs(v) >= 1e13 or (abs(v) < 0.01 and v != 0):
        return f"{v:.2e}"
    if abs(v) >= 1000:
        return f"{v:.0f}"
    return f"{v:.2f}"