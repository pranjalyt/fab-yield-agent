"""
prompt_builder.py — Builds the structured text prompt for the LLM agent.

This is the full cognitive context the agent receives at each step.
The agent outputs XML tags which the env parses back into FabAction.
"""
import re
from typing import Optional, Dict
from environment.models import FabObservation, FabAction


SYSTEM_PROMPT = """You are an expert semiconductor process integration engineer.
Your goal is to optimize wafer yield by running experiments and identifying 
the root cause of yield loss.

You have a limited experiment budget. Think like a scientist:
- In early steps (exploration): vary parameters broadly to map the response surface
- In middle steps (hypothesis): focus on your suspected bottleneck parameter
- In late steps (exploitation): converge on the optimum
- On step 12 or submit: propose your final process recipe

ALWAYS output your response in EXACTLY this XML format:
<experiment>
  temp: [value]
  etch_time: [value]
  pressure: [value]
  dopant: [value]
  spin_speed: [value]
</experiment>
<diagnosis>
  primary_bottleneck: [parameter name, exactly as listed in active params]
  reasoning: [one sentence explaining your causal reasoning]
  submit: [true/false — true only when ready to submit final recipe]
</diagnosis>"""


def build_prompt(obs: FabObservation) -> str:
    """
    Construct the full text prompt from a FabObservation.
    This is what gets passed to the LLM at each step.
    """
    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(f"CURRENT STATE:")
    lines.append(f"- Step: {obs.step + 1}/{12}")
    lines.append(f"- Phase: {obs.phase.upper()}")
    lines.append(f"- Budget remaining: {obs.budget_remaining} experiments")
    lines.append(f"- Current best yield: {obs.current_best_yield:.1f}%")
    lines.append(f"- Target yield: >{obs.target_yield:.0f}%")
    lines.append("")

    # ── Active parameters and ranges ─────────────────────────────────────────
    lines.append("ACTIVE PARAMETERS (with allowed ranges):")
    for p in obs.active_params:
        lo, hi = obs.param_ranges[p]
        lines.append(f"  {p}: [{lo}, {hi}]")
    lines.append("")

    # ── Experiment history ────────────────────────────────────────────────────
    if obs.experiment_history:
        lines.append("EXPERIMENT HISTORY:")
        for r in obs.experiment_history:
            param_str = ", ".join(
                f"{k}={v:.4g}" for k, v in r.params.items()
            )
            lines.append(
                f"  Exp {r.step}: {param_str} → "
                f"Yield: {r.yield_pct}%, Defect: {r.defect}"
            )
        lines.append("")

    # ── Agent's running hypothesis ────────────────────────────────────────────
    if obs.current_hypothesis:
        lines.append(f"YOUR CURRENT HYPOTHESIS: {obs.current_hypothesis}")
        lines.append("")

    # ── Reviewer feedback (if any) ────────────────────────────────────────────
    if obs.reviewer_feedback:
        lines.append(f"⚠️  REVIEWER FEEDBACK: {obs.reviewer_feedback}")
        lines.append("")

    # ── Phase-specific instruction ────────────────────────────────────────────
    phase_instructions = {
        "exploration": (
            "PHASE: EXPLORATION — Vary parameters broadly. "
            "Your goal is to understand which parameters matter most. "
            "Try experiments that differ significantly from each other."
        ),
        "hypothesis": (
            "PHASE: HYPOTHESIS — You have data. Now test your causal theory. "
            "Isolate your suspected primary bottleneck by changing it while "
            "holding others near their best-so-far values."
        ),
        "exploitation": (
            "PHASE: EXPLOITATION — Converge on the optimum. "
            "Make fine adjustments around your best result so far. "
            "You're close — don't explore, exploit."
        ),
        "submission": (
            "PHASE: SUBMISSION — This is your final experiment. "
            "Submit your best process recipe. "
            "Set submit: true in your diagnosis. "
            "The Senior Engineer will review your recipe against "
            "production qualification constraints."
        ),
    }
    lines.append(phase_instructions.get(obs.phase, ""))
    lines.append("")

    # ── Action request ────────────────────────────────────────────────────────
    lines.append("YOUR ACTION:")
    lines.append(
        "Output ONLY the XML tags below. No preamble, no explanation outside the tags."
    )

    # Build XML template with current best as defaults
    lines.append("")
    lines.append("<experiment>")
    for p in obs.active_params:
        lo, hi = obs.param_ranges[p]
        default = (lo + hi) / 2
        lines.append(f"  {p}: {default:.4g}")
    lines.append("</experiment>")
    lines.append("<diagnosis>")
    lines.append(f"  primary_bottleneck: {obs.active_params[0]}")
    lines.append("  reasoning: [your one-sentence causal reasoning]")
    lines.append("  submit: false")
    lines.append("</diagnosis>")

    return "\n".join(lines)


def parse_action(text: str, active_params: list) -> FabAction:
    """
    Parse the LLM's XML output into a FabAction.
    Robust to extra whitespace and minor formatting variations.
    """
    params = {}

    # Extract experiment block
    exp_match = re.search(r"<experiment>(.*?)</experiment>", text, re.DOTALL)
    if exp_match:
        exp_block = exp_match.group(1)
        for p in active_params:
            # Match "param: value" patterns
            m = re.search(rf"{re.escape(p)}\s*:\s*([0-9eE+\-.]+)", exp_block)
            if m:
                try:
                    params[p] = float(m.group(1))
                except ValueError:
                    pass

    # Extract diagnosis block
    primary_bottleneck = active_params[0]  # fallback
    reasoning = ""
    submit = False

    diag_match = re.search(r"<diagnosis>(.*?)</diagnosis>", text, re.DOTALL)
    if diag_match:
        diag_block = diag_match.group(1)

        pb_match = re.search(r"primary_bottleneck\s*:\s*(\S+)", diag_block)
        if pb_match:
            primary_bottleneck = pb_match.group(1).strip().lower()

        r_match = re.search(r"reasoning\s*:\s*(.+)", diag_block)
        if r_match:
            reasoning = r_match.group(1).strip()

        s_match = re.search(r"submit\s*:\s*(true|false)", diag_block, re.IGNORECASE)
        if s_match:
            submit = s_match.group(1).lower() == "true"

    return FabAction(
        params=params,
        primary_bottleneck=primary_bottleneck,
        reasoning=reasoning,
        submit=submit,
    )