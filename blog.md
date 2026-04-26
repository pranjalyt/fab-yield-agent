# Teaching an LLM to Think Like a Chip Engineer

**Team TensorTitans · OpenEnv India 2026 · Theme: Long-Horizon Planning + Multi-Agent Interactions + Self-Improvement + World Modeling . bonus: Fleet AI + Professional Tasks**

---

## The moment it clicked

we spend the few days in the drdo and had a talk to many scientist and engineer working there, they told us about the how it is very important for the country to be self reliant in the field of semiconductor and they told us that they is one thing which is casing loss of the millions of dollars every year due to the low yield of the wafers,
 most common reason for those are **Process Excursions**-
    **Thermal Non-Uniformity:** If the heater is 1°C hotter on the left side than the right, the chemicals will etch deeper on one side. The chips on the left will be "over-cooked" (dead), while the ones on the right are perfect.
    **Chemical Contamination:** If the gases used for etching have even 0.0001% of the wrong impurity (like trace metals), it can change the electrical properties of the silicon, causing "leakage" where electricity jumps where it shouldn't.
    **Focus Errors (Lithography):** Printing a chip is like using a high-tech projector. If the lens is slightly out of focus due to vibration or a tiny particle on the backside of the wafer, the billions of transistors won't align, and the chip won't work.

To save those millions of dollars we have to achieve lighitng fast speed to adjust the differnt parameters which is very hard to fro human to do at tym in samll interval of time other is accuracy and the perfection. 

After a deep research on that field we came to the **Fab yield optimization** 

so we thought that we can use the ai agent to solve this problem as it is much better faster than most of the humans. 

We tested this. The results were not encouraging.

The models we tried would pick a variable, commit to it, and then keep doubling down even when the data was screaming otherwise. They'd spot a correlation and call it a cause. They'd confidently deliver a wrong answer, and when you pushed back, they'd find a new way to justify the same wrong answer instead of actually reconsidering.

For most use cases, this is manageable. You retry. You rephrase. No one gets hurt.

But then we started thinking about semiconductor fabrication

As to achevie the perfection in ai agent we have to create a environment in which ai agent can learn and wht is better than the openenv at the movement. 

---

## Why fabrication specifically

A chip fab is the most controlled manufacturing environment humans have ever built. Positive pressure rooms. Vibration isolation. Particle filters catching things measured in nanometers. Temperature regulated to fractions of a degree. And it still fights physics constantly.

Here's what a process engineer actually deals with on a normal Tuesday:

- A storm passes 60 miles away. Atmospheric pressure shifts slightly. The etching chamber drifts. Yield drops on a lot that was running perfectly six hours ago.
- A delivery truck parks near the building. Microscopic vibrations travel through the floor. Deposition uniformity shifts by fractions of a percent. That's all it takes.
- The etchant in the tank has run 200 cycles. Its potency has quietly decayed. The effective etch rate is 0.7% slower than nominal. No alarm fires. Yield just degrades.
- Machine A runs at 600°C. The ambient temperature around Machine B changes. Machine B drifts. Engineers know to watch for this, but modeling it precisely is another problem entirely.

Before AI, engineers managed this with Statistical Process Control — wide safety margins, conservative targets, human oversight on significant drift. If a process needed exactly 500°C, you accepted anything from 495 to 505. That 1% buffer was the cost of operating at human reaction speed.

At 2nm — where a feature is a few silicon atoms wide — the buffer doesn't exist. A 1% deviation isn't degraded yield. It's total failure.

**TSMC engineers don't tune parameters by hand anymore. The physics has outrun human reaction time.**

Then came the second part of the realization: India is building its own fabs. The India Semiconductor Mission, DRDO — these aren't science projects. They're serious industrial programs. And they don't need 2nm. They need extreme reliability on 28nm to 65nm nodes. The reasoning gap we kept running into doesn't disappear at lower nodes. It's just as real, and just as costly.

So we asked one question, and it became the whole project:

> **What if we could train an LLM to think like a process integration engineer?**

That question became `fab-yield-agent`.

---

## What we actually built (and what we didn't)

Let's be precise about this, because it matters.

**We built the environment on the openenv**

`fab-yield-agent` is an OpenEnv-compliant RL training environment. An LLM acts as the agent and gets trained inside it. What we designed is the world  the physics the agent has to navigate, what it can observe, what actions it can take, and what it gets rewarded for.

Think of it like this: we didn't train the chess engine. We built the chessboard, defined the pieces, wrote the rules, and set up the training match.

### The hidden world

At the core of the environment is a **Response Surface Model (RSM) Simulator** with 15 process parameters: temperature, pressure, etch time, dopant concentration, spin speed, gas flow, humidity, vibration level, and more. These parameters interact in nonlinear, partially correlated ways. Some correlations are causal. Some are coincidental. The agent never sees the true equations.

It only sees what an engineer sees: **output yield and defect patterns after each experimental run.**

The agent's job: figure out the optimal process recipe in **12 experiments or fewer**, then submit it for Senior Engineer review.

### The four episode phases

| Phase | Experiments | What the Agent Must Do |
|---|---|---|
| Exploration | 1–4 | Sweep the parameter space broadly. Don't get attached to any hypothesis. |
| Hypothesis Formation | 5–8 | Identify the primary bottleneck. Start testing it specifically. |
| Exploitation | 9–11 | Converge on the optimum. Stop exploring. |
| Final Submission | 12 | Commit to a recipe that passes qualification constraints. |

This structure mirrors the actual workflow of a process integration engineer. And it forces a genuinely hard strategic decision at every phase boundary: do I keep exploring, or do I commit?

An agent that over-explores in phase 3 has burned its budget. An agent that over-exploits in phase 1 has committed to an untested hypothesis. Neither is recoverable.

### Why it's genuinely hard to game

Every episode, we randomize three things:

1. Which parameters are the true bottlenecks (the agent doesn't know)
2. The severity of lot-to-lot process variance
3. The Senior Engineer's qualification thresholds

There's no fixed target to memorize. An agent that pattern-matches will fail on novel episodes. It has to develop actual causal reasoning to generalize.

---

## The reward system — teaching judgment, not just outcomes

This is where most RL environments quietly fail. Make the reward too simple and agents find shortcuts that look like learning but aren't. Make it too sparse and training never starts.

We designed a **four-component composable reward**:

| Component | Weight | What It Actually Measures |
|---|---|---|
| Yield Improvement | 50% | Dense signal — every step, did yield go up? |
| Efficiency | 20% | Sparse — did you hit the target within the 12-experiment budget? |
| Causal Attribution | 15% | Did you correctly identify *why* yield changed? |
| Process Stability | 15% | Is the submitted recipe robust to lot-to-lot variance? |

The **Causal Attribution** component is the novel one. We parse the agent's natural-language reasoning — the `primary_bottleneck` and `reasoning` fields it outputs in structured XML — and score whether its stated hypothesis matches the true hidden causal structure of the episode.

This means: **we're not just rewarding outcomes. We're rewarding correct thinking.**

An agent that stumbles onto the right recipe for the wrong reasons gets penalized. An agent that reasons correctly but gets unlucky gets partial credit.

That distinction turned out to matter more than we expected. More on that in a moment.

---

## Training: what we ran, what we saw

We used **GRPO (Group Relative Policy Optimization)** via HuggingFace TRL + Unsloth, running on A10G GPUs via HuggingFace compute credits. The training loop connects directly to the environment — no static datasets. The agent generates real rollouts, the environment scores them, the reward signal drives improvement.

### The curriculum

We structured training across three difficulty levels:

- **Easy**: 5 parameters, yield target >80%, low noise. The agent learns the basic episode loop.
- **Medium**: 10 parameters, yield target >88%, moderate variance. The agent can no longer brute-force the parameter space.
- **Hard**: 15 parameters, yield target >92%, high lot-to-lot variance. Every episode is different. Generalization is the only path.

We escalate difficulty only when the agent reliably hits the target at the current level. This isn't just a nice training detail — it's load-bearing. We tried jumping straight to hard mode early on. The agent developed degenerate strategies that happened to work on hard episodes but failed completely on novel configurations. Curriculum learning fixed that.

### The reward curves

![Agent Performance: Mean Yield Reward — Run 1 (40 steps). EMA trend holds stable between 0.34–0.36, showing a well-conditioned signal.]

*Mean Yield Reward across training steps — Run 1 (40 steps). The EMA trend line shows a stable baseline, not collapsing, not exploding. The reward signal is alive and well-conditioned.*

![Agent Performance: Mean Yield Reward — Run 2 (90 steps). Reward dips mid-run during curriculum escalation, then begins recovering — a classic curriculum learning signature.]

*Run 2, extended to 90 steps. The mid-run dip around step 15–25 corresponds to escalation from Easy to Medium difficulty. The agent takes a hit, then adapts. The recovery is the signal that training is working.*

These aren't perfect upward curves — and that's fine. In a curriculum environment with genuine difficulty escalation, you expect dips at transition points. What you don't want is collapse or flatline. Neither happened.

### Visualizing the Chaos: The Human vs. Machine Game

To really understand why this is a job for AI, we built an interactive "Yield Simulator" game. 

In this game, the user is given 15 dynamic physical sliders—temperature, pressure, gas flow, etc.—that start drifting randomly. As a human, you try to keep them in the "green zone" to save the wafer. **You fail almost immediately.** 

The "game" makes two things crystal clear:
1. **Reaction Time:** Humans can’t monitor 15 variables at once when they are shifting in milliseconds.
2. **Coupled Variables:** If you fix the pressure, the temperature spikes. If you fix the temperature, the gas flow drops. It’s a multi-dimensional puzzle that humans aren't wired to solve in real-time.




### The loss picture

GRPO loss doesn't behave like supervised loss. It oscillates — sometimes dramatically — because the signal is relative: how did this policy do compared to other rollouts in the same group? As the policy improves, the baseline shifts, which changes the gradient. Wide oscillations in raw loss are expected. What you watch for is EMA stabilization.

![Policy Optimization: GRPO Loss Convergence — Run 1 (40 steps). EMA trend stabilizes near zero after initial oscillations.]

*GRPO Loss — Run 1. The EMA trend (bold red) stabilizes near zero, indicating the policy gradient updates are landing in a reasonable regime. The wide raw oscillations are normal for GRPO.*

![Policy Optimization: GRPO Loss Convergence — Run 2 (90 steps). Wider variance in raw loss but EMA holds stable throughout.]

*Run 2, extended. Wider variance in the raw loss at higher step counts is expected. The EMA holds, which means the optimizer isn't diverging.*

### Training progression

![Training Progression: Epochs vs Steps — Run 1 (40 steps). Linear progression confirms clean training loop.]

*Epochs vs. Training Steps — Run 1. Linear progression with no anomalies. The training loop is running cleanly without restarts or crashes.*

![Training Progression: Epochs vs Steps — Run 2 (90 steps). Steeper slope reflects higher throughput on A10G compute.]

*Run 2, extended to 90 steps. The steeper slope reflects higher throughput on A10G GPUs.*

### Multi-run comparison

We ran multiple configurations to understand what was actually driving improvement.

![W&B overlay: 150 Epochs / Group Size 8 (blue) vs. 42 Epochs / Group Size 24 (pink). Larger group size shows tighter early-training variance.]

*W&B multi-run overlay comparing two key configurations: 150 Epochs with Group Size 8 (blue) vs. 42 Epochs with Group Size 24 (pink). Larger group size produces tighter reward variance in early training — because GR![W&B Multi-Run Comparison](wandb_overlay.png)

**The Key Finding:** **Group Size 24** produced significantly cleaner early gradients. More rollouts per group stabilized the advantage estimates, which was critical in the first 20 steps before a meaningful policy baseline existed.

---

## What the Agent Actually Learns

The agent developed four non-chip-specific emergent skills through RL pressure:

- **Long-Horizon Planning:** Managing a 12-step experiment budget; learning when to explore vs. when to commit.
- **Mid-Episode Self-Correction:** Explicitly reversing a hypothesis when new data contradicts it — a behavior almost entirely absent in untrained models.
- **Noise Filtering:** Distinguishing true physical signals from spurious correlations in a 15-parameter space.
- **Causal Reasoning:** Moving beyond statistical pattern-matching to understand the physical relationships between variables (e.g., how temperature affects yield via etch uniformity).

---

## Key Surprises

1. **The Consistency Gate:** We moved from simple keyword rewards to parsing full reasoning traces to ensure the agent's logic actually connected to its actions.
2. **Noise as a Feature:** Agents trained on "clean" environments failed in reality. Realistic lot-to-lot variance was mandatory for generalization.
3. **Reasoning Signal:** The **15% Causal Attribution reward** had a disproportionate impact, forcing the model to become conservative and evidence-based.

---

## Comparison: Untrained vs. Trained

| Feature | Untrained Model (Base) | RL-Trained Agent |
|---|---|---|
| **Hypothesis** | Picks early, ignores feedback | Collects evidence for 3-4 steps first |
| **Feedback Loop** | Doubles down on errors | Explicitly reverses if data contradicts |
| **Causal Accuracy** | Near-random guessing | High; matches true bottlenecks |
| **Outcome** | Shallow yield improvement | Deep optimization and target hit |

---

## Beyond the Fab: Universal Reasoning

The skills developed here—planning under budget, error correction, and noise filtering—generalize directly to:
- **Scientific Planning** & Research
- **Strategic Resource Allocation**
- **Complex System Debugging**

For India's Semiconductor Mission, we are training the core intelligence required to run modern, reliable industrial operations.

---

*Built at OpenEnv India 2026 by Team TensorTitans.*
[github.com/pranjalyt/fab-yield-agent](https://github.com/pranjalyt/fab-yield-agent)