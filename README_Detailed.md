# 🏭 Semiconductor Fab Yield Optimization Agent

> **Team:** TensorTitans | **Hackathon:** OpenEnv India 2026 | **Theme:** Long-Horizon Planning + World Modeling

A production-grade Reinforcement Learning environment where an LLM agent learns to behave as a **process integration engineer** inside a chip fabrication plant. The agent must run virtual wafer experiments, discover causal relationships between process parameters and yield, and converge on an optimal manufacturing recipe — all within a strict experiment budget — before submitting for senior engineer review.

---

## Table of Contents

- [What Problem Does This Solve?](#-what-problem-does-this-solve)
- [Real-World Analogy](#-real-world-analogy)
- [Repository Structure](#-repository-structure)
- [System Architecture](#-system-architecture)
- [Core Component Deep Dives](#-core-component-deep-dives)
  - [RSMSimulator — The Hidden Ground Truth](#1-rsmsimulator--the-hidden-ground-truth)
  - [FabYieldEnv — The RL Environment](#2-fabyieldenv--the-rl-environment)
  - [SeniorEngineerReviewer — The Multi-Agent Layer](#3-seniorengineerreviewer--the-multi-agent-layer)
  - [FastAPI Server — The Interface Layer](#4-fastapi-server--the-interface-layer)
  - [LLM Agent — The Decision Maker](#5-llm-agent--the-decision-maker)
- [The 15-Parameter Process Space](#-the-15-parameter-process-space)
- [Reward System — Four-Component Design](#-reward-system--four-component-design)
- [Curriculum Learning — Difficulty Progression](#-curriculum-learning--difficulty-progression)
- [Episode Lifecycle & Phase System](#-episode-lifecycle--phase-system)
- [Causal Discovery Mechanism](#-causal-discovery-mechanism)
- [Defect Classification System](#-defect-classification-system)
- [Training Pipeline — GRPO + Unsloth](#-training-pipeline--grpo--unsloth)
- [OpenEnv Specification (openenv.yaml)](#-openenv-specification-openenvyaml)
- [Dockerfile — Container Design](#-dockerfile--container-design)
- [Dependencies — What Each Library Does](#-dependencies--what-each-library-does)
- [API Reference](#-api-reference)
- [Running Locally](#-running-locally)
- [Design Decisions & Why](#-design-decisions--why)
- [Hackathon Bonus Features](#-hackathon-bonus-features)
- [Glossary](#-glossary)

---

## 🧩 What Problem Does This Solve?

Semiconductor manufacturing (chip fabrication) is one of the most complex industrial processes on Earth. A modern chip fab runs wafers through hundreds of process steps, each controlled by multiple physical parameters (temperatures, pressures, gas flows, spin speeds, etc.). Even tiny deviations in these parameters cascade into **yield loss** — meaning fewer working chips per wafer.

Currently, **process integration engineers** at companies like TSMC or Intel spend weeks running physical experiments, analyzing defect maps, forming hypotheses about root causes, and iteratively tuning process recipes. This is expensive, slow, and requires deep domain expertise.

This project trains an LLM agent to perform this exact job:

- Run virtual "experiments" by submitting parameter configurations
- Observe yield percentage and defect signatures as feedback
- Reason about which parameters are causing yield loss (causal attribution)
- Converge on the best process recipe within a limited budget
- Pass a senior engineer qualification review before a recipe goes to production

---

## 🏭 Real-World Analogy

Think of it this way: a **wafer** is like a pizza. You're trying to bake it perfectly — not too hot, not too fast, with the right toppings. The "recipe" is the combination of 15 oven settings. You can only try 12 batches. After each batch, you're told how many slices came out edible (yield) and whether the burns are on the edges (edge ring defect), center (center spot), or random. Your job is to deduce which oven settings caused the burns and find the perfect combination.

Now scale that to a billion-transistor chip with 15 interdependent physical parameters and you have this project.

---

## 📁 Repository Structure

```
fab-yield-agent/
│
├── rsm_simulator.py          # Hidden ground truth — Response Surface Model engine
│
├── environment/              # RL environment definitions
│   ├── fab_env.py            # FabYieldEnv — the Gym-like environment class
│   ├── observation.py        # FabObservation dataclass (what the agent sees)
│   └── reviewer.py           # SeniorEngineerReviewer — multi-agent qualification layer
│
├── agents/                   # LLM agent implementations
│   └── llm_agent.py          # Prompt construction, XML parsing, inference loop
│
├── server/                   # FastAPI web server (OpenEnv interface)
│   └── app.py                # /reset, /step, /health endpoints
│
├── scripts/                  # Utilities and testing
│   └── test_episode.py       # End-to-end episode smoke test (no GPU required)
│
├── openenv.yaml              # Formal environment specification for the hackathon
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container build instructions for HuggingFace Spaces
└── README.md                 # This file
```

---

## 🏗️ System Architecture

The system is organized into four cleanly separated layers that interact through well-defined interfaces:

```
┌──────────────────────────────────────────────────────────────────┐
│                         LLM Agent Layer                          │
│  • Reads FabObservation (text/structured)                        │
│  • Outputs XML: <params>, <bottleneck>, <reasoning>, <submit>    │
│  • Trained with GRPO on structured reasoning traces              │
└───────────────────────────┬──────────────────────────────────────┘
                            │  POST /step (JSON action)
┌───────────────────────────▼──────────────────────────────────────┐
│                      FastAPI Server Layer                        │
│  • /reset  → initializes new episode                             │
│  • /step   → validates action, routes to environment             │
│  • /       → health check for HuggingFace Spaces                 │
└───────────────────────────┬──────────────────────────────────────┘
                            │  Python function calls
┌───────────────────────────▼──────────────────────────────────────┐
│                    FabYieldEnv Layer                             │
│  • Manages episode state (step counter, phase, history)          │
│  • Routes experiments to RSMSimulator                            │
│  • Computes 4-component reward vector                            │
│  • Invokes SeniorEngineerReviewer on submission                  │
└────────────┬──────────────────────────┬──────────────────────────┘
             │                          │
┌────────────▼──────────┐  ┌────────────▼──────────────────────────┐
│    RSMSimulator       │  │      SeniorEngineerReviewer            │
│ (Hidden from agent)   │  │  (Episode-varying qualification rules) │
│                       │  │                                        │
│  • 2nd-order RSM      │  │  • Checks yield threshold              │
│  • Random coefficients│  │  • Checks stability (lot variance)     │
│  • Causal structure   │  │  • Checks forbidden parameter ranges   │
│  • Defect classifier  │  │  • Generates rejection feedback text   │
└───────────────────────┘  └────────────────────────────────────────┘
```

The key architectural decision is **information hiding**: the `RSMSimulator` is never exposed to the agent — just like real fab engineers cannot see the underlying physics. The agent only sees the *outputs* (yield, defect type) and must infer the *inputs* (which parameters matter and why).

---

## 🔬 Core Component Deep Dives

### 1. RSMSimulator — The Hidden Ground Truth

**File:** `rsm_simulator.py`

This is the most technically sophisticated component. It implements a **second-order Response Surface Model (RSM)** — a mathematical technique widely used in semiconductor manufacturing and industrial process optimization (Box-Behnken design, Central Composite Design).

#### What is a Response Surface Model?

An RSM captures how a measured output (here: wafer yield) responds to multiple input factors (process parameters). Instead of building a first-principles physical model (which would require solving differential equations of plasma physics, diffusion kinetics, etc.), an RSM fits a polynomial surface to observed data points.

The yield function implemented here is:

```
yield = base_yield
      + β · x                    (linear terms)
      + α · x²                   (quadratic/curvature terms)
      + Σ γᵢⱼ · xᵢ · xⱼ         (pairwise interaction terms)
      + ε                        (Gaussian process noise)
```

Where `x` is the **normalized** parameter vector in `[-1, 1]` space.

#### Why Normalize to [-1, 1]?

Each real-world parameter has wildly different units and scales: temperature is in °C (160–220), dopant concentration is in atoms/cm³ (~10¹⁵), spin speed is in RPM (1500–3500). Directly comparing these would make the coefficients meaningless. By mapping each parameter linearly to `[-1, 1]`, the beta coefficients become directly comparable — a large `|β|` truly means that parameter has a strong influence on yield, regardless of its original unit.

The normalization formula:
```python
x[i] = 2.0 * (params[p] - lo) / (hi - lo) - 1.0
```

This maps `lo → -1` and `hi → +1`, centering the design space at 0.

#### Why Second-Order (Quadratic) Terms?

Linear models assume yield monotonically improves or worsens as you move a parameter. Reality is more complex — there's an optimal temperature, and yield drops if you go too high or too low. The negative quadratic terms (`alpha = -|N(0.06, 0.02)|`) guarantee the response surface has a **maximum** (not a minimum or saddle), meaning a finite optimal recipe exists. This is physically realistic: every process parameter has a sweet spot.

#### Why Interaction Terms?

In real fabs, parameters don't act independently. Pressure and gas flow interact to control plasma density. Temperature and RF power together determine etch profile. The model captures this with cross-terms `γᵢⱼ · xᵢ · xⱼ`. Only 4–6 sparse interaction pairs are activated per episode (chosen randomly) to keep the problem tractable while requiring the agent to reason about dependencies, not just individual parameter effects.

#### Episode-Fresh Randomization

Every `reset()` creates a brand new `RSMSimulator` with freshly sampled coefficients:

```python
self.beta = rng.normal(0, 0.18, n)        # linear effects
self.alpha = -np.abs(rng.normal(0.06, 0.02, n))  # curvature (always negative)
```

This means the agent cannot memorize a fixed recipe across episodes. It must generalize — learning *how* to optimize, not *what* to set. The seed makes episodes reproducible for evaluation fairness.

#### Theoretical Optimum Computation

After sampling coefficients, the simulator analytically computes the optimal parameter values:

```python
opt_x = np.clip(-self.beta / (2 * self.alpha), -1.0, 1.0)
```

This is the standard unconstrained optimum of a quadratic: `x* = -b/(2a)` for `f(x) = ax² + bx + c`. The clipping handles cases where the unconstrained optimum falls outside the physical parameter range. These optimal values are then denormalized back to real units and stored as `optimum_params` — used by the environment to compute efficiency rewards.

#### Causal Structure

The primary bottleneck parameter is the one with the largest absolute linear coefficient:
```python
primary_idx = int(np.argmax(np.abs(self.beta)))
self.primary_param = self.active_params[primary_idx]
```

Secondary parameters are those with `|β| > 0.07` (above a significance threshold). This ground truth is returned by `causal_structure()` — never shown to the agent, but used to score the agent's causal attribution guess at each step.

#### Lot-to-Lot Variance Simulation

Real production doesn't run one wafer — it runs **lots** of 25 wafers, and yield must be consistent. The `lot_variance()` method simulates this by running 20 virtual experiments with the submitted recipe and computing the standard deviation of yields:

```python
def lot_variance(self, params: Dict, n_lots: int = 20) -> float:
    yields = [self.predict(params)[0] for _ in range(n_lots)]
    return float(np.std(yields))
```

Since `predict()` adds Gaussian noise `N(0, 0.015)` each call, variance comes from process noise. A recipe near the optimum (flat response surface) will have lower variance than one on a steep slope.

---

### 2. FabYieldEnv — The RL Environment

**File:** `environment/fab_env.py`

This is the Gym-like environment that manages episode state and connects all components. It follows the standard `reset() → step() → done` interface used in RL.

#### reset() Method

```python
def reset(self, seed=None, difficulty=3) -> FabObservation:
```

- Creates a fresh `RSMSimulator` with the given seed and difficulty
- Creates a fresh `SeniorEngineerReviewer` with episode-specific qualification constraints
- Resets step counter, history, phase tracker
- Returns initial observation with baseline metrics from a center-point experiment (all parameters at midrange)

Why center-point as the first observation? It gives the agent a meaningful starting yield to compare against. An agent starting from random would waste steps just establishing a baseline. Starting at center-point also mirrors real fab practice — engineers begin with a "nominal" recipe before perturbing.

#### step() Method

```python
def step(self, action: FabAction) -> Tuple[FabObservation, RewardDict, bool, InfoDict]:
```

The step function:

1. **Validates** the action (parameter values within allowed ranges)
2. **Routes** to `RSMSimulator.predict()` to get `(yield_pct, defect_type)`
3. **Computes** all four reward components
4. **Updates** experiment history (full list of all previous params, yields, defects, agent guesses)
5. **Advances** the phase tracker (exploration → hypothesis → exploitation → submission)
6. **Checks** termination conditions (budget exhausted or `submit=True`)
7. **Returns** the updated observation, reward breakdown, done flag, info dict

The full history is included in every observation, making this a **partial observability** problem resolved through memory — the agent has complete access to its own experimental record.

#### Phase Management

The environment tracks which of the four phases the agent is in based purely on step count:

```python
phases = {
    range(1, 5):  "exploration",
    range(5, 9):  "hypothesis",
    range(9, 12): "exploitation",
    12:           "submission"
}
```

Each phase has a different `phase_hint` in the observation — natural language guidance that nudges the agent's behavior appropriately. This is a form of **curriculum guidance within an episode**: the agent is softly told what strategy to adopt at each stage.

---

### 3. SeniorEngineerReviewer — The Multi-Agent Layer

**File:** `environment/reviewer.py`

This component simulates the human approval gate that exists in real fabs before any process change goes to production. It implements **episode-varying qualification constraints** — one of the hackathon's bonus challenge requirements.

#### How It Works

At episode initialization, the reviewer is configured with random qualification constraints sampled within predetermined ranges. These might include:

- **Minimum yield threshold**: e.g., "yield must exceed 89.3%"
- **Maximum allowed lot variance**: e.g., "std dev must be below 1.8%"
- **Forbidden parameter ranges**: e.g., "temperature must not exceed 210°C" (simulating equipment constraints)
- **Required parameter ranges**: e.g., "etch time must be between 45–70s" (simulating integration constraints from upstream process)

When the agent submits a recipe (`submit=True`), the environment calls:

```python
approved, feedback = reviewer.review(params, yield_pct, lot_variance)
```

If rejected, `approved=False` and `feedback` contains specific human-readable text explaining *why* the recipe failed. This feedback is added to the observation, allowing the agent to adapt and try again (if budget permits). If the agent has no steps remaining, the episode ends with a stability penalty.

#### Why Episode-Varying Constraints?

Static constraints would allow the agent to memorize "always keep temperature below X". Real fabs have dynamic constraints driven by customer specs, equipment maintenance windows, upstream/downstream process integration requirements, and regulatory guidelines. By randomizing constraints per episode, the agent must learn to **read and respond to constraints dynamically** — a genuinely harder and more useful skill.

This is the **Snorkel AI bonus** feature: programmatic constraint generation that creates diverse qualification scenarios without manual labeling.

---

### 4. FastAPI Server — The Interface Layer

**File:** `server/app.py`

The environment is exposed as an HTTP API using FastAPI, following the OpenEnv interface standard. This allows any agent — LLM-based or traditional RL — to interact with the environment over HTTP.

#### Why FastAPI?

FastAPI provides automatic OpenAPI documentation, Pydantic model validation, async support, and is production-ready. For a hackathon submission running on HuggingFace Spaces, it's the ideal choice: lightweight, well-documented, and natively compatible with `uvicorn`.

#### Endpoints

**`POST /reset`**
- Accepts: `{ "seed": int, "difficulty": int }`
- Creates a new episode, returns the initial `FabObservation` as JSON
- Both fields are optional with sensible defaults

**`POST /step`**
- Accepts the full action dict (params, primary_bottleneck, reasoning, submit)
- Validates parameter ranges against `PARAM_RANGES` before touching the simulator
- Returns: `{ observation, rewards, done, info }`

**`GET /`**
- Health check endpoint
- Returns `{ "status": "ok", "env": "semiconductor-fab-yield-agent" }`
- Used by HuggingFace Spaces to verify the container is alive

#### State Management

The server holds one active environment instance in memory. For the hackathon's single-agent use case, this is sufficient. A production deployment would use session tokens or Redis to support concurrent agent instances.

---

### 5. LLM Agent — The Decision Maker

**File:** `agents/llm_agent.py`

The agent is an LLM (fine-tuned language model) that receives structured observations and must output structured actions in a specific XML format.

#### Observation-to-Prompt Conversion

Each `FabObservation` is formatted into a detailed text prompt including:
- Current step, phase, and budget remaining
- Current best yield vs. target yield
- Phase-specific guidance (exploration/hypothesis/exploitation hints)
- Full experiment history table (parameters tried, yields achieved, defects observed, previous bottleneck guesses)
- Active parameter list with their valid ranges
- Reviewer feedback (if a recipe was rejected)

The prompt is carefully constructed to provide all information needed for reasoning without exposing the ground truth causal structure.

#### XML Action Format

The agent must output structured XML:

```xml
<action>
  <params>
    <temp>195.0</temp>
    <etch_time>62.0</etch_time>
    <pressure>2.1</pressure>
    <dopant>1.4e15</dopant>
    <spin_speed>2200</spin_speed>
  </params>
  <primary_bottleneck>temp</primary_bottleneck>
  <reasoning>Increasing temperature from 180 to 195 improved yield by 8pp, suggesting it's the primary bottleneck. Staying in that direction.</reasoning>
  <submit>false</submit>
</action>
```

#### Why XML Instead of JSON?

XML with named tags is more reliably parseable from LLM outputs than JSON. LLMs are prone to JSON formatting errors (missing quotes, trailing commas). Named XML tags also make the structure more human-readable during training, helping the model learn the correct format. The parser uses regex or `xml.etree` to extract each field, with fallback defaults if a field is malformed.

#### Causal Attribution Scoring

The `primary_bottleneck` field is parsed and compared against `RSMSimulator.causal_structure()` for reward:
- **Full credit (1.0)**: exact match with primary parameter
- **Partial credit (0.4)**: named parameter is in the secondary parameters list
- **No credit (0.0)**: wrong guess

This incentivizes the agent to develop genuine causal reasoning — not just optimizing yield, but understanding *why* yield is where it is.

---

## 📐 The 15-Parameter Process Space

Each parameter corresponds to a real semiconductor process step. They're organized in curriculum order (easy to hard):

| # | Parameter | Range | Unit | Physical Role |
|---|-----------|-------|------|---------------|
| 1 | `temp` | 160–220 | °C | Primary thermal budget; affects diffusion, reaction rates, crystal quality |
| 2 | `etch_time` | 30–90 | seconds | Duration of plasma etching; interacts strongly with pressure |
| 3 | `pressure` | 1.5–3.0 | torr | Plasma density and ion mean free path; affects etch uniformity |
| 4 | `dopant` | 8e14–2e15 | atoms/cm³ | Transistor threshold voltage; too low = leakage, too high = reduced mobility |
| 5 | `spin_speed` | 1500–3500 | RPM | Photoresist coating uniformity; affects critical dimension control |
| 6 | `anneal_time` | 30–120 | seconds | Duration of post-implant anneal; activates dopants |
| 7 | `anneal_temp` | 900–1100 | °C | Temperature of anneal; crystal damage recovery |
| 8 | `dep_rate` | 10–50 | nm/min | Thin film deposition rate; affects film stress and conformality |
| 9 | `rf_power` | 100–500 | watts | RF generator power for plasma; controls etch selectivity |
| 10 | `gas_ar` | 20–100 | sccm | Argon gas flow; dilutes reactive species, controls plasma |
| 11 | `gas_cf4` | 5–30 | sccm | CF₄ fluorocarbon flow; provides fluorine etchant for silicon |
| 12 | `wafer_rot` | 0–30 | RPM | Wafer rotation during processing; corrects radial non-uniformity |
| 13 | `chuck_temp` | 15–40 | °C | Electrostatic chuck temperature; controls wafer stress |
| 14 | `cmp_pressure` | 1–5 | psi | Chemical-Mechanical Planarization pressure; affects removal rate |
| 15 | `cmp_velocity` | 30–90 | RPM | CMP pad velocity; interacts with pressure for removal rate |

Each parameter's range is chosen to reflect realistic fab operating windows — not so wide that the search space is intractable, not so narrow that every recipe works.

---

## 💰 Reward System — Four-Component Design

The reward is deliberately multi-component because real engineering success is multi-dimensional. A recipe that maximizes yield but is wildly inconsistent lot-to-lot is useless in production. A recipe that blames the wrong parameter for yield loss is unscientific and cannot be transferred.

### Component 1: Yield Reward (50% weight) — Dense

```python
yield_reward = (current_yield - baseline_yield) / (target_yield - baseline_yield)
yield_reward = max(0.0, min(1.0, yield_reward))
```

This is a **dense** reward — given at every step regardless of whether the agent submits. It provides continuous gradient signal throughout the episode, preventing the sparse reward problem that makes RL hard. The normalization makes it scale-invariant across difficulties (easy target 80% vs. hard target 92%).

Why 50%? Yield is the primary objective. Everything else serves yield.

### Component 2: Efficiency Reward (20% weight) — Sparse

```python
if current_yield >= target_yield:
    efficiency_reward = budget_remaining / max_budget
else:
    efficiency_reward = 0.0
```

This reward is **sparse** — only granted when the target is hit, and its magnitude depends on how many experiments remain. Hitting 92% yield in 7 steps is worth more than hitting it in 11 steps. This encourages strategic, targeted exploration rather than brute-force grid search.

### Component 3: Causal Attribution Reward (15% weight) — Parsed from NL

```python
if agent_guess == ground_truth.primary_param:
    causal_reward = 1.0
elif agent_guess in ground_truth.secondary_params:
    causal_reward = 0.4
else:
    causal_reward = 0.0
```

This reward is computed by **parsing the agent's natural language reasoning**. It's a unique design choice: the environment reads the agent's stated hypothesis and scores it against the hidden causal structure. This creates a training signal that rewards *correct understanding* of the system, not just correct actions.

Why does this matter? An agent that hit the yield target by accident (random walk) should be rewarded less than one that correctly identified the bottleneck and targeted it. This reward discriminates between lucky and intelligent behavior.

### Component 4: Stability Reward (15% weight) — Terminal Only

```python
# Only computed on submission
lot_std = simulator.lot_variance(submitted_params, n_lots=20)
stability_reward = max(0.0, 1.0 - lot_std / 0.05)  # target: <5% std
if reviewer_rejected:
    stability_reward = 0.0
```

This reward is **terminal** — only computed when the agent submits a recipe. It measures manufacturing consistency. A standard deviation below 5% is considered acceptable. Reviewer rejection zeroes this reward, creating a hard penalty for ignoring qualification constraints.

### Total Reward

```python
total = 0.50 * yield_reward + 0.20 * efficiency_reward +
        0.15 * causal_reward + 0.15 * stability_reward
```

The weights were chosen to ensure that yield optimization is the primary objective while the secondary objectives meaningfully differentiate expert from novice behavior.

---

## 📈 Curriculum Learning — Difficulty Progression

One of the most important training techniques used here is **curriculum learning**: training on easy problems first, then progressively harder ones. This mirrors how human experts are taught — junior engineers handle simpler processes before being assigned to full 15-parameter integration problems.

| Phase | Difficulty | Active Parameters | Budget | Target Yield | Why |
|-------|------------|-------------------|--------|--------------|-----|
| 1 | Easy | 5 (first 5) | 8 steps | >80% | Establishes basic optimization skill |
| 2 | Medium | 10 (first 10) | 10 steps | >88% | Adds anneal, deposition, RF parameters |
| 3 | Hard | 15 (all) | 12 steps | >92% | Full industrial complexity |

The parameter ordering is intentional: the first 5 parameters (`temp`, `etch_time`, `pressure`, `dopant`, `spin_speed`) are the highest-impact parameters in a typical CMOS process. By starting with these, the easy curriculum builds foundational intuition before the agent must manage CMP, gas chemistry, and wafer rotation subtleties.

Training progression typically moves from Easy → Medium → Hard as the agent's average episode reward exceeds a threshold (e.g., >0.7 for 100 consecutive episodes on the current difficulty).

---

## 🔄 Episode Lifecycle & Phase System

Every episode follows a structured 12-step arc (at maximum difficulty):

```
Step 1-4: EXPLORATION PHASE
━━━━━━━━━━━━━━━━━━━━━━━━━
Goal: Map the parameter space broadly
Strategy: Vary parameters one at a time (one-factor-at-a-time)
          or use a fractional factorial design
Phase hint: "Explore broadly — try extreme values to map the landscape"
Expected: Identify which parameters have strong yield sensitivity

Step 5-8: HYPOTHESIS PHASE
━━━━━━━━━━━━━━━━━━━━━━━━━
Goal: Test specific causal theories
Strategy: Design experiments that confirm/disconfirm the hypothesized
          bottleneck (change one parameter, hold others constant)
Phase hint: "Focus on your leading hypothesis — test it deliberately"
Expected: Narrow down primary/secondary bottleneck parameters

Step 9-11: EXPLOITATION PHASE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Goal: Converge on the optimum
Strategy: Gradient following — small steps toward better yield
Phase hint: "You're close — fine-tune around your best recipe"
Expected: Yield approaching or exceeding target threshold

Step 12: SUBMISSION PHASE
━━━━━━━━━━━━━━━━━━━━━━━━
Goal: Submit best recipe for qualification
Strategy: Submit the highest-yield, most stable recipe from history
Phase hint: "Time to commit — submit your best recipe now"
Expected: Reviewer approval or rejection with feedback
```

This phase structure is inspired by real Design of Experiments (DoE) methodology used in industrial process development.

---

## 🧪 Causal Discovery Mechanism

The most intellectually challenging aspect of this environment is the **causal discovery task** — figuring out *why* yield is what it is, not just *that* it is.

### The Problem

The agent sees: "yield = 84.3%, defect = edge_ring"
The question is: was this because of pressure? gas_ar? their interaction? temperature?

### The Signal

The RSM structure ensures that the parameter with the largest |β| coefficient has the strongest causal influence on yield. But the agent doesn't see β — it only sees yield differences between experiments.

By running controlled experiments (change one parameter at a time while holding others fixed), the agent can estimate marginal yield effects for each parameter and identify the primary driver.

### The Incentive

The `causal_attribution` reward component (15% of total) is parsed from the agent's `primary_bottleneck` field at every step. This creates continuous pressure to develop and refine causal hypotheses throughout the episode, not just at submission time.

### Why This Matters for Training

Standard RL would optimize purely for yield — it doesn't care why yield is high. By adding a causal reward signal, we train agents that develop genuine understanding of the system. This generalizes better: an agent that understood causality in episode 1000 can apply that understanding in episode 1001 even with completely different RSM coefficients.

---

## 🔴 Defect Classification System

The simulator produces physically motivated defect signatures that serve as interpretable clues for the agent:

### `edge_ring`
**Physical cause**: Non-uniform plasma at wafer edges  
**RSM condition**: `|x_pressure| + |x_gas_ar| > 1.3`  
**Interpretation**: Pressure and argon flow are far from optimal, creating plasma density gradients that cause edge-localized etching non-uniformity. The agent should adjust pressure and gas_ar toward center values.

### `center_spot`
**Physical cause**: Thermal hotspot at wafer center  
**RSM condition**: `|x_temp| + |x_rf_power| > 1.2`  
**Interpretation**: Temperature and RF power are in extreme regions, causing localized thermal damage at the wafer center where heat accumulates. The agent should moderate temperature and RF power.

### `random_scatter`
**Physical cause**: Contamination from dopant off-spec  
**RSM condition**: `|x_dopant| > 0.85`  
**Interpretation**: Dopant concentration is far from optimal. Random scattering of defects (vs. patterned) indicates a chemical/ionic contamination mechanism rather than a physical uniformity problem. The agent should bring dopant closer to the target range.

### `none`
**Condition**: `yield > 88%`  
All process parameters are near-optimal; defect density is below detection threshold.

The defect classifier creates a **multi-modal signal**: the agent doesn't just see yield numbers, it sees interpretable defect patterns that directly point toward parameter adjustments. This is exactly the information a real process engineer uses — wafer defect maps are the primary diagnostic tool in fab process development.

---

## 🤖 Training Pipeline — GRPO + Unsloth

**Reference:** `training/colab_training.ipynb`

The LLM agent is trained using **GRPO (Group Relative Policy Optimization)** — a variant of PPO designed specifically for LLM fine-tuning that avoids the need for a separate value/critic network.

### Why GRPO over PPO?

Standard PPO requires a critic network to estimate value functions, doubling memory requirements. For LLMs, this is prohibitive. GRPO instead uses within-group baseline estimation: for each prompt, it samples multiple completions and uses their average reward as the baseline, estimating advantage relative to that group. This is both memory-efficient and empirically effective for language model RL.

### Why Unsloth?

Unsloth is a library that implements optimized CUDA kernels for LLM training, achieving 2–4x training speedup with 60% less VRAM compared to standard HuggingFace Trainer. For a hackathon running on a free Colab instance, this is essential — it makes the difference between training completing in hours vs. days.

### Training Stack

```python
from trl import GRPOTrainer       # GRPO implementation
from peft import LoraConfig        # Parameter-efficient fine-tuning
from unsloth import FastLanguageModel  # Memory-optimized loading
```

**LoRA (Low-Rank Adaptation)**: Instead of fine-tuning all model weights (billions of parameters), LoRA injects small trainable rank decomposition matrices into attention layers. Only ~0.1–1% of parameters are updated during training, making it feasible on consumer hardware.

### Curriculum Training Schedule

```
Phase 1 (Easy): Train until avg_reward > 0.70 over 100 episodes
     ↓
Phase 2 (Medium): Train until avg_reward > 0.65 over 100 episodes
     ↓
Phase 3 (Hard): Train until avg_reward > 0.60 over 100 episodes
```

### Reward Shaping for GRPO

The four-component reward vector is reduced to a scalar for GRPO training. The weighting (50/20/15/15) was tuned through ablation experiments to balance exploration incentives with causal reasoning development.

### WandB Integration

The `wandb` library tracks all training metrics: reward breakdown per component, yield trajectory within episodes, causal attribution accuracy over training, and curriculum advancement events.

---

## 📄 OpenEnv Specification (openenv.yaml)

The `openenv.yaml` file is the formal contract between this environment and the OpenEnv hackathon platform. It defines:

```yaml
name: semiconductor-fab-yield-agent
version: 1.0.0
theme: long_horizon_planning
```

**Tasks defined:**
- `easy_yield_optimization`: 5 params, budget 8, target 80%
- `medium_yield_optimization`: 10 params, budget 10, target 88%
- `hard_yield_optimization`: 15 params, budget 12, target 92%

**Action space schema:** Dict with `params`, `primary_bottleneck`, `reasoning`, `submit`

**Observation space schema:** Dict with step, phase, history, hints, reviewer_feedback

**Reward range:** [0.0, 1.0] normalized

**Multi-agent declaration:** `senior_engineer_reviewer` is declared as a `rule_based` agent, satisfying the hackathon's multi-agent requirement.

The YAML file serves as machine-readable documentation that the OpenEnv platform uses to automatically generate evaluation harnesses, leaderboards, and baseline comparisons.

---

## 🐳 Dockerfile — Container Design

The Dockerfile follows an eight-step pattern optimized for the HuggingFace Spaces deployment platform:

```dockerfile
FROM python:3.11-slim         # Step 1: Minimal Python base image
USER root                      # Step 2: Root for system installs
RUN apt-get install git build-essential  # Step 3: Compiler toolchain
RUN useradd -m -u 1000 user   # Step 4: Create HF-required user (UID 1000)
WORKDIR $HOME/app              # Step 5: Set working directory
COPY requirements.txt . && pip install  # Step 6: Install Python deps
COPY --chown=user . .          # Step 7: Copy application code
USER user                      # Step 8: Switch to non-root user
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Key Design Decisions

**Why `python:3.11-slim`?** The slim variant excludes development tools and documentation, reducing image size from ~900MB to ~130MB. This speeds up Spaces deployments and reduces cold start time.

**Why install `build-essential`?** Several dependencies (`unsloth`, `triton`, `bitsandbytes`) include C/CUDA extensions that must be compiled at install time on the target platform. Without `gcc` and `make`, `pip install` fails for these packages.

**Why UID 1000?** HuggingFace Spaces requires the application to run as a non-root user with UID 1000. The `useradd -u 1000` explicitly sets this UID, and the `USER user` switch at the end satisfies HF's security requirement.

**Why install before copying code?** Docker layer caching means unchanged layers are reused. If requirements.txt doesn't change, the pip install layer is cached and subsequent builds skip it — dramatically speeding up development iteration.

**Port 7860** is the standard HuggingFace Spaces port for Gradio and FastAPI applications.

---

## 📦 Dependencies — What Each Library Does

| Library | Version | Role |
|---------|---------|------|
| `openenv-core` | latest | OpenEnv hackathon framework — defines environment base classes and scoring |
| `pydantic>=2.0` | ≥2.0 | Data validation for API request/response models; Pydantic v2 is 5-10x faster than v1 |
| `fastapi` | latest | ASGI web framework for the `/reset` and `/step` API endpoints |
| `uvicorn` | latest | ASGI server that runs FastAPI; production-ready, async |
| `numpy` | latest | Core numerical library: array operations for RSM computation, normalization, linear algebra |
| `requests` | latest | HTTP client for agent-to-server communication in test scripts |
| `python-dotenv` | latest | Loads environment variables from `.env` files (API keys, WandB tokens) |
| `scipy` | latest | Scientific computing: used for optimization routines and statistical analysis |
| `wandb` | latest | Weights & Biases — experiment tracking, metric logging, hyperparameter sweeps |
| `peft` | latest | Parameter-Efficient Fine-Tuning — implements LoRA adapters for LLM training |
| `accelerate` | latest | HuggingFace Accelerate — handles distributed training, mixed precision |
| `bitsandbytes` | latest | 4-bit/8-bit quantization for memory-efficient model loading |
| `trl<0.9.0` | <0.9.0 | Transformer Reinforcement Learning — GRPO, PPO trainers for LLMs |
| `unsloth` | latest | 2-4x faster LLM training with 60% VRAM reduction via optimized CUDA kernels |

The `trl<0.9.0` pin is important — TRL 0.9.0+ changed the GRPO API significantly, and the training notebook was validated against the older API.

---

## 🌐 API Reference

### POST /reset

Initialize a new episode.

**Request:**
```json
{
  "seed": 42,        // Optional. Integer seed for reproducibility
  "difficulty": 1    // Optional. 1=easy, 2=medium, 3=hard (default: 3)
}
```

**Response:** `FabObservation`
```json
{
  "step": 0,
  "phase": "exploration",
  "budget_remaining": 12,
  "target_yield": 92.0,
  "current_best_yield": 78.4,
  "active_params": ["temp", "etch_time", "pressure", "dopant", "spin_speed", "..."],
  "param_ranges": {
    "temp": [160.0, 220.0],
    "etch_time": [30.0, 90.0],
    "..."
  },
  "experiment_history": [],
  "phase_hint": "Explore broadly — try extreme values to map the yield landscape.",
  "reviewer_feedback": null
}
```

### POST /step

Submit an action and receive the next observation.

**Request:**
```json
{
  "params": {
    "temp": 200.0,
    "etch_time": 55.0,
    "pressure": 2.1,
    "dopant": 1.3e15,
    "spin_speed": 2500
  },
  "primary_bottleneck": "temp",
  "reasoning": "Increasing temperature from 180 to 200 improved yield by 8pp. Continuing in this direction.",
  "submit": false
}
```

**Response:**
```json
{
  "observation": { "...FabObservation..." },
  "rewards": {
    "yield_reward": 0.74,
    "efficiency_reward": 0.0,
    "causal_attribution": 1.0,
    "stability_reward": 0.0,
    "total": 0.52
  },
  "done": false,
  "info": {
    "yield_pct": 88.7,
    "defect": "none",
    "step": 5,
    "phase": "hypothesis"
  }
}
```

### GET /

Health check.

**Response:**
```json
{ "status": "ok", "env": "semiconductor-fab-yield-agent" }
```

---

## 🚀 Running Locally

### Prerequisites

- Python 3.11+
- pip
- No GPU required for running the environment (only needed for LLM training)

### Setup

```bash
# Clone the repository
git clone https://github.com/pranjalyt/fab-yield-agent.git
cd fab-yield-agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation (No GPU Needed)

```bash
python scripts/test_episode.py
```

This runs a complete 12-step episode with a simple scripted agent (not an LLM), verifying:
- RSMSimulator initializes correctly
- FabYieldEnv reset/step cycle works
- SeniorEngineerReviewer activates on submission
- All reward components compute correctly
- API server can be started

Expected output:
```
Episode started. Difficulty=3, Seed=42
Step 1 | Phase: exploration | Yield: 82.1% | Defect: edge_ring | Reward: 0.31
Step 2 | Phase: exploration | Yield: 85.3% | Defect: none      | Reward: 0.48
...
Step 12 | Phase: submission | Submitted recipe. Yield: 91.7%
Reviewer: APPROVED ✓
Final reward: 0.71
```

### Start the API Server

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

The server will be available at `http://localhost:7860`. Test it:

```bash
# Health check
curl http://localhost:7860/

# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "difficulty": 1}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "params": {"temp": 200.0, "etch_time": 55.0, "pressure": 2.1, "dopant": 1.3e15, "spin_speed": 2500},
    "primary_bottleneck": "temp",
    "reasoning": "Testing temperature sensitivity",
    "submit": false
  }'
```

### LLM Training (Requires GPU)

```bash
# Open the Colab notebook
# training/colab_training.ipynb

# Or run locally with GPU
python training/train.py --difficulty 1 --model_name mistralai/Mistral-7B-v0.1
```

---

## 🎯 Design Decisions & Why

### Decision 1: RSM over Physics Simulation

We chose a statistical RSM rather than a first-principles physics simulator (e.g., TCAD). A full plasma physics + diffusion + CMP simulator would take hours per experiment and require solving coupled PDEs. An RSM runs in microseconds. For RL training that requires millions of environment steps, computational efficiency is non-negotiable. The RSM still captures the essential challenge — non-linear multi-parameter optimization with hidden causal structure — without the prohibitive compute cost.

### Decision 2: Dense + Sparse Reward Combination

Using only dense reward (yield at every step) encourages greedy local search. Using only sparse reward (only at submission) makes training extremely slow due to credit assignment difficulty. The 50% dense yield reward gives consistent gradient, while the 20% sparse efficiency reward encourages strategic planning and budget management. This combination is inspired by the reward shaping literature and empirically outperforms either approach alone.

### Decision 3: 12-Experiment Budget

12 is carefully chosen. In real DoE methodology, a full 2⁵ factorial design for 5 parameters requires 32 runs. A half-fraction requires 16. Central Composite Design for 5 parameters requires 26. Our budget of 12 (for 15 parameters) forces the agent to be dramatically more sample-efficient than textbook DoE — it must learn to prioritize high-information experiments, which is exactly the skill that makes expert engineers valuable.

### Decision 4: Phase Hints Instead of Phase Constraints

We provide phase hints (natural language guidance) but don't hard-constrain the agent's strategy. A hard constraint ("you MUST explore randomly in steps 1-4") would make the environment easier but would also short-circuit interesting emergent behaviors. Some episodes might benefit from going straight to exploitation if the starting yield is already near-target. Soft hints respect agent autonomy while providing helpful guidance.

### Decision 5: Pydantic v2 for API Validation

Pydantic v2 is ~10x faster at validation than v1. For an environment that needs to run thousands of steps per second during training, this matters. The API validates every parameter value against its allowed range, providing clear error messages when the agent proposes out-of-range values — better training signal than silent clipping.

---

## 🏆 Hackathon Bonus Features

This submission addresses multiple OpenEnv India 2026 bonus categories:

### Long-Horizon Planning (Theme #2)
The 12-step episode with distinct phases explicitly requires planning across a long horizon. The agent must balance exploration (gathering information) against exploitation (converging on the optimum) across a structured timeline.

### World Modeling (Theme #3)
The agent implicitly builds a world model — it must infer the hidden RSM structure from observations. Correctly identifying the primary bottleneck (causal attribution reward) requires maintaining and updating beliefs about the underlying process dynamics.

### Snorkel AI Bonus — Programmatic Constraint Generation
Episode-varying qualification constraints are generated programmatically at runtime, creating diverse evaluation scenarios without manual labeling. Each episode presents a unique set of reviewer constraints drawn from configurable distributions.

### Fleet AI Bonus — Rejection Feedback Loop
When the reviewer rejects a submitted recipe, structured feedback is incorporated into the observation, and the agent has the opportunity to re-optimize and resubmit (if budget permits). This feedback loop mirrors real manufacturing workflows and tests the agent's ability to adapt to constraint feedback.

---

## 📖 Glossary

| Term | Definition |
|------|-----------|
| **Wafer** | A thin silicon disc (200–300mm diameter) on which chips are fabricated simultaneously |
| **Yield** | Percentage of working chips per wafer; 92% means 92% of the die are functional |
| **RSM** | Response Surface Methodology — statistical technique mapping inputs to outputs with a polynomial model |
| **DoE** | Design of Experiments — systematic approach to planning experiments for maximum information |
| **Lot** | A batch of typically 25 wafers processed together; lot-to-lot variance measures consistency |
| **Defect** | Physical damage or contamination on the wafer surface causing chip failure |
| **Process Recipe** | The complete set of parameter values that define a manufacturing process |
| **Bottleneck Parameter** | The single process parameter with the greatest influence on yield |
| **Causal Attribution** | Identifying which parameter(s) caused the observed yield behavior |
| **GRPO** | Group Relative Policy Optimization — RL algorithm for LLM training |
| **LoRA** | Low-Rank Adaptation — memory-efficient LLM fine-tuning technique |
| **TSMC** | Taiwan Semiconductor Manufacturing Company — world's largest chip contract manufacturer |
| **CMOS** | Complementary Metal-Oxide-Semiconductor — dominant transistor technology |
| **CMP** | Chemical-Mechanical Planarization — polishing process to flatten chip surfaces |
| **sccm** | Standard cubic centimeters per minute — unit for gas flow rates |
| **torr** | Unit of pressure; 1 torr = 1/760 atmospheres (used in vacuum chambers) |
| **Dense Reward** | Reward given at every timestep |
| **Sparse Reward** | Reward given only at specific events (e.g., episode termination) |

---

## 📄 License

This project was created for OpenEnv India 2026 by Team TensorTitans. All code is open source.

---
