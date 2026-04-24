---
title: Semiconductor Fab Yield Agent
emoji: 🏭
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🏭 Semiconductor Fab Yield Optimization — RL Environment

An RL environment where an LLM agent learns to be a **process integration engineer** at a chip fab.

**Team:** TensorTitans | **Hackathon:** OpenEnv India 2026 | **Theme:** Long-Horizon Planning + World Modeling

---

## 🧠 What is this?

A fab produces chips. Each wafer has a **yield** (% of working chips). Yield is controlled by ~15 process parameters (temperature, pressure, etch time, dopant concentration, etc.). The agent must:

1. Run virtual experiments — tweaking parameters, observing yield + defect patterns
2. Form **causal hypotheses**: "did yield drop because of temperature, or because temperature affected etch uniformity?"
3. Converge on the optimal process recipe within a 12-experiment budget
4. Submit the recipe for **Senior Engineer review** (episode-varying qualification constraints)

This is the actual job of a process integration engineer at TSMC/Intel.

---

## 🏗️ Architecture

```
RSMSimulator (hidden)          FabYieldEnv                 LLM Agent
─────────────────────          ───────────────             ──────────
15-param response surface  →   reset() → observation   →   prompt
Hidden causal structure    ←   step(action) ← params    ←   XML output
lot-to-lot variance        →   reward_dict (4 components)
                               SeniorEngineerReviewer
                               (episode-varying constraints)
```

### Episode Phases
| Phase | Steps | Agent Goal |
|---|---|---|
| Exploration | 1–4 | Map the space broadly |
| Hypothesis | 5–8 | Test specific bottleneck theory |
| Exploitation | 9–11 | Converge on optimum |
| Submission | 12 | Submit final recipe |

### Reward Components
| Component | Weight | Signal |
|---|---|---|
| Yield | 50% | Dense — every step |
| Efficiency | 20% | Sparse — only if target hit |
| Causal Attribution | 15% | Parsed from agent's diagnosis |
| Stability | 15% | Terminal — lot-to-lot variance |

---

## 🚀 API

### `POST /reset`
```json
{ "seed": 42, "difficulty": 1 }
```
Returns initial `FabObservation`.

### `POST /step`
```json
{
  "params": { "temp": 200.0, "etch_time": 55.0, "pressure": 2.1, "dopant": 1.3e15, "spin_speed": 2500 },
  "primary_bottleneck": "temp",
  "reasoning": "Increasing temperature from 180 to 200 improved yield by 8pp.",
  "submit": false
}
```
Returns `{ observation, rewards, done, info }`.

### `GET /`
Health check.

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
python scripts/test_episode.py   # verify everything works (no GPU needed)
uvicorn server:app --host 0.0.0.0 --port 7860
```

---

## 📊 Training

See `training/colab_training.ipynb` for the full GRPO training loop using TRL + Unsloth.

Curriculum: Easy (5 params, >80%) → Medium (10 params, >88%) → Hard (15 params, >92%)