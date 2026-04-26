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

## 🔗 Quick Links

| Resource | Link |
|---|---|
| 🤗 HF Space (Environment API) | [semiconductor-yield-agent](https://huggingface.co/spaces/PranjalZetsu/semiconductor-yield-agent) |
| 🎮 Interactive UI | [semiconductor-yield-agent-ui](https://huggingface.co/spaces/PranjalZetsu/semiconductor-yield-agent-ui) |
| 🌐 Frontend Dashboard | [fab-ui-phi.vercel.app](https://fab-ui-phi.vercel.app) |
| 🧠 Trained Model | [Fab_Yield_Agent_Qwen-q4](https://huggingface.co/PranjalZetsu/Fab_Yield_Agent_Qwen-q4) |
| 📓 Training Notebook | [fab_yield_agent_training.ipynb](https://huggingface.co/spaces/PranjalZetsu/semiconductor-yield-agent/blob/main/fab_yield_agent_training.ipynb) |
| 📊 W&B Training Run | [tensor-titans-fab on W&B](https://wandb.ai/gamersdelightxd-/tensor-titans-fab?nw=nwuserpranjal_dubey) |
| 📝 Blog Post | [Medium — fab-yield-agent](https://medium.com/p/9d5e78528f95) |

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

## 📊 Training Results

Trained using **GRPO** (Group Relative Policy Optimization) via Unsloth + HuggingFace TRL on Qwen2.5-7B-Instruct (4-bit quantized).

**Key finding:** G=24 group size with 42 epochs outperformed G=8 with 150 epochs — larger groups produce cleaner advantage estimates exactly when the agent has no baseline yet.

| Run | Epochs | Group Size | Result |
|---|---|---|---|
| Run 1 | 150 | G=8 | Noisy gradients, unstable early training |
| Run 2 | 42 | G=24 | Tighter reward variance, better generalization ✅ |

📊 **[Full W&B training curves → tensor-titans-fab](https://wandb.ai/gamersdelightxd-/tensor-titans-fab?nw=nwuserpranjal_dubey)**

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

| 🎥 Demo Video | [YouTube](https://youtu.be/GXZUcbty9gg) |


## 📓 Training

Full GRPO training loop in [`fab_yield_agent_training.ipynb`](https://huggingface.co/spaces/PranjalZetsu/semiconductor-yield-agent/blob/main/fab_yield_agent_training.ipynb) — runnable on Google Colab (A100).

Experiment tracking: **Weights & Biases** → [tensor-titans-fab](https://wandb.ai/gamersdelightxd-/tensor-titans-fab?nw=nwuserpranjal_dubey)

Curriculum: Easy (5 params, >80%) → Medium (10 params, >88%) → Hard (15 params, >92%)