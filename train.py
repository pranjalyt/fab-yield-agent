"""
train.py — GRPO Training Brain for fab-yield-agent
====================================================
Migrated from Colab. Run this locally or on your HF Space alongside server.py.

Setup:
    cp .env.example .env          # fill in your HF_TOKEN
    python train.py

Requirements:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
    pip install python-dotenv
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import os
import re
import sys
import torch
import requests
from torch.optim import AdamW
from unsloth import FastLanguageModel

# Load .env file if present (safe no-op if file is missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # dotenv optional — env vars can also be set by the shell / HF Spaces secrets UI


# ─── CONFIG ─────────────────────────────────────────────────────────────────
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
SPACE_URL = os.environ.get("SPACE_URL", "http://localhost:7860")

if not HF_TOKEN:
    sys.exit(
        "\n❌  HF_TOKEN is not set.\n"
        "    Option A — create a .env file (see .env.example):\n"
        "               cp .env.example .env  →  fill in HF_TOKEN\n"
        "    Option B — export it in your shell:\n"
        "               export HF_TOKEN=hf_...\n"
        "    Option C — add it as a Secret in your HF Space UI.\n"
    )

API_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

MAX_SEQ_LENGTH = 2048

# GRPO hyperparams — override any of these via .env or shell exports
EPOCHS     = int(os.environ.get("EPOCHS",        5))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE",    4))
LR         = float(os.environ.get("LEARNING_RATE", 5e-6))


# ─── MODEL SETUP ────────────────────────────────────────────────────────────
def load_model():
    """Load Qwen2.5-7B with 4-bit quantisation and LoRA adapters."""
    print("🧠 Loading Qwen2.5-7B-Instruct (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("✅ Model loaded and LoRA adapters attached.")
    return model, tokenizer


# ─── PARSER ─────────────────────────────────────────────────────────────────
def parse_action(text: str, active_params: list = None) -> dict:
    """Parse the model's XML-structured output into a plain dictionary."""
    action = {
        "params": {},
        "primary_bottleneck": "",
        "reasoning": "",
        "think": "",
        "submit": False,
    }

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        action["think"] = think_match.group(1).strip()

    exp_match = re.search(r"<experiment>(.*?)</experiment>", text, re.DOTALL | re.IGNORECASE)
    if exp_match:
        for line in exp_match.group(1).strip().split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                key, val = key.strip().lower(), val.strip()
                if key == "submit":
                    action["submit"] = val.lower() in ("true", "yes", "1")
                else:
                    try:
                        action["params"][key] = float(val)
                    except ValueError:
                        pass

    diag_match = re.search(r"<diagnosis>(.*?)</diagnosis>", text, re.DOTALL | re.IGNORECASE)
    if diag_match:
        for line in diag_match.group(1).strip().split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                key, val = key.strip().lower(), val.strip()
                if "bottleneck" in key:
                    action["primary_bottleneck"] = val
                elif "reason" in key:
                    action["reasoning"] = val

    return action


# ─── PROMPT BUILDER ─────────────────────────────────────────────────────────
def _fmt(v):
    """Format a float value cleanly for prompt display."""
    if v == 0:
        return "0"
    if abs(v) >= 1e13 or (abs(v) < 0.01 and v != 0):
        return f"{v:.2e}"
    if abs(v) >= 1000:
        return f"{v:.0f}"
    return f"{v:.2f}"


def build_prompt(obs: dict) -> str:
    """Translate the API observation dict into a Qwen chat prompt."""
    step          = obs.get("step", 1)
    budget        = obs.get("budget_remaining", 12)
    best          = obs.get("current_best_yield", 0.0)
    target        = obs.get("target_yield", 92.0)
    phase         = obs.get("phase", "exploration")
    phase_hint    = obs.get("phase_hint", "")
    task_target   = obs.get("task_target", "Yield Optimization")
    reviewer_fb   = obs.get("reviewer_feedback", "")
    fin_fb        = obs.get("financial_feedback", "")
    history       = obs.get("experiment_history", [])
    active_params = obs.get("active_params", [])
    param_ranges  = obs.get("param_ranges", {})

    prompt  = "You are a Senior Process Engineer.\n\n"
    prompt += f"═══ OBJECTIVE ═══\n  Task: {task_target}\n\n"
    prompt += (
        f"═══ CURRENT STATE ═══\n"
        f"  Experiments run: {step - 1}/{step - 1 + budget}  "
        f"Budget: {budget}  "
        f"Best yield: {best:.1f}%  Target: >{target}%\n"
        f"  Phase: {phase.upper()} — {phase_hint}\n"
    )

    if fin_fb:
        prompt += f"\n⚠ FINANCIAL CONTROLLER:\n  {fin_fb}\n"
    if reviewer_fb:
        prompt += f"\n⚠ SENIOR REVIEWER:\n  {reviewer_fb}\n"

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

    prompt += "\n═══ YOUR ACTION ═══\n"
    prompt += "You MUST think step-by-step before acting. Analyze the history and constraints.\n"
    prompt += "Output your response strictly following this XML format:\n"
    prompt += "<think>\n  [Your reasoning]\n</think>\n"
    prompt += "<diagnosis>\n  primary_bottleneck: [param]\n</diagnosis>\n"
    prompt += "<experiment>\n  submit: false\n  temp: [value]\n  etch_time: [value]\n  # ... all other params\n</experiment>\n\n"
    prompt += "BEGIN YOUR RESPONSE NOW:\n<think>\n"

    return prompt


# ─── EPISODE RUNNER ─────────────────────────────────────────────────────────
def run_episode(model, tokenizer, difficulty: int = 1) -> tuple[list, float]:
    """Play one full episode. Returns (trajectory, total_reward)."""
    print(f"\n🚀 Starting Episode (Difficulty {difficulty})...")

    obs = requests.post(
        f"{SPACE_URL}/reset",
        json={"difficulty": difficulty},
        headers=API_HEADERS,
    ).json()

    trajectory   = []
    total_reward = 0.0
    done         = False

    while not done:
        step   = obs.get("step", 1)
        prompt = build_prompt(obs)

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True,
        )
        response_text = "<think>\n" + response_text   # re-attach force-fed opening tag

        active_params = obs.get("active_params", [])
        action_dict   = parse_action(response_text, active_params)

        print(f"\n[{'=' * 55}]")
        print(f"🔄 STEP {step} | Target: {obs.get('task_target', 'Yield Optimization')}")
        think_text = action_dict.get("think", "")
        print(f"🧠 Thinking:\n> {think_text}" if think_text else "🧠 Thinking: [none detected]")
        print(
            f"\n🛠️  Bottleneck: {action_dict.get('primary_bottleneck')} "
            f"| Submit: {action_dict.get('submit')}"
        )

        res = requests.post(
            f"{SPACE_URL}/step",
            json=action_dict,
            headers=API_HEADERS,
        )
        if res.status_code != 200:
            print(f"❌ API Error {res.status_code}: {res.text}")
            break

        step_data = res.json()
        obs        = step_data["observation"]
        reward     = step_data["rewards"]["total"]
        done       = step_data["done"]

        print(f"\n📈 Yield: {obs.get('current_best_yield', 0)}% | Reward: +{reward:.3f}")
        if obs.get("financial_feedback"):
            print(f"💰 {obs['financial_feedback']}")
        if obs.get("reviewer_feedback"):
            print(f"👨‍💻 {obs['reviewer_feedback']}")
        print(f"[{'=' * 55}]\n")

        trajectory.append({"prompt": prompt, "response": response_text, "reward": reward})
        total_reward += reward

    print(
        f"🏁 Episode done! "
        f"Yield: {obs.get('current_best_yield', 0)}% | "
        f"Total reward: {total_reward:.3f}"
    )
    return trajectory, total_reward


# ─── GRPO TRAINING LOOP ─────────────────────────────────────────────────────
def train(model, tokenizer):
    """Micro-batched GRPO: rollouts → advantage normalisation → gradient update."""
    optimizer = AdamW(model.parameters(), lr=LR)
    print(f"⚙️  GRPO ready | epochs={EPOCHS}  group_size={GROUP_SIZE}  lr={LR}\n")

    for epoch in range(EPOCHS):
        print(f"\n{'=' * 50}")
        print(f"📈 EPOCH {epoch + 1}/{EPOCHS}")

        group_trajectories, group_rewards = [], []

        # ── Phase 1: Rollouts (inference mode = low VRAM) ──
        FastLanguageModel.for_inference(model)
        for g in range(GROUP_SIZE):
            print(f"\n--- Rollout {g + 1}/{GROUP_SIZE} ---")
            traj, ep_reward = run_episode(model, tokenizer, difficulty=1)
            if traj:
                group_trajectories.append(traj)
                group_rewards.append(ep_reward)

        if len(group_trajectories) < 2:
            print("⚠️  < 2 successful rollouts — skipping update.")
            continue

        # ── Phase 2: Advantage normalisation ──
        rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
        mean_reward    = rewards_tensor.mean()
        std_reward     = rewards_tensor.std() + 1e-8
        advantages     = (rewards_tensor - mean_reward) / std_reward
        print(f"\n📊 Mean: {mean_reward.item():.3f} | Std: {std_reward.item():.3f}")

        # ── Phase 3: Gradient update (training mode) ──
        FastLanguageModel.for_training(model)
        optimizer.zero_grad()
        total_loss, total_steps = 0.0, 0

        for i, traj in enumerate(group_trajectories):
            adv = advantages[i].item()
            for step_data in traj:
                prompt   = step_data["prompt"]
                response = step_data["response"]

                inputs        = tokenizer(prompt + response, return_tensors="pt").to("cuda")
                prompt_length = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

                logits       = model(**inputs).logits
                shift_logits = logits[0, prompt_length - 1:-1, :]
                shift_labels = inputs.input_ids[0, prompt_length:]

                log_probs        = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                action_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

                total_loss  -= action_log_probs.sum() * adv
                total_steps += 1

        avg_loss = total_loss / total_steps
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"✅ Update done | GRPO Loss: {avg_loss.item():.4f}")

    print("\n🎉 Training complete!")


# ─── ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"Env server URL : {SPACE_URL}\n")

    model, tokenizer = load_model()

    # Smoke-test — one episode before committing to full training
    print("\n🧪 Smoke-test episode...")
    traj, score = run_episode(model, tokenizer, difficulty=1)
    print(f"Smoke-test score: {score:.3f}\n")

    train(model, tokenizer)