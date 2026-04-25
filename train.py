"""
train.py — GRPO Training Brain for fab-yield-agent
====================================================
Migrated from Colab. Run this locally or on your HF Space alongside server.py.

Architecture:
  - Custom PyTorch GRPO Loop (Policy Ratio + Clipping)
  - Gradient Accumulation (VRAM Efficient)
  - Step-Level Advantage Normalization
  - Dynamic Curriculum Difficulty (Easy -> Med -> Hard)
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
import os
import re
import wandb
import sys
import torch
import requests
from torch.optim import AdamW
from unsloth import FastLanguageModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─── CONFIG ─────────────────────────────────────────────────────────────────
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
SPACE_URL = os.environ.get("SPACE_URL", "http://localhost:7860")

if not HF_TOKEN:
    sys.exit("\n❌  HF_TOKEN is not set.")

API_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

MAX_SEQ_LENGTH = 2048
EPOCHS     = int(os.environ.get("EPOCHS",        150))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE",    8))
LR         = float(os.environ.get("LEARNING_RATE", 5e-6))


# ─── MODEL SETUP ────────────────────────────────────────────────────────────
def load_model():
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer


# ─── PARSER ─────────────────────────────────────────────────────────────────
def parse_action(text: str, active_params: list = None) -> dict:
    action = {"params": {}, "primary_bottleneck": "", "reasoning": "", "think": "", "submit": False}
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match: action["think"] = think_match.group(1).strip()

    exp_match = re.search(r"<experiment>(.*?)</experiment>", text, re.DOTALL | re.IGNORECASE)
    if exp_match:
        for line in exp_match.group(1).strip().split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                key, val = key.strip().lower(), val.strip()
                if key == "submit": action["submit"] = val.lower() in ("true", "yes", "1")
                else:
                    try: action["params"][key] = float(val)
                    except ValueError: pass

    diag_match = re.search(r"<diagnosis>(.*?)</diagnosis>", text, re.DOTALL | re.IGNORECASE)
    if diag_match:
        for line in diag_match.group(1).strip().split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                key, val = key.strip().lower(), val.strip()
                if "bottleneck" in key: action["primary_bottleneck"] = val
    return action


# ─── PROMPT BUILDER ─────────────────────────────────────────────────────────
def _fmt(v):
    if v == 0: return "0"
    if abs(v) >= 1e13 or (abs(v) < 0.01 and v != 0): return f"{v:.2e}"
    if abs(v) >= 1000: return f"{v:.0f}"
    return f"{v:.2f}"

def build_prompt(obs: dict) -> str:
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
    prompt += (f"═══ CURRENT STATE ═══\n  Experiments run: {step - 1}/{step - 1 + budget}  "
               f"Budget: {budget}  Best yield: {best:.1f}%  Target: >{target}%\n  Phase: {phase.upper()} — {phase_hint}\n")

    if fin_fb: prompt += f"\n⚠ FINANCIAL CONTROLLER:\n  {fin_fb}\n"
    if reviewer_fb: prompt += f"\n⚠ SENIOR REVIEWER:\n  {reviewer_fb}\n"

    prompt += "\n═══ EXPERIMENT HISTORY ═══\n"
    if not history:
        prompt += "  No experiments yet.\n"
    else:
        recent_history = history[-5:]
        for rec in recent_history:
            param_str = ", ".join(f"{k}={_fmt(v)}" for k, v in rec["params"].items())
            prompt += f"  Exp {rec['step']}: {param_str}\n    → Yield: {rec['yield_pct']:.1f}%  Defects: {rec['defect']}\n"

    prompt += "\n═══ PARAMETER SPACE ═══\n"
    for p in active_params:
        if p in param_ranges:
            lo, hi = param_ranges[p]
            prompt += f"  {p}: [{_fmt(lo)} — {_fmt(hi)}]\n"

    prompt += "\n═══ YOUR ACTION ═══\nYou MUST think step-by-step before acting. Analyze the history and constraints.\n"
    prompt += "Output your response strictly following this XML format:\n<think>\n  [Your reasoning]\n</think>\n"
    prompt += "<diagnosis>\n  primary_bottleneck: [param]\n</diagnosis>\n"
    prompt += "<experiment>\n  submit: false\n  temp: [value]\n  etch_time: [value]\n  # ... all other params\n</experiment>\n\n"
    prompt += "BEGIN YOUR RESPONSE NOW:\n<think>\n"
    return prompt


# ─── EPISODE RUNNER ─────────────────────────────────────────────────────────
def run_episode(model, tokenizer, difficulty: int = 1) -> tuple[list, float]:
    print(f"\n🚀 Starting Episode (Difficulty {difficulty})...")
    obs = requests.post(f"{SPACE_URL}/reset", json={"difficulty": difficulty}, headers=API_HEADERS).json()
    trajectory, total_reward, done = [], 0.0, False

    while not done:
        step = obs.get("step", 1)
        prompt = build_prompt(obs)

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        response_text = "<think>\n" + response_text
        action_dict = parse_action(response_text, obs.get("active_params", []))

        # GRPO: Capture Reference Log Probs
        full_inputs = tokenizer(prompt + response_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            ref_logits = model(**full_inputs).logits
            shift_logits = ref_logits[0, prompt_length - 1:-1, :]
            shift_labels = full_inputs.input_ids[0, prompt_length:]
            ref_log_probs_all = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            ref_action_lp = ref_log_probs_all.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        print(f"\n[{'=' * 55}]\n🔄 STEP {step} | Target: {obs.get('task_target', 'Yield Optimization')}")
        res = requests.post(f"{SPACE_URL}/step", json=action_dict, headers=API_HEADERS)
        if res.status_code != 200: break

        step_data = res.json()
        obs, reward, done = step_data["observation"], step_data["rewards"]["total"], step_data["done"]
        print(f"📈 Yield: {obs.get('current_best_yield', 0)}% | Step Reward: +{reward:.3f}")

        trajectory.append({
            "prompt": prompt,
            "response": response_text,
            "reward": reward,
            "prompt_length": prompt_length,
            "ref_log_probs": ref_action_lp.cpu(),
        })
        total_reward += reward

    print(f"🏁 Episode done! Final Yield: {obs.get('current_best_yield', 0)}% | Total reward: {total_reward:.3f}")
    return trajectory, total_reward


# ─── GRPO TRAINING LOOP ─────────────────────────────────────────────────────
def train(model, tokenizer):
    wandb.init(
        project="tensor-titans-fab",
        name="A10G-150-Epoch-Burn",
        config={"epochs": EPOCHS, "group_size": GROUP_SIZE, "learning_rate": LR},
    )
    optimizer = AdamW(model.parameters(), lr=LR)
    print(f"⚙️  GRPO ready | epochs={EPOCHS}  group_size={GROUP_SIZE}  lr={LR}\n")

    try:
        for epoch in range(EPOCHS):
            print(f"\n{'=' * 50}\n📈 EPOCH {epoch + 1}/{EPOCHS}")

            current_diff = 1 if epoch < 50 else (2 if epoch < 100 else 3)
            group_trajectories = []
            FastLanguageModel.for_inference(model)

            for g in range(GROUP_SIZE):
                print(f"\n--- Rollout {g + 1}/{GROUP_SIZE} ---")
                traj, _ = run_episode(model, tokenizer, difficulty=current_diff)
                if traj: group_trajectories.append(traj)

            if len(group_trajectories) < 2: continue

            # Advantage Normalisation
            all_step_rewards = [s["reward"] for traj in group_trajectories for s in traj]
            rewards_tensor   = torch.tensor(all_step_rewards, dtype=torch.float32)
            rewards_tensor   = torch.clamp(rewards_tensor, min=-1.5, max=1.5)
            r_mean           = rewards_tensor.mean()
            r_std            = rewards_tensor.std() + 1e-8
            print(f"\n📊 Step Reward Mean: {r_mean.item():.3f} | Std: {r_std.item():.3f}")

            torch.cuda.empty_cache()

            # ── Phase 3: Gradient Update ──
            FastLanguageModel.for_training(model)
            optimizer.zero_grad()

            total_steps = sum(len(traj) for traj in group_trajectories)
            total_loss = 0.0

            for traj in group_trajectories:
                for step_data in traj:
                    step_adv = (step_data["reward"] - r_mean.item()) / r_std.item()

                    full_inputs   = tokenizer(step_data["prompt"] + step_data["response"], return_tensors="pt").to("cuda")
                    prompt_length = step_data["prompt_length"]

                    logits = model(**full_inputs).logits
                    shift_logits = logits[0, prompt_length - 1:-1, :]
                    shift_labels = full_inputs.input_ids[0, prompt_length:]

                    log_probs        = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                    action_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

                    # Shape Guard for BPE mismatches
                    ref_log_probs = step_data["ref_log_probs"].to("cuda")
                    min_len       = min(action_log_probs.shape[0], ref_log_probs.shape[0])
                    action_log_probs, ref_log_probs = action_log_probs[:min_len], ref_log_probs[:min_len]

                    # GRPO Logic (Ratio + Clip)
                    ratio         = torch.exp(action_log_probs - ref_log_probs)
                    clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                    step_loss     = -torch.min(ratio * step_adv, clipped_ratio * step_adv).sum()

                    # Gradient Accumulation (VRAM Hack)
                    (step_loss / total_steps).backward()
                    total_loss += step_loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            avg_loss = total_loss / total_steps
            wandb.log({"epoch": epoch + 1, "mean_reward": r_mean.item(), "grpo_loss": avg_loss})
            print(f"✅ Update done | GRPO Loss: {avg_loss:.4f}")

            if (epoch + 1) % 25 == 0:
                print(f"💾 Saving checkpoint at epoch {epoch + 1}...")
                ckpt_path = f"/tmp/checkpoint-epoch-{epoch + 1}"
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

        # Final Save Logic
        print("\n🎉 Training complete!")
        final_path = "/tmp/final_model_tensor_titans"
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        
        try:
            print("☁️ Pushing final model to Hugging Face Hub...")
            model.push_to_hub("tensor-titans-qwen-7b-grpo", token=HF_TOKEN)
            tokenizer.push_to_hub("tensor-titans-qwen-7b-grpo", token=HF_TOKEN)
            print("🚀 Model successfully pushed to your HF Profile!")
        except Exception as e:
            print(f"⚠️ Hub push failed: {e}. Find files in /tmp/.")

    finally:
        wandb.finish()


# ─── ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"Env server URL : {SPACE_URL}\n")

    model, tokenizer = load_model()

    FastLanguageModel.for_inference(model)
    print("\n🧪 Smoke-test episode...")
    traj, score = run_episode(model, tokenizer, difficulty=1)
    
    train(model, tokenizer)