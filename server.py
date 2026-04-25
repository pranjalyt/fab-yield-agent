"""
server.py — FastAPI server for the Fab Yield RL Environment.

Mirrors hf-orchestrator/server.py pattern:
  - FastAPI app called 'app' (Uvicorn target: server:app)
  - /reset  POST endpoint → returns observation dict
  - /step   POST endpoint → returns obs + rewards + done + info
  - /       GET  health check (HF Spaces ping)
  - Port 7860 (HF Spaces requirement)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from environment.env import FabYieldEnv

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Semiconductor Fab & AI Hardware Optimization Environment",
    description="Multi-Agent RL Environment for OpenEnv Hackathon",
    version="1.1.0",
)

# Global env instance
env = FabYieldEnv(difficulty=1)

# ─── Request / Response Models ────────────────────────────────────────────────
class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: int = 1

class StepRequest(BaseModel):
    think: str = "" # NEW: Test-Time Compute
    params: Dict[str, float]
    primary_bottleneck: str = ""
    reasoning: str = ""
    submit: bool = False

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    rewards: Dict[str, float]
    done: bool
    info: Dict[str, Any]

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    try:
        state = env.state()
        step = state.get("step")
    except RuntimeError:
        step = None
    return {"status": "online", "architecture": "OpenEnv", "current_step": step}

@app.post("/reset")
def reset_environment(request: ResetRequest = ResetRequest()):
    # env.reset() now returns a standard dict
    obs_dict = env.reset(seed=request.seed, difficulty=request.difficulty)
    return obs_dict

@app.post("/step")
def step_environment(request: StepRequest) -> StepResponse:
    try:
        # Convert Pydantic request to dict for OpenEnv step()
        action_dict = request.model_dump()
        obs_dict, rewards, done, info = env.step(action_dict)
        
        return StepResponse(
            observation=obs_dict,
            rewards=rewards,
            done=done,
            info=info
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    """OpenEnv standard state reader."""
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/action_space")
def get_action_space():
    return env.action_space

@app.get("/observation_space")
def get_observation_space():
    return env.observation_space


import subprocess
import os

@app.on_event("startup")
def startup_event():
    print("🚀 Booting Server and triggering Training Run...")
    
    # 1. Set the exact environment variables
    train_env = os.environ.copy()
    train_env["EPOCHS"] = "150"          # Change to "150" when ready for the real burn
    train_env["GROUP_SIZE"] = "12"       # G=8 for the A10G math stability
    train_env["SPACE_URL"] = "http://localhost:7860"
    
    # 2. Launch train.py as an independent background process
    subprocess.Popen(["python", "train.py"], env=train_env)