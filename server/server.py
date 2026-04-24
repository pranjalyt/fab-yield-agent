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

from environment.env import FabYieldEnv, FabAction, FabObservation

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Semiconductor Fab Yield Optimization Environment",
    description="RL Training Environment for OpenEnv Hackathon — Wafer Yield Agent",
    version="1.0.0",
)

# Global env instance (stateful per server process — mirrors hf-orchestrator)
env = FabYieldEnv(difficulty=1)
_current_obs: Optional[FabObservation] = None


# ─── Request / Response Models ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: int = 1   # 1=easy (5 params), 2=medium, 3=hard (15 params)


class StepRequest(BaseModel):
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
    """Hugging Face pings this to check if the Space is alive."""
    return {
        "status": "online",
        "message": "Fab Yield Optimization Environment is running.",
        "architecture": "OpenEnv",
        "current_step": _current_obs.step if _current_obs else None,
        "current_best_yield": _current_obs.current_best_yield if _current_obs else None,
    }


@app.post("/reset")
def reset_environment(request: ResetRequest = ResetRequest()):
    """Start a new episode. Returns initial observation."""
    global _current_obs
    obs = env.reset(seed=request.seed, difficulty=request.difficulty)
    _current_obs = obs
    return obs.model_dump()


@app.post("/step")
def step_environment(request: StepRequest) -> StepResponse:
    """Take one action in the environment. Returns next obs + reward breakdown."""
    global _current_obs

    if _current_obs is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first."
        )
    if _current_obs.done:
        raise HTTPException(
            status_code=400,
            detail=f"Episode is done (result: {_current_obs.episode_result}). Call /reset to start a new one."
        )

    action = FabAction(
        params=request.params,
        primary_bottleneck=request.primary_bottleneck,
        reasoning=request.reasoning,
        submit=request.submit,
    )

    obs, rewards, done, info = env.step(action)
    _current_obs = obs

    return StepResponse(
        observation=obs.model_dump(),
        rewards=rewards,
        done=done,
        info={
            "causal_structure": info.get("causal_structure", {}),
            "reviewer_approved": info.get("reviewer_approved", True),
            "defect": info.get("defect", "none"),
        }
    )


@app.get("/observation")
def get_observation():
    """Get the current observation without stepping."""
    if _current_obs is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return _current_obs.model_dump()


@app.get("/action_space")
def get_action_space():
    return env.action_space


@app.get("/observation_space")
def get_observation_space():
    return env.observation_space