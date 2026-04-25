"""
Solid-State Battery Optimization Environment.
Demonstrates Cross-Domain Sim-to-Real Transfer using the exact same RSM core math.
"""
from environment.env import FabYieldEnv, TASKS

# Override the Tasks for the Battery Domain
BATTERY_TASKS = [
    {
        "name": "High-Capacity EV Battery",
        "target": 95.0,
        "hint": "Maximize energy density. Budget is high, push lithium concentrations."
    },
    {
        "name": "Fast-Charging Electronics Cell",
        "target": 90.0,
        "hint": "Thermal stability is critical. Financial Controller limits high-temp sintering."
    }
]

# Parameter name mapping (Semiconductor -> Material Science)
PARAM_MAP = {
    "temp": "sintering_temp",
    "pressure": "compaction_pressure",
    "rf_power": "laser_intensity",
    "gas_ar": "argon_flow",
    "dopant": "lithium_concentration",
    # ... other parameters map similarly
}

class BatteryOptimizationEnv(FabYieldEnv):
    """
    Inherits the exact same physics and multi-agent logic as the FabEnv,
    but dynamically translates the parameter names and tasks.
    """
    def reset(self, seed: int = None, difficulty: int = None) -> dict:
        # Get the standard semiconductor observation
        obs_dict = super().reset(seed, difficulty)
        
        # Override the task to a Battery Task
        import random
        self._current_task = random.choice(BATTERY_TASKS)
        obs_dict["task_target"] = f"{self._current_task['name']} — {self._current_task['hint']}"
        
        # Translate the active parameters
        translated_active = [PARAM_MAP.get(p, p) for p in obs_dict["active_params"]]
        translated_ranges = {PARAM_MAP.get(k, k): v for k, v in obs_dict["param_ranges"].items()}
        
        obs_dict["active_params"] = translated_active
        obs_dict["param_ranges"] = translated_ranges
        
        return obs_dict