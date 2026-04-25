import requests

# ⚠️ Make sure this URL matches your actual Space URL exactly
SPACE_URL = "https://pranjalzetsu-semiconductor-yield-agent.hf.space" 

print("--- 1. Testing /reset ---")
reset_res = requests.post(f"{SPACE_URL}/reset", json={"difficulty": 1})
obs = reset_res.json()
print(f"Active Params: {obs.get('active_params')}")
print(f"Budget Remaining: {obs.get('budget_remaining')}")

print("\n--- 2. Testing /step ---")
dummy_action = {
    "params": {
        "temp": 190.0, 
        "etch_time": 60.0, 
        "pressure": 2.2, 
        "dopant": 1.4e15, 
        "spin_speed": 2500.0
    },
    "primary_bottleneck": "temp",
    "reasoning": "Verifying API handshake.",
    "submit": False
}

step_res = requests.post(f"{SPACE_URL}/step", json=dummy_action)
step_data = step_res.json()

print(f"New Step: {step_data['observation']['step']}")
print(f"Yield Reward: {step_data['rewards']['yield']}")
print(f"Total Reward: {step_data['rewards']['total']}")