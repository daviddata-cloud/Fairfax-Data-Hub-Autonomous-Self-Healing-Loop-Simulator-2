"""
Standalone tests for weather_agent.py — no server, no Ollama, no browser.

  1. Step-Back problem->method mapping (matches the slide's examples)
  2. Pseudo-spectral solver dynamics (field moves & crosses the threshold)

Run:  python test_ps.py
"""
from weather_agent import (
    map_problem_to_method,
    map_problem_to_action,
    run_pseudo_spectral,
    SAFETY_THRESHOLD_F,
)

# ---------------------------------------------------------------------------
# 1. STEP-BACK: problem features -> method features (the slide)
# ---------------------------------------------------------------------------
print("Step-Back  map: problem -> method")
print("-" * 60)
cases = [
    {"dimension": "1D", "pde_type": "forward"},
    {"dimension": "2D", "pde_type": "forward"},
    {"dimension": "2D", "pde_type": "inverse"},   # slide's example -> PINN, 256
    {"dimension": "3D", "pde_type": "forward"},
]
for feats in cases:
    m = map_problem_to_method(feats)
    print(f"  {{{feats['dimension']}, {feats['pde_type']}}}"
          f"  ->  {{{m['solver']}, {m['collocation_points']}}}")

# Assert the slide's headline example maps as shown
slide = map_problem_to_method({"dimension": "2D", "pde_type": "inverse"})
assert slide == {"solver": "PINN", "collocation_points": 256}, slide
print("✅ PASS: {2D, inverse} -> {PINN, 256} (matches the slide)")

# ---------------------------------------------------------------------------
# 2. Pseudo-spectral solver: field moves and the state machine cycles
# ---------------------------------------------------------------------------
STEPS = 18
print(f"\nPseudo-spectral solver — {STEPS} steps  (threshold {SAFETY_THRESHOLD_F} °F)")
print(f"{'step':>4} | {'temp °F':>8} | {'wind mph':>8} | {'state':>8} | bar")
print("-" * 60)

temps, crit, norm = [], False, False
for i in range(1, STEPS + 1):
    w = run_pseudo_spectral(128)
    temp, wind = w["temp"], w["wind"]
    temps.append(temp)
    action = map_problem_to_action(temp)
    state = action["mode"]
    crit = crit or state == "CRITICAL"
    norm = norm or state == "NORMAL"
    bar = "#" * int(max(0, temp - 60) / 2)
    print(f"{i:>4} | {temp:>8.1f} | {wind:>8.1f} | {state:>8} | {bar}")

print("-" * 60)
print(f"min temp: {min(temps):.1f} °F   max temp: {max(temps):.1f} °F")
assert max(temps) != min(temps), "❌ Temperature never changed — solver is static!"
if crit and norm:
    print("✅ PASS: temperature both rises above and falls below the threshold (state cycles).")
elif crit:
    print("⚠️  Only CRITICAL reached — raise PS_COOLING.")
else:
    print("⚠️  Never reached CRITICAL — raise PS_INJECT.")