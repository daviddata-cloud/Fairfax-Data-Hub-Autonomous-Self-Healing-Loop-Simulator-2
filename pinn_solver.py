"""
pinn_solver.py — adapter that connects your real PINN repo to weather_agent.py

HOW TO USE
----------
1. Rename this file to `pinn_solver.py`.
2. Put it either:
      - next to weather_agent.py (simplest), OR
      - anywhere on your PYTHONPATH / installed as a package named `pinn_solver`.
3. Fill in the body of solve() to call YOUR PINN code (from the companion repo:
   https://github.com/daviddata-cloud/physics-informed-nn-PINN- ).

The agent will auto-detect this file — no changes needed in weather_agent.py.
It calls solve() only when the Step-Back mapping selects the PINN branch
(e.g. PROBLEM_FEATURES = {"dimension": "2D", "pde_type": "inverse"}).

CONTRACT
--------
solve(problem: dict) -> dict

  problem (input) keys:
    "dimension"           : "1D" | "2D" | "3D"
    "pde_type"            : "forward" | "inverse"
    "collocation_points"  : int   (e.g. 128 or 256)
    "ambient"             : float (baseline temperature, °F)

  return (output) keys:
    "temp" : float   # scalar the dashboard plots (e.g. peak temperature)
    "wind" : float   # scalar (e.g. max gradient magnitude); use 0.0 if N/A
    "text" : str     # short human-readable description of this solve

Keep it robust: if training fails, raise — the agent will log it and fall back
to its built-in stub for that step, so the loop never crashes.
"""


def solve(problem: dict) -> dict:
    dimension = problem.get("dimension", "1D")
    pde_type = problem.get("pde_type", "forward")
    n_colloc = int(problem.get("collocation_points", 128))
    ambient = float(problem.get("ambient", 72.0))

    # -------------------------------------------------------------------------
    # TODO: replace everything below with a call into YOUR PINN repo.
    #
    # Example shape (pseudo-code — adapt to your actual API):
    #
    #   from my_pinn_package import ThermalPINN
    #   model = ThermalPINN(dim=dimension, mode=pde_type, n_colloc=n_colloc)
    #   model.train(epochs=2000)
    #   field = model.predict_field()          # numpy array
    #
    #   import numpy as np
    #   peak = float(np.max(field))
    #   grad = float(np.max(np.abs(np.gradient(field))))
    #   return {
    #       "temp": peak,
    #       "wind": min(40.0, grad * 2.0),
    #       "text": f"PINN {dimension}/{pde_type} solve, {n_colloc} pts",
    #   }
    # -------------------------------------------------------------------------

    # Until you wire in the real model, return a clearly-labeled placeholder so
    # you can confirm the adapter path is being used end-to-end.
    return {
        "temp": ambient + 20.0,
        "wind": 10.0,
        "text": f"PINN(adapter placeholder) {dimension}/{pde_type}, {n_colloc} pts",
    }


# Optional: quick self-test —  python pinn_solver.py
if __name__ == "__main__":
    demo = {"dimension": "2D", "pde_type": "inverse",
            "collocation_points": 256, "ambient": 72.0}
    print("solve(", demo, ") ->")
    print("   ", solve(demo))