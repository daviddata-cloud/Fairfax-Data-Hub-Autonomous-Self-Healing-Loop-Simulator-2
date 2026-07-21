import os
import json
import time
import re
import threading
import requests
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# =============================================================================
# STEP-BACK FRAMEWORK  (matches the "Step back — and How Do We Do It?" slide)
# -----------------------------------------------------------------------------
# The slide's idea: don't jump to a solver. First read the PROBLEM FEATURES,
# then MAP problem -> method, producing METHOD FEATURES, then act.
#
#   Toy example:  {day, rain}            -> map -> {breakfast, transport}
#                 e.g. {Monday, yes}     ->        {breakfast, car}
#
#   PDE example:  {dimension, pde_type}  -> map -> {solver, #collocation_points}
#                 e.g. {2D, inverse}     ->        {PINN, 256}
#
# This agent implements BOTH mappings explicitly (see map_problem_to_method
# and map_problem_to_action below), then runs the chosen solver.
# =============================================================================

# Feature vocabularies (the boxes on the slide)
DIMENSIONS   = ("1D", "2D", "3D")
PDE_TYPES    = ("forward", "inverse")
SOLVERS      = ("pseudo-spectral", "PINN")
COLLOCATION  = (128, 256)

# The governing safety threshold used by the action mapping.
SAFETY_THRESHOLD_F = 85.0

# ---- The current problem we are solving (its PROBLEM FEATURES) --------------
# Change these to see the Step-Back map pick a different method, exactly like
# the slide: {2D, inverse} -> {PINN, 256}.
PROBLEM_FEATURES = {
    "dimension": "1D",       # one of DIMENSIONS
    "pde_type":  "forward",  # one of PDE_TYPES
}

def map_problem_to_method(features: dict) -> dict:
    """STEP-BACK, PDE branch: problem features -> method features.

    Mirrors the slide's 'map: problem -> method' with a small rule table.
    {dimension, pde_type} -> {solver, #collocation_points}
    """
    dim  = features.get("dimension", "1D")
    ptype = features.get("pde_type", "forward")

    # Rule of thumb (illustrative, editable):
    #  - inverse problems or high dimension -> PINN with more collocation points
    #  - low-dim forward problems           -> classical pseudo-spectral, fewer points
    if ptype == "inverse" or dim == "3D":
        method = {"solver": "PINN", "collocation_points": 256}
    else:
        method = {"solver": "pseudo-spectral", "collocation_points": 128}

    # Safety clamps so we never emit an off-vocabulary value
    if method["solver"] not in SOLVERS:
        method["solver"] = "pseudo-spectral"
    if method["collocation_points"] not in COLLOCATION:
        method["collocation_points"] = 128
    return method

def map_problem_to_action(temp_f: float) -> dict:
    """STEP-BACK, toy branch: problem features -> action features.

    Mirrors the slide's {day, rain} -> {breakfast, transport}. Here the
    problem feature is the peak temperature and the action features are the
    infrastructure settings written to server_config.py.
    {temperature} -> {mode, air_intake, max_threads}
    """
    if temp_f > SAFETY_THRESHOLD_F:
        return {"mode": "CRITICAL",
                "air_intake": "INTERNAL_RECIRCULATION",
                "max_threads": 8}
    return {"mode": "NORMAL",
            "air_intake": "EXTERNAL_FRESH_AIR",
            "max_threads": 64}

# 1. Instantiate local offline Llama configuration
#    format="json" forces Ollama to emit valid JSON so parsing doesn't fail.
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.1,
    base_url="http://127.0.0.1:11434",
    format="json",
)

CURRENT_STATE = "UNKNOWN"
LOG_DIR = "telemetry_logs"
LOOP_COUNT = 0

# Server endpoint the dashboard listens on
SERVER_URL = "http://127.0.0.1:5000/update-status"

# --- File-watch config (autonomous background-color feature) ---
WATCH_FILE = "sensor_input.json"     # the file the agent watches
_last_watch_content = None           # tracks last-seen file CONTENT (solid on all OSes)
HEX_RE = re.compile(r'^#[0-9a-fA-F]{6}$')

# Deterministic theme palette — code decides the color, not the model.
THEME_CRITICAL = "#2a0d0d"   # dark red
THEME_NORMAL   = "#0b1f2a"   # dark blue / teal
THEME_DEFAULT  = "#0b0f19"   # baseline dark

CRITICAL_WORDS = ("critical", "alarm", "alert", "hot", "heat", "overload",
                  "emergency", "fire", "storm", "danger", "warning", "fault")
NORMAL_WORDS   = ("normal", "calm", "ok", "okay", "safe", "clear",
                  "cool", "fine", "stable", "baseline")

# =============================================================================
# PSEUDO-SPECTRAL THERMAL FIELD MODEL  (the "pseudo-spectral" solver box)
# -----------------------------------------------------------------------------
# 1-D heat-diffusion PDE solved pseudo-spectrally (spectral in space via FFT,
# exact integrating factor in time), with operator-split Newtonian cooling.
# =============================================================================
PS_N        = 128         # spatial grid points (matches a collocation choice)
PS_L        = 10.0        # domain length
PS_NU       = 0.25        # thermal diffusivity (nu)
PS_DT       = 0.05        # time step
PS_SUBSTEPS = 8           # integration substeps per agent loop
PS_AMBIENT  = 72.0        # ambient baseline temperature (°F)
PS_COOLING  = 0.9         # Newtonian relaxation rate toward ambient
PS_INJECT   = 40.0        # heat-pulse amplitude during a heatwave front

_ps_x     = np.linspace(0, PS_L, PS_N, endpoint=False)
_ps_k     = 2 * np.pi * np.fft.fftfreq(PS_N, d=PS_L / PS_N)
_ps_field = PS_AMBIENT + 2.0 * np.exp(-((_ps_x - PS_L / 2) ** 2) / 0.5)

def _ps_diffuse_once(u, dt, nu, k):
    """One pseudo-spectral diffusion step for du/dt = nu * d2u/dx2 (periodic).
    Exact integrating factor exp(-nu k^2 dt) applied in Fourier space."""
    u_hat = np.fft.fft(u)
    u_hat *= np.exp(-nu * (k ** 2) * dt)
    return np.real(np.fft.ifft(u_hat))

def _ps_spectral_gradient(u, k):
    """Spatial derivative via spectral differentiation: du/dx = ifft(i k fft(u))."""
    return np.real(np.fft.ifft(1j * k * np.fft.fft(u)))

def run_pseudo_spectral(collocation_points: int):
    """Evolve the pseudo-spectral thermal field and derive telemetry.
    collocation_points is accepted so the chosen method feature is honored."""
    global _ps_field, _ps_x, _ps_k, PS_N, LOOP_COUNT
    LOOP_COUNT += 1

    # Honor the method's collocation-point choice by resizing the grid if needed
    if collocation_points != PS_N:
        PS_N = collocation_points
        _ps_x = np.linspace(0, PS_L, PS_N, endpoint=False)
        _ps_k = 2 * np.pi * np.fft.fftfreq(PS_N, d=PS_L / PS_N)
        _ps_field = PS_AMBIENT + 2.0 * np.exp(-((_ps_x - PS_L / 2) ** 2) / 0.5)

    if (LOOP_COUNT // 3) % 2 == 1:
        print(f"\n📈 [PS Solver] Step {LOOP_COUNT}: injecting heat-pulse front.")
        _ps_field += PS_INJECT * np.exp(-((_ps_x - PS_L / 2) ** 2) / 0.4)
        scenario = "Pseudo-spectral heat-pulse front injected into thermal field"
    else:
        print(f"\n📉 [PS Solver] Step {LOOP_COUNT}: field relaxing toward ambient.")
        scenario = "Pseudo-spectral thermal field relaxing toward ambient baseline"

    for _ in range(PS_SUBSTEPS):
        _ps_field = _ps_diffuse_once(_ps_field, PS_DT, PS_NU, _ps_k)
        _ps_field += -PS_COOLING * (_ps_field - PS_AMBIENT) * PS_DT

    peak_temp = float(np.max(_ps_field))
    grad = _ps_spectral_gradient(_ps_field, _ps_k)
    wind = float(min(40.0, np.max(np.abs(grad)) * 2.0))
    return {"temp": round(peak_temp, 1), "wind": round(wind, 1), "text": scenario}

def run_pinn_stub(collocation_points: int):
    """Placeholder for the PINN solver branch (the other box on the slide).

    A full PINN lives in the companion repo; here we provide a lightweight
    stand-in so the {solver: PINN} branch of the mapping is actually taken.
    It produces telemetry of the same shape as the pseudo-spectral solver.
    """
    global LOOP_COUNT
    LOOP_COUNT += 1
    print(f"\n🧠 [PINN Solver:stub] Step {LOOP_COUNT}: "
          f"would train PINN with {collocation_points} collocation points.")
    phase = (LOOP_COUNT // 3) % 2
    if phase == 1:
        return {"temp": 96.0, "wind": 22.0,
                "text": f"PINN(stub) inverse-mode estimate, {collocation_points} pts"}
    return {"temp": 74.0, "wind": 6.0,
            "text": f"PINN(stub) forward-mode estimate, {collocation_points} pts"}

# --- Real-PINN adapter -------------------------------------------------------
# Tries to load a real PINN solver from the companion repo. Falls back to the
# stub above if none is found, so the agent always runs.
#
# The agent expects the PINN repo to expose ONE function:
#     solve(problem: dict) -> dict   # returns {"temp", "wind", "text"}
# where problem = {"dimension","pde_type","collocation_points","ambient"}.
# See pinn_solver_template.py for a copy-paste wrapper.
_PINN_SOLVE = None          # will hold the real solve() if found
_PINN_SOURCE = "stub"       # for logging: which backend is active

def _load_real_pinn():
    """Locate a real PINN solver. Order: installed package -> local file -> None."""
    global _PINN_SOLVE, _PINN_SOURCE
    # 1) installed package:  pip install pinn_solver   (exposes solve())
    try:
        import pinn_solver  # type: ignore
        if hasattr(pinn_solver, "solve"):
            _PINN_SOLVE = pinn_solver.solve
            _PINN_SOURCE = "package:pinn_solver"
            return
    except Exception:
        pass
    # 2) local file next to this script:  ./pinn_solver.py  (exposes solve())
    try:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "pinn_solver.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("pinn_solver_local", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            if hasattr(mod, "solve"):
                _PINN_SOLVE = mod.solve
                _PINN_SOURCE = "local:pinn_solver.py"
                return
    except Exception as e:
        print(f"⚠️ [PINN] local pinn_solver.py found but failed to load: {e}")
    # 3) nothing found -> stub
    _PINN_SOLVE = None
    _PINN_SOURCE = "stub"

_load_real_pinn()

def run_pinn(collocation_points: int):
    """Run the real PINN if available, else the stub. Normalizes the result
    to {'temp','wind','text'} so downstream code is identical either way."""
    global LOOP_COUNT
    if _PINN_SOLVE is None:
        return run_pinn_stub(collocation_points)

    LOOP_COUNT += 1
    problem = {
        "dimension": PROBLEM_FEATURES.get("dimension", "1D"),
        "pde_type": PROBLEM_FEATURES.get("pde_type", "forward"),
        "collocation_points": collocation_points,
        "ambient": PS_AMBIENT,
    }
    print(f"\n🧠 [PINN Solver:{_PINN_SOURCE}] Step {LOOP_COUNT}: "
          f"solving with {collocation_points} collocation points.")
    try:
        out = _PINN_SOLVE(problem) or {}
        return {
            "temp": round(float(out.get("temp", PS_AMBIENT)), 1),
            "wind": round(float(out.get("wind", 0.0)), 1),
            "text": str(out.get("text", f"PINN estimate, {collocation_points} pts")),
        }
    except Exception as e:
        print(f"⚠️ [PINN] real solver raised '{e}', falling back to stub this step.")
        return run_pinn_stub(collocation_points)

def solve_with_method(method: dict):
    """Dispatch to the solver chosen by map_problem_to_method (the branch that
    makes this a real problem->method framework, not a fixed pipeline)."""
    if method["solver"] == "PINN":
        return run_pinn(method["collocation_points"])
    return run_pseudo_spectral(method["collocation_points"])

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def classify_theme(content: str) -> str:
    """Map file content to a guaranteed-valid dark hex color, deterministically."""
    text = content.lower()
    if any(w in text for w in CRITICAL_WORDS):
        return THEME_CRITICAL
    if any(w in text for w in NORMAL_WORDS):
        return THEME_NORMAL
    return THEME_DEFAULT

# Step-Back-structured prompt: principle first, then problem, method, action.
prompt_template = """
You are a self-healing datacenter engineering agent in Fairfax, VA.
Reason using the STEP-BACK framework before deciding.

[STEP 1 — STEP BACK: PRINCIPLE]
State the general engineering principle governing datacenter thermal safety
(one sentence: why sustained high temperature threatens hardware).

[STEP 2 — PROBLEM]
Telemetry below was produced by the chosen numerical solver.
Peak Temperature: {temp} °F
Gradient-derived Wind: {wind} mph
Field state: {text}

[STEP 3 — METHOD]
Compare the peak temperature against the 85 °F safety threshold.

[STEP 4 — ACTION]
- IF temp > 85°F: CRITICAL. AIR_INTAKE = 'INTERNAL_RECIRCULATION', MAX_THREADS = 8.
- IF temp <= 85°F: NORMAL. AIR_INTAKE = 'EXTERNAL_FRESH_AIR', MAX_THREADS = 64.

Output ONLY this raw JSON, no markdown, no commentary:
{{
    "principle": "the step-back principle, one sentence",
    "mode": "CRITICAL" or "NORMAL",
    "reason": "one-line technical summary",
    "fixed_code": "the python assignment lines for server_config.py"
}}
"""

def extract_and_parse_json(raw_text: str) -> dict:
    """Regex block extractor to handle output formatting variances safely."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Model returned empty output.")

    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON block found. Raw output was: {raw_text[:200]!r}")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Found braces but JSON was invalid: {e}. Snippet: {match.group(0)[:200]!r}")

def build_config_from_action(action: dict) -> str:
    """Deterministic config generation from the action-feature mapping."""
    return (f"AIR_INTAKE = '{action['air_intake']}'\n"
            f"MAX_THREADS = {action['max_threads']}")

def check_watched_file():
    """If WATCH_FILE content changed, map it to a validated dark color and push it."""
    global _last_watch_content

    if not os.path.exists(WATCH_FILE):
        return

    with open(WATCH_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if _last_watch_content is None:
        _last_watch_content = content
        return
    if content == _last_watch_content:
        return

    _last_watch_content = content
    print(f"\n📂 [File Watch] '{WATCH_FILE}' changed.")

    color = classify_theme(content)
    if not HEX_RE.match(color):
        color = THEME_DEFAULT

    print(f"🎨 [File Watch] Applying background {color} (content: {content[:60]!r})")
    requests.post(SERVER_URL, json={"bg_color": color}, timeout=3)

def start_file_watcher_thread():
    """Run the file watcher in its OWN thread so theming responds within ~1.5s."""
    def _loop():
        print(f"👀 [File Watch] Thread started — polling '{WATCH_FILE}' every 1.5s")
        while True:
            try:
                check_watched_file()
            except Exception as e:
                print(f"⚠️ [File Watch] Handled error: {e}")
            time.sleep(1.5)
    threading.Thread(target=_loop, daemon=True).start()

def start_autonomous_engine():
    global CURRENT_STATE
    print(f"🤖 [Engine Init] 100% Autonomous loop engaged. Syncing logs into './{LOG_DIR}/'")

    # ---- STEP-BACK (once): map the PROBLEM FEATURES -> METHOD FEATURES -------
    method = map_problem_to_method(PROBLEM_FEATURES)
    print("🧭 [Step-Back] map: problem -> method")
    print(f"    problem features {{{PROBLEM_FEATURES['dimension']}, {PROBLEM_FEATURES['pde_type']}}}"
          f"  ->  method features {{{method['solver']}, {method['collocation_points']}}}")

    while True:
        try:
            # ============ PROBLEM (run the chosen solver) ============
            weather = solve_with_method(method)

            # ============ ACTION MAPPING (toy branch) ============
            action = map_problem_to_action(weather["temp"])
            authoritative_mode = action["mode"]

            prompt = PromptTemplate.from_template(prompt_template)
            chain = prompt | llm

            print("🤖 Llama processing telemetry inputs (step-back reasoning)...")
            response = chain.invoke({"temp": weather["temp"], "wind": weather["wind"], "text": weather["text"]})
            result = extract_and_parse_json(response.content)

            inferred_mode = authoritative_mode
            reason = result.get("reason", "Baseline operation profiles confirmed.")
            principle = result.get("principle", "")
            if principle:
                print(f"🧠 [Step-Back Principle] {principle}")

            model_said = result.get("mode", "NORMAL")
            if model_said != authoritative_mode:
                print(f"⚠️ [Model Disagreement] Model claimed '{model_said}' but temp={weather['temp']}°F → forcing '{authoritative_mode}'")

            # Deterministic config from the action features
            fixed_code = build_config_from_action(action)

            # ============ ACT (self-heal on drift) ============
            if inferred_mode != CURRENT_STATE:
                print(f"🚨 [STATE DRIFT INGESTED] Transitioning state: '{CURRENT_STATE}' -> '{inferred_mode}'")
                print(f"💡 Reason: {reason}")

                print("💾 Overwriting active production target file 'server_config.py'...")
                with open("server_config.py", "w", encoding="utf-8") as f:
                    f.write(fixed_code)

                timestamp_id = time.strftime("%Y%m%d_%H%M%S")
                log_filename = f"{LOG_DIR}/healed_config_{timestamp_id}_{inferred_mode}.py"
                print(f"📁 Creating historical log backup file on disk: {log_filename}")
                with open(log_filename, "w", encoding="utf-8") as f:
                    f.write(f"# Snapshot Logged: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Step-Back Method: {method['solver']}, {method['collocation_points']} pts\n")
                    f.write(f"# Step-Back Principle: {principle}\n")
                    f.write(f"# Agent Self-Healing Reason: {reason}\n")
                    f.write(fixed_code)

                CURRENT_STATE = inferred_mode
            else:
                print(f"✅ [System In Alignment] State matches active baseline ({CURRENT_STATE}). No file write required.")

            # Push frame to dashboard (includes the chosen method for display)
            requests.post(SERVER_URL, json={
                "is_alert": (CURRENT_STATE == "CRITICAL"),
                "location": "Fairfax, VA (Step-Back Stream Engine)",
                "temperature": int(weather["temp"]),
                "wind_speed": int(weather["wind"]),
                "condition": str(weather["text"]),
                "current_code": str(fixed_code),
                "method": f"{method['solver']} / {method['collocation_points']} pts"
            }, timeout=3)

        except Exception as error:
            print(f"⚠️ Custom Engine Handled Exception: {error}")

        print("⏳ Sleeping for 5 seconds before checking next telemetry state...")
        time.sleep(5)

if __name__ == "__main__":
    start_file_watcher_thread()
    start_autonomous_engine()