# Fairfax-Data-Hub-Autonomous-Self-Healing-Loop-Simulator-2
An autonomous, offline AI agent that watches a telemetry signal, decides when it crosses a danger threshold, and rewrites its own configuration file to switch a (simulated) datacenter into a safety mode 

# 🌎 Fairfax Autonomous 3D Control Center

An **autonomous, offline AI agent** that watches a telemetry signal, decides when it crosses a danger threshold, and **rewrites its own configuration file to switch a (simulated) datacenter into a safety mode — with no human pressing a button.** It reasons with a **local Llama 3 model** (via Ollama) and shows everything on a live 3D dashboard. No cloud, no API keys — everything runs on your machine.

> **Honesty note (read this first):** This project does **not** read real-world weather. It generates a telemetry signal from a **physics simulation** (a pseudo-spectral heat-diffusion model) and treats that as the thing being monitored. Everywhere this README says "temperature" or "telemetry," it means a *simulated* signal. Being upfront about this matters — especially for a science-fair or contest setting.

---

## 📋 Table of Contents

1. [What This Project Is](#-what-this-project-is)
2. [What "Autonomous" Actually Means Here](#-what-autonomous-actually-means-here)
3. [Architecture](#-architecture)
4. [The Two Independent Loops](#-the-two-independent-loops)
5. [The Pseudo-Spectral Simulation](#-the-pseudo-spectral-simulation)
6. [Step-Back: Problem → Method Mapping](#-step-back-problem--method-mapping)
7. [Connecting a Real PINN Solver](#-connecting-a-real-pinn-solver)
8. [Installation](#-installation)
9. [How to Start](#-how-to-start)
10. [How to Stop](#-how-to-stop)
11. [Autonomous Theming (`sensor_input.json`)](#-autonomous-theming-sensor_inputjson)
12. [Testing](#-testing)
13. [Making It Faster](#-making-it-faster)
14. [Project Structure](#-project-structure)
15. [Configuration Notes](#-configuration-notes)
16. [Troubleshooting](#-troubleshooting)
17. [For Students / Contest Notes](#-for-students--contest-notes)
18. [Related Work](#-related-work)
19. [License](#-license)

---

## 🧩 What This Project Is
<img width="1908" height="945" alt="image" src="https://github.com/user-attachments/assets/3203a029-bbba-416d-8397-fa8b2168f383" />
<img width="1881" height="747" alt="image" src="https://github.com/user-attachments/assets/f6d85c90-5997-4807-ae26-f4ac245c4084" />

Two programs that cooperate:

- **`weather_agent.py`** — the autonomous engine. Every few seconds it computes a telemetry value, asks a local Llama 3 model to reason about it, **decides in code** whether the system is in a safe or critical state, and — when the state changes — **rewrites its own production config file** (`server_config.py`) and saves a timestamped backup. It also watches an external file to re-theme the dashboard.
- **`server.py`** — a Flask web server hosting a real-time dashboard: a live streaming line chart, a status panel, and an interactive **3D map of 6 Virginia locations** rendered with Three.js.

**Core safety principle of the design:** *the LLM reasons and narrates, but code makes every decision that matters.* The numeric threshold, the config it writes, and the theme color are all computed deterministically. The model never produces anything that gets executed or trusted blindly. This is what keeps an unpredictable small model from ever driving the system into a wrong or unsafe state.

---

## 🤖 What "Autonomous" Actually Means Here

It's easy to call something an "AI agent." Here is the concrete, defensible version:

**The agent runs a closed loop — perceive → reason → decide → act → repeat — with no human in the loop, and its "act" step changes a real file on disk.**

The single most important line to understand:

> When the simulated temperature crosses **85 °F**, the agent **overwrites `server_config.py`** by itself to switch the system into `CRITICAL` mode (recirculate air, throttle threads), then logs a backup. When it drops back below 85 °F, it rewrites the file back to `NORMAL`. **No human triggers this.**

That file-rewrite is the autonomous action. The red warning badge on the dashboard is just the *visible sign* of that decision — the real "action" is the self-modification of the system's configuration.

**Why it qualifies as an autonomous agent (not just a script):**
- **It acts on itself** — edits its own production config file in response to what it observes.
- **It self-heals** — detects drift from the desired state and corrects it without being told.
- **It runs continuously and independently** — two concurrent loops, no human trigger per cycle.
- **It reasons before acting** — an LLM step interprets each situation, while deterministic code guards every consequential decision.

---

## 🏗 Architecture

## Approach: AI Explains, Code Decides

### Core concept

This project demonstrates a general pattern for autonomous monitoring where an
AI reasons about a live signal but **deterministic code makes every real
decision**. The AI only explains what it sees in plain language; a human-set
threshold rule — executed by code — makes the actual alert/action call. Because
the AI never holds decision authority, a wrong or hallucinated model output can
never trigger an unsafe action.

The pattern has two layers:

- **Signal layer** — produces the value to watch. The method changes by domain.
- **Decision layer** — always the same: perceive the signal, let the AI explain,
  then let code compare against a human-defined threshold and decide, with a
  human approval gate for any production action.

Only the signal layer changes per use case. The decision layer stays constant.
That is what makes the approach general.

### Two decisions, kept separate

- **Step-Back (choose the method):** before acting, read the problem's features
  and pick the right tool. This selects *how* to analyze the signal.
- **Threshold decision (make the alert correct):** deterministic code compares
  the signal to a human-set rule. This is what makes any single alert
  trustworthy — not the AI, and not the analysis method.

These are different steps. Step-Back picks the tool; code makes the call.

### Domain 1 — Temperature (validated on a simulated signal)

- Signal layer: Step-Back selects a solver — pseudo-spectral (forward problems,
  128 collocation points) or a PINN / Physics-Informed Neural Network (inverse
  problems, 256 collocation points) — to solve a Partial Differential Equation
  (PDE) and produce the value.
- Decision layer: code compares the value to a fixed threshold (e.g. 85°F) and
  switches to safety mode if exceeded; the AI's opinion is used only for the
  human-readable narration and is overridden if it disagrees.
- Equations are used here because temperature is a continuous physical field and
  we want to **predict** its trend or **fill in** values where sensors are
  missing (sparse data / uncertain boundary conditions).

### Domain 2 — IT logs (design, not yet built)

- IT logs are not a physical field and have no PDE. The signal layer is replaced
  with a knowledge-graph pipeline: stream logs into a store, continuously
  extract entities and relations into a knowledge graph, then use **GraphRAG**
  retrieval to pull the relevant sub-graph and an LLM to produce a root-cause
  hypothesis and a plain-English explanation.
- GraphRAG finds **relationships and root-cause candidates**, not proven
  causation. It narrows the search and proposes the likely cause and its
  propagation path; confirming true cause-and-effect still requires causal
  analysis or controlled testing.
- The heavy monitoring is done by streaming anomaly detection (cheap, fast,
  runs 24/7). The LLM is triggered **on demand** to explain — it is not the
  thing watching continuously. Think "sentinel vs. expert": lightweight
  detection stands guard; GraphRAG + LLM is called in only when there's
  something to analyze.
- The decision layer is identical to the temperature case: code compares to a
  human-set threshold and decides, with a human gate for production actions.

### Honest status (what is built vs. what is new)

- **Built (A):** the autonomous perceive → reason → decide → act loop with
  code as the decision authority. Runs today on a **simulated** signal from a
  physics simulation, not real-world data.
- **Built elsewhere (B):** GraphRAG with a typed knowledge graph, semantic
  retrieval, and page-level provenance — developed for research-document
  navigation, not for streaming IT logs.
- **New work required:** the pipeline that turns **streaming logs into a
  continuously updated knowledge graph**, then connects that GraphRAG retrieval
  into the reasoning step of the loop. This bridge exists in neither A nor B
  today.
- **Not fine-tuning:** this design stores data externally and retrieves it at
  query time (GraphRAG), rather than baking knowledge into model weights.
  Fine-tuning would only be considered later, and only to help a model read
  domain-specific log formats — decision logic still stays in external data and
  code rules.

### One-paragraph summary

I built a working autonomous-agent prototype where an AI reasons about a live
signal but code makes every real decision — so the model can't cause an unsafe
action. The same loop can watch our Azure IT logs and trigger remediation, or
watch an early-response signal and draft an alert for human review. Today it
runs on a simulated signal; the next step is connecting real data sources
(including a streaming-logs-to-knowledge-graph pipeline for GraphRAG) and adding
a human approval gate for production.


<img width="1022" height="952" alt="image" src="https://github.com/user-attachments/assets/cfd83b38-5a96-4f6f-8255-c42b68aeab3f" />

<img width="1022" height="949" alt="image" src="https://github.com/user-attachments/assets/d14dff39-9d28-42f1-8fae-a86c33d96af0" />



```
┌─────────────────────┐         ┌──────────────────────┐
│  weather_agent.py   │         │      server.py       │
│  (autonomous loop)  │         │   (Flask dashboard)  │
├─────────────────────┤         ├──────────────────────┤
│ 1. Simulate signal  │         │  /                   │  ← dashboard HTML
│ 2. Watch sensor file│──POST──▶│  /update-status      │  ← receives agent data
│ 3. Ask Llama (JSON) │         │  /api/history        │  ← line chart data
│ 4. Decide in CODE   │         │  /api/locations      │  ← 3D map data (6 cities)
│ 5. Heal config file │         │  /api/style          │  ← autonomous bg color
│ 6. Log + POST       │         └──────────────────────┘
└─────────────────────┘
```

**Decision flow each loop (~every 5 s):**
1. Run the chosen solver to produce a telemetry value (temperature + wind).
2. (In a separate thread) check `sensor_input.json`; if it changed, re-theme the dashboard.
3. Send the telemetry to Llama for a Step-Back reasoning summary (JSON output enforced).
4. **Code** decides `CRITICAL` if `temp > 85 °F`, else `NORMAL` (the model's opinion is only used for the human-readable text; if it disagrees, code overrides it and logs `⚠️ Model Disagreement`).
5. If the state changed, overwrite `server_config.py` and save a timestamped backup log.
6. POST the frame to the dashboard.

---

## 🔁 The Two Independent Loops

**Loop 1 — the main autonomy cycle (`start_autonomous_engine`)**

```
        ┌────────────────────────────────────────────────┐
        │                                                │
        ▼                                                │
  ┌───────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐
  │  PERCEIVE │──▶│   REASON   │──▶│  DECIDE  │──▶│   ACT    │
  │ (simulate)│   │  (Llama)   │   │  (code)  │   │ (rewrite │
  │           │   │            │   │          │   │  config) │
  └───────────┘   └────────────┘   └──────────┘   └──────────┘
```

**Loop 2 — the file-watcher thread (`start_file_watcher_thread`)**

A **daemon thread** polls `sensor_input.json` every 1.5 s, fully decoupled from the (slow) LLM loop. When the file's *content* changes, it maps the text to a validated dark background color and pushes it to the dashboard. Running it in its own thread is what makes theming respond in ~1.5 s even while Llama is mid-inference — a single-threaded design would make theming wait behind every model call.

---

## 🧮 The Pseudo-Spectral Simulation

The telemetry is **computed**, not hardcoded. `weather_agent.py` solves a 1-D heat-diffusion PDE using a **pseudo-spectral method**:

```
∂u/∂t = ν · ∂²u/∂x²      (heat diffusion)
```

- **Spectral in space** — the field is transformed with the FFT; spatial derivatives become multiplication by the wavenumber `k`.
- **Exact integrating factor in time** — each step multiplies the spectrum by `exp(−ν · k² · dt)`, the analytic solution of the diffusion operator in Fourier space (`_ps_diffuse_once`).
- **Operator-split Newtonian cooling** — after diffusion, a relaxation term `−cooling · (u − ambient) · dt` pulls the field back toward ambient, keeping it bounded.
- **Spectral differentiation for "wind"** — the reported wind is the maximum slope of the field, `du/dx = ifft(i · k · fft(u))` (`_ps_spectral_gradient`).

The **peak of the computed field** becomes the temperature telemetry, so the NORMAL ⇄ CRITICAL cycle emerges from the physics rather than from a hardcoded pattern.

> **Scope note:** this is a 1-D diffusion *toy model*, not a real atmospheric weather simulation (true numerical weather prediction solves coupled 3-D fluid equations). The *numerical method*, however, is genuinely pseudo-spectral and demonstrates the technique correctly.

### Tunable physics parameters (top of `weather_agent.py`)

| Parameter | Meaning | Effect |
|---|---|---|
| `PS_NU` | thermal diffusivity | higher = heat spreads faster |
| `PS_COOLING` | Newtonian relaxation rate | higher = cools back to ambient faster |
| `PS_INJECT` | heat-pulse amplitude | higher = hotter CRITICAL peaks |
| `PS_AMBIENT` | baseline temperature | the value the field relaxes toward |
| `SAFETY_THRESHOLD_F` | CRITICAL cutoff | the governing threshold (85 °F) |

---

## 🧭 Step-Back: Problem → Method Mapping

Beyond just solving, the agent implements the **Step-Back** idea: *don't jump straight to a solver — first read the problem's features, then map problem → method to pick the right tool, then act.*

### Toy branch — problem features → action features

```
{day, rain}  ──map──▶  {breakfast, transport}     (the classic example)
```

In this project (`map_problem_to_action`):

```
{temperature}  ──map──▶  {mode, air_intake, max_threads}
   e.g. {96 °F} ──▶ {CRITICAL, INTERNAL_RECIRCULATION, 8}
```

### PDE branch — problem features → method features

```
{dimension, pde_type}  ──map──▶  {solver, #collocation_points}
   e.g. {2D, inverse}  ──▶  {PINN, 256}
```

This is `map_problem_to_method(features)`. It reads the problem's `dimension {1D,2D,3D}` and `pde_type {forward,inverse}` and returns a `solver {pseudo-spectral, PINN}` plus `#collocation_points {128, 256}`. The agent then **dispatches to the chosen solver** via `solve_with_method()`. This branch is what makes it a real problem→method framework rather than a fixed pipeline.

| Problem features | Mapped method | Solver run |
|---|---|---|
| `{1D, forward}` | `{pseudo-spectral, 128}` | `run_pseudo_spectral()` |
| `{2D, forward}` | `{pseudo-spectral, 128}` | `run_pseudo_spectral()` |
| `{2D, inverse}` | `{PINN, 256}` | `run_pinn()` |
| `{3D, forward}` | `{PINN, 256}` | `run_pinn()` |

Change `PROBLEM_FEATURES` at the top of `weather_agent.py` to see the map select a different method. The agent prints the mapping on startup:

```
🧭 [Step-Back] map: problem -> method
    problem features {1D, forward}  ->  method features {pseudo-spectral, 128}
```

> The mapping *rules* are illustrative (edit them freely). The pseudo-spectral branch is a complete, genuine solver; the PINN branch uses an adapter (next section).

---

## 🔌 Connecting a Real PINN Solver

When the mapping selects `PINN`, the agent calls `run_pinn()`, which uses an **adapter** that auto-detects a real PINN solver and falls back to a built-in stub if none is found — so the agent always runs.

**It looks for a real PINN in this order:**
1. An installed package named `pinn_solver` exposing `solve()`.
2. A local file `pinn_solver.py` next to `weather_agent.py` exposing `solve()`.
3. If neither is found → a built-in stub (clearly labeled in logs).

At startup, when the PINN branch is selected, it prints which backend is active:

```
    PINN backend: local:pinn_solver.py     (or package:..., or stub)
```

**The contract your `solve()` must honor:**

```python
solve(problem: dict) -> dict
#   input : {"dimension","pde_type","collocation_points","ambient"}
#   output: {"temp": float, "wind": float, "text": str}
```

A ready-to-fill template is provided as `pinn_solver_template.py` — rename it to `pinn_solver.py`, drop it next to `weather_agent.py`, and fill in the `solve()` body to call your PINN code. **No edits to `weather_agent.py` are needed** — the adapter finds it automatically. If your real solver raises an error mid-run, the agent logs it and falls back to the stub for that step only, so the loop never crashes.

To see the PINN branch fire, set at the top of `weather_agent.py`:
```python
PROBLEM_FEATURES = {"dimension": "2D", "pde_type": "inverse"}
```

---

## 🛠 Installation

Create a project folder (e.g. `p_agent`) and place all the files inside it.

### Prerequisites
- [Anaconda / Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Ollama](https://ollama.com/download) (for running Llama 3 locally)

### 1. Create the conda environment (Python 3.12)

```bash
conda create -n myenv312 python=3.12 -y
conda activate myenv312
```

### 2. Install Python dependencies

```bash
pip install flask requests numpy langchain-ollama langchain-core
```

<details>
<summary>Full package list</summary>

| Package | Purpose |
|---|---|
| `flask` | Web server + dashboard |
| `requests` | Agent → server communication |
| `numpy` | Pseudo-spectral FFT solver |
| `langchain-ollama` | Local Llama 3 chat interface |
| `langchain-core` | Prompt templating |

Chart.js and Three.js (r128) load in the browser via CDN — no install needed.
</details>

### 3. (If you hit LangChain version errors) Upgrade the LangChain ecosystem

```bash
pip install --upgrade "langchain>=0.3.0" "langchain-community>=0.3.0" "langchain-core>=0.3.0" langchain-ollama ollama
```

**Why this happens:** an older `langchain-community` (e.g. `0.2.6`) can act as a stubborn anchor that drags your install back to 2024-era versions. Upgrading to the `0.3.x` ecosystem clears those errors. Warnings about `pillow`, `gradio`, or `weaviate-client` are safe to ignore.

### 4. Verify the environment

Create `test_env.py`:

```python
import flask
import langchain_ollama
import requests
import numpy
print("🎉 Success! Your Python environment is clean and ready for the 3D demo!")
```

Run it:

```bash
python test_env.py
```

### 5. Install and start Ollama + pull a model

```bash
# Start the Ollama service (leave running)
ollama serve

# In another terminal, pull a model
ollama pull llama3
```

<!-- PASTE OLLAMA SCREENSHOT HERE (was: 6f675d5e-4942-49d6-8822-64f7a17076ea) -->

Verify:

```bash
ollama list
ollama run llama3      # optional interactive test; type /bye to exit
```

---

## 🚀 How to Start

You need **three things running**: Ollama, the server, and the agent.

**Terminal 1 — Ollama** (if not already running as a service)

```bash
ollama serve
```

**Terminal 2 — Dashboard server**

```bash
conda activate myenv312
python server.py
```

<!-- PASTE SERVER-START SCREENSHOT HERE (was: f8789857-cc32-4b52-b176-90c11b12b29d) -->

**Terminal 3 — Autonomous agent**

```bash
conda activate myenv312
python weather_agent.py
```

<!-- PASTE AGENT-START SCREENSHOT HERE (was: 1e66dd36-1381-407c-b90c-0812e71e8ed3) -->

Then open your browser to:

```
http://127.0.0.1:5000
```

<!-- PASTE DASHBOARD SCREENSHOT HERE (was: 34d18e0a-1b7f-4393-b094-28390adcedc0) -->

Within a few seconds the dashboard begins updating and the agent terminal prints `📡 [Web Server Ingest]` frames. The temperature line appears once several data points accumulate (give it 4–5 loops).

---

## 🛑 How to Stop

Press **`Ctrl + C`** in the server and agent terminals.

```bash
# In Terminal 2 and Terminal 3
Ctrl + C
```

To stop Ollama, press `Ctrl + C` in its terminal. To deactivate the environment: `conda deactivate`.

> **⚠️ Important:** `server.py` bakes its HTML in when it starts. After editing `server.py`, you **must fully stop it (`Ctrl+C`) and restart it** — a browser refresh alone keeps serving the old page. Hard-refresh the browser with `Ctrl+F5` afterward.

---

## 📂 Autonomous Theming (`sensor_input.json`)

This file is the agent's external trigger, handled by the independent watcher thread. The agent **watches it automatically** and re-themes the dashboard whenever it changes — no restart required.

### How it works
1. Create the file in the project folder:
   ```json
   {"status": "normal", "note": "all systems calm"}
   ```
   <!-- PASTE SENSOR-FILE SCREENSHOT HERE (was: 4d4743d9-8e89-43bd-9cf2-f04a4dee3a18) -->
   <!-- PASTE CALM-DASHBOARD SCREENSHOT HERE (was: 17be25b6-de85-41ab-9f91-a4ad905473a1) -->
2. The watcher thread reads the file every 1.5 s and compares its **content** (not modification time, so a save is never missed).
3. When the content changes, the text is mapped **deterministically** to a dark, readable background color (critical/alarm words → dark red; normal/calm words → dark blue).
4. The color is validated against a strict hex pattern (`#RRGGBB`) before being applied.
5. The dashboard background updates live (polled every 2 s).

### Try it
Save this to `sensor_input.json`:

```json
{"status": "critical", "note": "excessive heat alarm, grid overload"}
```

Within ~1.5 s the agent logs `📂 [File Watch]` and the background shifts to **dark red**.

<!-- PASTE FILE-WATCH LOG SCREENSHOT HERE (was: 692b47d3-60bb-4ae2-a70e-be58abb625bf) -->
<!-- PASTE CRITICAL-DASHBOARD SCREENSHOT HERE (was: 0c1885b9-813f-4909-b56f-a73b72d5d627) -->

Change it back to calm wording and the theme returns to **cool blue/teal**.

> The color is chosen by deterministic code, not the model — so it always applies, in both directions, every time. The model only ever produces a *value*, never a script that runs.

> **On the `⚠️ Model Disagreement` log:** this is **not** an error. It's the safety net working. If the small model wrongly suggests CRITICAL at, say, 73 °F, the code overrides it to NORMAL and logs the disagreement. Safe to ignore.

---

## 🧪 Testing

Test the solver and the Step-Back mapping in isolation — no server, no Ollama, no browser:

```bash
python test_ps.py
```

It (1) checks the mapping matches the reference example `{2D, inverse} → {PINN, 256}`, and (2) runs the pseudo-spectral solver for 18 steps, asserting the field moves and the state machine cycles between NORMAL and CRITICAL. Example output:

```
Step-Back  map: problem -> method
  {1D, forward}  ->  {pseudo-spectral, 128}
  {2D, inverse}  ->  {PINN, 256}
✅ PASS: {2D, inverse} -> {PINN, 256}

Pseudo-spectral solver — 18 steps  (threshold 85.0 °F)
   1 |     73.0 |   NORMAL | ######
   5 |    109.4 | CRITICAL | ########################
   8 |     80.4 |   NORMAL | ##########
✅ PASS: temperature both rises above and falls below the threshold (state cycles).
```

You can also test your PINN adapter alone:

```bash
python pinn_solver.py     # if you've created it from the template
```

---

## ⚡ Making It Faster

Each loop calls Llama, and on CPU a large model is slow. Since the LLM only *narrates* (code makes every real decision), you can safely use a much smaller, faster model:

```bash
ollama pull llama3.2:1b
```

Then change one line in `weather_agent.py`:

```python
llm = ChatOllama(
    model="llama3.2:1b",     # was "llama3"
    temperature=0.1,
    base_url="http://127.0.0.1:11434",
    format="json",
)
```

Smaller models are less accurate at reasoning, but that only affects the narration — you may see the `⚠️ Model Disagreement` line more often, which is fine by design.

---

## 📁 Project Structure

```
p_agent/
├── server.py                  # Flask dashboard (3D map, chart, status, theming)
├── weather_agent.py           # Autonomous AI agent (mapping + solver + self-heal loop)
├── server_config.py           # Auto-generated & overwritten by the agent (the self-healing target)
├── sensor_input.json          # Watched file → autonomous background theming
├── pinn_solver_template.py    # Template → rename to pinn_solver.py to plug in a real PINN
├── test_ps.py                 # Standalone tests: mapping + pseudo-spectral solver
├── test_env.py                # Quick environment sanity check
└── telemetry_logs/            # Timestamped config-healing backups (auto-created)
```

---

## 🔧 Configuration Notes

| Setting | Location | Default |
|---|---|---|
| Llama model | `weather_agent.py` → `ChatOllama(model=...)` | `llama3` |
| Ollama URL | `weather_agent.py` → `base_url` | `http://127.0.0.1:11434` |
| Server URL | `weather_agent.py` → `SERVER_URL` | `http://127.0.0.1:5000/update-status` |
| Watched file | `weather_agent.py` → `WATCH_FILE` | `sensor_input.json` |
| Problem features | `weather_agent.py` → `PROBLEM_FEATURES` | `{1D, forward}` |
| Loop interval | `weather_agent.py` → `time.sleep(5)` | 5 seconds |
| Safety threshold | `weather_agent.py` → `SAFETY_THRESHOLD_F` | `85.0 °F` |

---

## 🩺 Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| Blank page at `:5000` | `server.py` not running — start it in its own terminal. |
| Stuck on "Waiting for Agent heartbeat pulse..." | Agent can't reach the server — confirm `SERVER_URL` and that `server.py` is up. |
| Temperature line not showing | Wait 4–5 loops for points to accumulate; check `http://127.0.0.1:5000/api/history` returns data. |
| Line hard to see | On a dark-red (critical) background the orange line has low contrast — set `sensor_input.json` to calm wording. |
| Everything is slow | Use a smaller model — see [Making It Faster](#-making-it-faster). |
| `No module named 'numpy'` | `pip install numpy`. |
| `Could not find any JSON token blocks` | Ollama not returning JSON — ensure `format="json"` is set and `ollama serve` is running. |
| LangChain import / version errors | Run the `0.3.x` upgrade command in Installation step 3, then rerun `test_env.py`. |
| Chart / 3D not visible | Old server process — **restart `server.py`**, then `Ctrl+F5`. |
| Background never changes | Old server process (missing `/api/style`) — restart `server.py`. |
| `⚠️ Model Disagreement` in logs | Not an error — the safety net working. Code overrode a wrong model suggestion. |
| State stuck in one mode | Tune `PS_COOLING` (raise to cool faster) or `PS_INJECT` (raise for hotter peaks); verify with `test_ps.py`. |

---

## 🎓 For Students / Contest Notes

If you're presenting this at a science fair or AI contest, here's how to talk about it honestly and effectively:

**The one-sentence pitch:** *"A local AI agent that monitors a signal and, when it crosses a danger threshold, autonomously rewrites its own configuration file to put the system into a safety mode — no human involved."*

**What to emphasize (this is the real contribution):**
- The **autonomous action**: the agent edits `server_config.py` on its own and logs a backup. Show the file changing live.
- The **safety design**: the LLM reasons, but *code* makes every real decision — so a hallucinating model can't cause harm. This is a genuinely good engineering point.
- It runs **100% offline** on a laptop with a local model.

**What to be honest about:**
- It monitors a **simulated** signal, not real weather. Say so plainly. Judges respect this and penalize overclaiming.
- The pseudo-spectral model and the PINN mapping are **bonus math** — impressive, but only mention them if you can explain them. A simple demo you fully understand beats a complex one you can't defend.

**A good demo flow:** start both programs → show the dashboard → edit `sensor_input.json` to "critical" and watch the theme and state change → open `server_config.py` before and after to show the agent rewrote it → open the `telemetry_logs/` folder to show the automatic backups.

---

## 🔗 Related Work

- **[Physics-Informed Neural Networks (PINNs) for engineering forward-modeling](https://github.com/daviddata-cloud/physics-informed-nn-PINN-)** — embeds governing equations (Arrhenius kinetics, heat balance, IEEE-738) directly into the training loss, with multi-seed evaluation and an honest study of when physics helps and when it doesn't. Where this project *simulates* a governing equation with a classical pseudo-spectral solver and feeds it to an agent, that project *learns* solutions to governing equations by constraining a neural network with the physics — two complementary takes on physics-informed engineering computation. The PINN adapter here is designed to connect to that repo.

---

## 📜 License

MIT — free to use, modify, and distribute.
