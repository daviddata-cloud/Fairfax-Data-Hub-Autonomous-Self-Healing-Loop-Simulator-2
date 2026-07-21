from flask import Flask, request, jsonify, render_template_string
import time
import random

app = Flask(__name__)

# Core global data status cache (drives the left status panel — Fairfax)
dashboard_status = {
    "is_alert": False,
    "location": "Fairfax, VA",
    "temperature": 70,
    "wind_speed": 5,
    "condition": "Waiting for Agent heartbeat pulse...",
    "current_code": "# System initializing. Streaming telemetry data...",
    "bg_color": "#0b0f19",
    "method": "— (awaiting agent)"
}

historical_history = []

# --- 6 Virginia locations for the 3D chart ---
# Fairfax is fed by the real autonomous agent; the other 5 are simulated
# so the 3D scene stays alive during a demo.
va_locations = {
    "Fairfax":        {"temp": 70, "wind": 5},
    "Arlington":      {"temp": 71, "wind": 6},
    "Richmond":       {"temp": 78, "wind": 8},
    "Virginia Beach": {"temp": 74, "wind": 12},
    "Roanoke":        {"temp": 69, "wind": 4},
    "Charlottesville":{"temp": 73, "wind": 5},
}

def drift(value, low, high, step):
    """Nudge a simulated value up/down a little, clamped to a range."""
    value += random.uniform(-step, step)
    return max(low, min(high, value))

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fairfax Autonomous 3D Control Center</title>
    <!-- Three.js (r128) -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, system-ui, sans-serif; background: #0b0f19; color: #e2e8f0; margin: 0; padding: 20px; display: flex; flex-direction: column; gap: 20px; font-size: 18px; }
        .row { display: flex; gap: 20px; width: 100%; }
        .panel { width: 45%; padding: 25px; background: #111827; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); border: 1px solid #1f2937; }
        .visual-container { width: 50%; display: flex; flex-direction: column; background: #111827; border-radius: 12px; padding: 15px; border: 1px solid #1f2937; }
        #canvas3d { width: 100%; height: 420px; border-radius: 12px; background: #030712; border: 1px solid #1f2937; cursor: grab; }
        #canvas3d:active { cursor: grabbing; }
        .chart-container { width: 100%; background: #111827; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); box-sizing: border-box; border: 1px solid #1f2937; }
        /* Fixed-height wrapper so Chart.js has a box to draw into */
        .chart-canvas-wrap { position: relative; width: 100%; height: 520px; }
        .badge { display: inline-block; padding: 8px 18px; border-radius: 20px; font-weight: bold; font-size: 16px; margin-top: 10px; }
        .danger { background: #dc2626; color: white; animation: pulse 2s infinite; }
        .success { background: #16a34a; color: white; }
        h2 { font-size: 26px; }
        h3 { font-size: 21px; }
        pre { background: #030712; padding: 15px; border-radius: 8px; color: #38bdf8; font-family: monospace; border: 1px solid #1f2937; overflow-x: auto; font-size: 16px; }
        .hint { font-size: 14px; color: #6b7280; margin-top: 8px; text-align: center; }
        @keyframes pulse { 0% { opacity: 0.7; } 50% { opacity: 1; } 100% { opacity: 0.7; } }
    </style>
</head>
<body>

    <div class="row">
        <!-- Infrastructure Status Tracking Information Grid -->
        <div class="panel">
            <h2>📊 Fairfax Data Hub (Autonomous Self-Healing Loop)</h2>
            <p><strong>Geographic Location Node:</strong> {{ data.location }}</p>
            <p><strong>Datacenter Core Temperature:</strong> <span id="temp-val">{{ data.temperature }}</span> °F</p>
            <p><strong>External Wind Velocity:</strong> <span id="wind-val">{{ data.wind_speed }}</span> mph</p>
            <p><strong>Atmospheric Metric Profile:</strong> {{ data.condition }}</p>
            <p><strong>Step-Back Method (problem&nbsp;→&nbsp;method):</strong> {{ data.method }}</p>

            {% if data.is_alert %}
                <span class="badge danger">🚨 CRITICAL AIRFLOW: Emergency Isolation Active</span>
            {% else %}
                <span class="badge success">🟢 ENVIRONMENT OPTIMAL: High Performance Profile</span>
            {% endif %}

            <h3>⚙️ Active Production Script Config (`server_config.py`):</h3>
            <pre><code>{{ data.current_code }}</code></pre>
        </div>

        <!-- 3D Engine: 6 Virginia locations -->
        <div class="visual-container">
            <h3>🌎 Virginia Regional Telemetry — 3D (6 Locations)</h3>
            <div id="canvas3d"></div>
            <div class="hint">Drag to rotate · auto-spins · bar height = temp · red = &gt;85°F</div>
        </div>
    </div>

    <!-- Analytics Line Chart -->
    <div class="chart-container">
        <h3>📈 Live Analytical Infrastructure Streaming Timelines (Fairfax)</h3>
        <div class="chart-canvas-wrap">
            <canvas id="streamingChart"></canvas>
        </div>
    </div>

    <script>
        // ================= CHART.JS =================
        const ctx = document.getElementById('streamingChart').getContext('2d');
        const telemetryChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Ambient Temperature (°F)', data: [], borderColor: '#ff4500', backgroundColor: 'rgba(255, 69, 0, 0.05)', tension: 0.25, yAxisID: 'y' },
                    { label: 'Wind Velocity (mph)', data: [], borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.05)', tension: 0.25, yAxisID: 'y1' }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y:  { type: 'linear', position: 'left',  grid: { color: '#1f2937' }, ticks: { color: '#9ca3af', font: { size: 15 } } },
                    y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#9ca3af', font: { size: 15 } } },
                    x:  { ticks: { color: '#9ca3af', font: { size: 14 } } }
                },
                plugins: { legend: { labels: { color: '#e2e8f0', font: { size: 16 } } } }
            }
        });

        setInterval(() => {
            fetch('/api/history')
                .then(res => res.json())
                .then(historyData => {
                    if (!historyData || historyData.length === 0) return;
                    const latest = historyData[historyData.length - 1];
                    document.getElementById('temp-val').textContent = latest.temp;
                    document.getElementById('wind-val').textContent = latest.wind;
                    telemetryChart.data.labels = historyData.map(d => d.time);
                    telemetryChart.data.datasets[0].data = historyData.map(d => d.temp);
                    telemetryChart.data.datasets[1].data = historyData.map(d => d.wind);
                    telemetryChart.update('none');
                }).catch(err => console.error("History pipeline drop:", err));
        }, 1500);

        // ================= THREE.JS: 6-LOCATION 3D BAR CHART =================
        const container = document.getElementById('canvas3d');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        container.appendChild(renderer.domElement);

        // Lighting (needed for solid Lambert bars)
        scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        const dir = new THREE.DirectionalLight(0xffffff, 0.8);
        dir.position.set(5, 10, 7);
        scene.add(dir);

        // A group holds everything so we can rotate the whole chart
        const chartGroup = new THREE.Group();
        scene.add(chartGroup);

        // Floor grid
        const grid = new THREE.GridHelper(14, 14, 0x1f2937, 0x1f2937);
        chartGroup.add(grid);

        const TEAL = 0x00ffcc, RED = 0xff4500;
        const HScale = 0.08; // temp -> bar height

        // Build a text label sprite from a canvas texture
        function makeLabelSprite(text) {
            const c = document.createElement('canvas');
            c.width = 256; c.height = 64;
            const g = c.getContext('2d');
            g.fillStyle = '#e2e8f0';
            g.font = 'bold 26px sans-serif';
            g.textAlign = 'center';
            g.textBaseline = 'middle';
            g.fillText(text, 128, 32);
            const tex = new THREE.CanvasTexture(c);
            const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
            spr.scale.set(3, 0.75, 1);
            return spr;
        }
        function updateLabelSprite(spr, text) {
            const c = spr.material.map.image;
            const g = c.getContext('2d');
            g.clearRect(0, 0, c.width, c.height);
            g.fillStyle = '#e2e8f0';
            g.font = 'bold 26px sans-serif';
            g.textAlign = 'center';
            g.textBaseline = 'middle';
            g.fillText(text, 128, 32);
            spr.material.map.needsUpdate = true;
        }

        // Bars object keyed by city name
        const bars = {}; // name -> { mesh, label }
        const cityOrder = ["Fairfax","Arlington","Richmond","Virginia Beach","Roanoke","Charlottesville"];
        // Approx geographic layout of the 6 VA cities (map-like spread)
        // x = east-west (east +), z = north-south (north = back / -z)
        const cityCoords = {
            "Fairfax":         { x:  2.6, z: -5.2 },
            "Arlington":       { x:  3.6, z: -5.6 },
            "Richmond":        { x:  2.2, z:  1.7 },
            "Virginia Beach":  { x:  6.4, z:  5.2 },
            "Roanoke":         { x: -6.4, z:  3.1 },
            "Charlottesville": { x: -2.0, z: -0.9 }
        };

        // --- 3D NETWORK FRAME: connect every city to every other, at the BAR TOPS ---
        const netMat = new THREE.LineBasicMaterial({ color: 0xd1d5db, transparent: true, opacity: 0.55 });
        const netLines = []; // { line, a, b } so we can restretch to live bar-top heights
        for (let a = 0; a < cityOrder.length; a++) {
            for (let b = a + 1; b < cityOrder.length; b++) {
                const pa = cityCoords[cityOrder[a]];
                const pb = cityCoords[cityOrder[b]];
                const g = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(pa.x, 0.05, pa.z),
                    new THREE.Vector3(pb.x, 0.05, pb.z)
                ]);
                const line = new THREE.Line(g, netMat);
                chartGroup.add(line);
                netLines.push({ line, a: cityOrder[a], b: cityOrder[b] });
            }
        }

        cityOrder.forEach((name) => {
            const p = cityCoords[name];

            const geo = new THREE.BoxGeometry(1.0, 1, 1.0); // unit height, we scale Y
            const mat = new THREE.MeshLambertMaterial({ color: TEAL });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.position.set(p.x, 0, p.z);
            chartGroup.add(mesh);

            // glowing node marker sitting on the grid at each location
            const node = new THREE.Mesh(
                new THREE.SphereGeometry(0.18, 12, 12),
                new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
            );
            node.position.set(p.x, 0.05, p.z);
            chartGroup.add(node);

            const label = makeLabelSprite(name);
            label.position.set(p.x, 0, p.z);
            chartGroup.add(label);

            bars[name] = { mesh, label, x: p.x, z: p.z, shortName: name };
        });

        function applyLocations(locs) {
            locs.forEach(loc => {
                const b = bars[loc.name];
                if (!b) return;
                const h = Math.max(0.5, loc.temp * HScale);
                b.mesh.scale.y = h;
                b.mesh.position.y = h / 2;          // sit base on the grid
                b.mesh.material.color.setHex(loc.temp > 85 ? RED : TEAL);
                b.top = h;                          // remember bar-top height for the net
                b.label.position.y = h + 0.9;       // float above the bar
                updateLabelSprite(b.label, loc.name + "  " + loc.temp + "°F");
            });

            // Restretch the net so every line connects the two bar TOPS
            netLines.forEach(n => {
                const ba = bars[n.a], bb = bars[n.b];
                const ya = (ba.top || 0.5), yb = (bb.top || 0.5);
                n.line.geometry.setFromPoints([
                    new THREE.Vector3(ba.x, ya, ba.z),
                    new THREE.Vector3(bb.x, yb, bb.z)
                ]);
                n.line.geometry.attributes.position.needsUpdate = true;
                n.line.geometry.computeBoundingSphere(); // keep it from being culled after moving up
            });
        }

        // Camera framing — high, near top-down so the map scatter reads clearly
        camera.position.set(0, 16, 8);
        camera.lookAt(0, 0, 0);

        // Manual drag-to-rotate + gentle auto-spin (no OrbitControls dependency)
        let autoSpin = true, dragging = false, lastX = 0, targetRotY = 0.4;
        container.addEventListener('mousedown', e => { dragging = true; autoSpin = false; lastX = e.clientX; });
        window.addEventListener('mouseup', () => { dragging = false; });
        window.addEventListener('mousemove', e => {
            if (!dragging) return;
            targetRotY += (e.clientX - lastX) * 0.01;
            lastX = e.clientX;
        });

        function animate() {
            requestAnimationFrame(animate);
            if (autoSpin) targetRotY += 0.003;
            chartGroup.rotation.y += (targetRotY - chartGroup.rotation.y) * 0.1;
            renderer.render(scene, camera);
        }
        animate();

        // Poll the 6-location endpoint
        function refreshLocations() {
            fetch('/api/locations')
                .then(res => res.json())
                .then(locs => applyLocations(locs))
                .catch(err => console.error("Locations pipeline drop:", err));
        }
        refreshLocations();
        setInterval(refreshLocations, 2000);

        // Poll the autonomously-chosen background color and apply it live
        setInterval(() => {
            fetch('/api/style')
                .then(r => r.json())
                .then(s => { if (s.bg_color) document.body.style.background = s.bg_color; })
                .catch(err => console.error("Style pipeline drop:", err));
        }, 2000);

        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, data=dashboard_status)

@app.route('/api/history')
def get_history_stream():
    return jsonify(historical_history)

@app.route('/api/style')
def get_style():
    """Serves the current dashboard background color (set autonomously by the agent)."""
    return jsonify({"bg_color": dashboard_status.get("bg_color", "#0b0f19")})

@app.route('/api/locations')
def get_locations():
    """Returns current telemetry for all 6 VA locations.
    Fairfax reflects real agent data; the other 5 drift a little each poll."""
    for name, loc in va_locations.items():
        if name == "Fairfax":
            loc["temp"] = int(dashboard_status["temperature"])
            loc["wind"] = int(dashboard_status["wind_speed"])
        else:
            loc["temp"] = round(drift(loc["temp"], 55, 100, 1.5))
            loc["wind"] = round(drift(loc["wind"], 0, 30, 1.0))
    return jsonify([{"name": n, "temp": v["temp"], "wind": v["wind"]} for n, v in va_locations.items()])

@app.route('/update-status', methods=['POST'])
def update_status():
    global dashboard_status, historical_history
    data = request.json

    dashboard_status.update(data)

    # Keep the Fairfax 3D bar in sync with the real agent
    va_locations["Fairfax"]["temp"] = int(data.get("temperature", 70))
    va_locations["Fairfax"]["wind"] = int(data.get("wind_speed", 5))

    timestamp = time.strftime("%H:%M:%S")
    historical_history.append({
        "time": timestamp,
        "temp": int(data.get("temperature", 70)),
        "wind": int(data.get("wind_speed", 5))
    })
    if len(historical_history) > 15:
        historical_history.pop(0)

    print(f"📡 [Web Server Ingest] Frame stored at {timestamp}")
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)