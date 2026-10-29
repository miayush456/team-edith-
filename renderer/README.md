# SYNAPSE 2.5D Dual-Car Race Renderer

A modern, broadcast-grade **2.5D WebGL F1 TV Chase Viewer** built with Three.js for the SYNAPSE Autonomous Racing platform.

---

## 🌟 Visual & Environmental Features

- **🎬 F1 Grand Prix Hero Section & Cinematic Highlight Showcase:**
  - **Broadcast Hero Overlay:** Elegant neo-brutalist F1 pre-race launchpad displayed on initial load.
  - **Circuit Selection Grid:** Direct visual card selector for all official championship circuits (**Oval Speedway**, **Flowing Esses**, **Tight Loop**).
  - **Dual Racer Selectors:** Configurable head-to-head matchmaker for Car 1 (Cyan `#00f0ff`) and Car 2 (Papaya `#ff5500`), supporting both default built-in AI telemetry and instant custom replay JSON uploads.
  - **Hero BGM Audio Engine:** Dedicated background music player for the Hero Section. Automatically plays `renderer/assets/hero_bgm.mp3` (or `.wav`), with procedural ambient synth chord progression fallback when the file is not yet added. Fades out smoothly when launching the Grand Prix.
  - **6 Aesthetic Cinematic Racing Highlights & Cuts:**
    In the background of the hero section, the camera continuously tracks the cars across 6 aesthetic broadcast camera cuts with optical shutter flashes, repeating every 4.5 seconds:
    1. *Apex Curb Ground Cam:* Low-slung curb perspective ($Y = 0.42\text{m}$) clipping the corner rumble strip.
    2. *Telephoto Straight Zoom:* Compressed $24^\circ$ telephoto lens placed $36\text{m}$ down the straight looking directly head-on at the oncoming cars.
    3. *Trackside Lateral Tracking Pan:* Parallel side tracking shot with wheel rim and halo perspective ($Y = 1.65\text{m}$).
    4. *Helicopter Drone Overhead Sweep:* Wide-angle elevated drone perspective ($Y = 18.5\text{m}$) capturing the full circuit curves and grandstands.
    5. *Low Rear Diffuser & Rain LED Chase:* Close chase behind the carbon diffuser, titanium exhaust, and flashing red rain LED.
    6. *Spectator Grandstand POV:* Grandstand terrace perspective behind waving spectator flags.
  - **Instant Intro Re-opener:** Click `🎬 Intro` in the bottom broadcast bar anytime during racing to reopen the Hero Section showcase.
- **Authentic Sunlit F1 Grid & Enhanced 3D Realism:**
  - Bright, daytime Formula 1 atmosphere with warm sunlight (`3.2` intensity), vivid azure sky, soft horizon haze, and zero dark black voids.
  - **Procedural Asphalt Bump Map:** Micro-surface grain texture generating specular sheen and road grip realism.
  - **Rubberized Racing Line:** Dark Pirelli rubber grooved ribbon along apexes and braking zones.
  - **Glowing Carbon-Ceramic Brake Discs:** Front and rear brake rotors glow hot orange-red (`#ff2200`) under hard braking ($>0.12$ deceleration).
  - **Beveled Apex Rumble Curbs:** 3D extruded curbs with $0.065\text{m}$ height profile and beveled edges in alternating FIA red and white.
  - **Distance Braking Boards:** FIA standard $150\text{m}$, $100\text{m}$, and $50\text{m}$ marker boards placed on outer barrier walls before corner entry zones.
  - **FIA F1 Starting Grid Slots:** Authentic staggered grid positions (Slot 1 Pole on racing line, Slot 2 staggered 9m behind). Each slot features thick white front stop bars, fluorescent yellow centerline alignment guides, tire alignment chevrons, and bold asphalt-painted position decals (`1 POLE` and `2 P2`).
  - **Low-Poly 3D Audience & Grandstand Fans:**
    - Dozens of colorful low-poly spectators populating all grandstand tiers wearing authentic F1 team shirts (Ferrari Red, McLaren Papaya, Red Bull Blue, Mercedes Teal, Aston Green, Williams Blue, Cyan, Yellow, White).
    - Trackside spectator clusters standing along perimeter grass banks opposite the pit straight.
    - Animated fluttering team flags waving in the wind.
  - High-clearance start/finish gantry ($9.5\text{m}$ clearance) with pillars placed strictly **3.5m behind the outer safety barriers**.
  - All perimeter stadium floodlights and spectator grandstands are positioned **15–18m outside the outer barriers** on the open countryside turf — completely off the racing tarmac and runoffs.
- **Realistic Circuit Materials:**
  - **Asphalt Road Surface:** Authentic asphalt grey (`#2e333d`) with procedural grain bump map, specular sheen, sharp white boundary lines, dashed centerline, and checkered finish line.
  - **Runoff Traps:** Warm sandy golden gravel traps (`#d9b372`) and F1 Royal Blue painted runoff aprons (`#1e3a8a`).
  - **Apex Curbs:** Vibrant red (`#e10600`) and pure white (`#ffffff`) 3D beveled apex curbing.
  - **Safety Barriers:** Clean white/silver TecPro barrier walls (`#f8fafc`) with vibrant blue safety railings.
  - **Infield & Countryside:** Lush summer green grass turf (`#3d8b37`).
- **High-Detail F1 Car Models:**
  - **Procedural Livery & Sponsor Decals:** Crisp nose cone livery striping, carbon fiber edge borders, team branding banner (`SYNAPSE AI` / `SYNAPSE GP`), and bold racing number roundels (`#1` & `#2`).
  - **Aerodynamic Wing Mirrors:** Carbon fiber stalks and sculpted mirror pods with reflective chrome glass on cockpit flanks.
  - **Multi-Tier Front Wing:** Dihedral main plane, upper cascade flap, and curved outer endplate vortex canards.
  - **Underbody Rear Diffuser & Central Exhaust:** Carbon floor diffuser with 4 vertical aerodynamic strakes and central titanium exhaust tailpipe.
  - **Swan-Neck Rear Wing Mounts & DRS Actuator:** Dual curved carbon pylons mounting the rear wing from above, with DRS hydraulic actuator housing pod.
  - **Detailed Wheels & Brakes:** Directional center lock wheel nuts (Anodized Red on left, Anodized Blue on right), painted high-performance brake calipers, glowing carbon-ceramic brake discs, and Pirelli yellow sidewall striping.
  - **Radial Contact Shadow:** Smooth ambient occlusion ground contact shadow with radial gradient falloff under chassis.
  - 3D curved titanium Halo safety ring, driver helmet with iridescent gold/dark visor, and cockpit digital steering wheel dash.
  - **Car A (Racer Alpha):** Pearlescent Electric Cyan (`#00f0ff`) with gloss clearcoat and white aerodynamic flaps.
  - **Car B (Racer Beta):** Metallic Solar Papaya Orange (`#ff5500`) with gold aerodynamic flaps.
- **🎵 Real-Time Sound Engine & Custom Audio Asset Support:**
  - **Dual Engine Architecture:** Built-in procedural Web Audio API synthesizer + automatic custom WAV/MP3 loader from `renderer/assets/`.
  - **Audible Custom Audio Assets Supported:**
    - `renderer/assets/crowd_noise.mp3` (or `.wav`) — Real grandstand crowd cheering & murmur (audible baseline gain `0.52`, scaling up to `0.86` during high-speed action).
    - `renderer/assets/hero_bgm.mp3` (or `.wav`) — **Required file for the Hero Section background music!** (Smoothly fades in on intro and fades out upon race launch).
    - `renderer/assets/bg_sound.mp3` (or `.wav`) — TV broadcast background music / trackside ambient sound bed.
    - `renderer/assets/engine_sound.mp3` (or `.wav`) — Recorded F1 engine audio.
  - **Procedural Fallback (Zero Assets Required):** Multi-oscillator V6 turbo-hybrid engine roar mapped across 8 sequential gears (5,000–13,800 RPM), resonant grandstand crowd formant filter ($420\text{Hz}$) with speed/overtake cheering swells, and soft ambient synth broadcast music bed.
  - **Stereo Spatial Panning:** Car A and Car B have distinct spatial audio positioning.
  - **Audio Toggle & Shortcut:** `#btn-audio` toggle button on the bottom bar with keyboard shortcut (`M`). Auto-resumes on first user gesture.
- **🏁 Neo-Brutalist F1 TV Broadcast Graphics UI:**
  - **Broadcast Typography:** Authentic F1 broadcast typography using `Titillium Web` (italic bold caps), `Chakra Petch` HUD numerals, and `JetBrains Mono` telemetry readouts.
  - **F1 Steering Wheel 15-LED Shift Light HUD:** 15 sequential RPM LEDs (5 Green $\rightarrow$ 5 Red $\rightarrow$ 5 Blue) with redline flashing at $>92\%$ RPM.
  - **DRS Status Indicator:** Neon green `#00e676` DRS pill that illuminates on high-speed straights.
  - **Official F1 Timing Tower:** High-contrast carbon badge, driver number (`#1`, `#2`), 3-letter codes (`ALF`, `BET`), tire compound badges (`[S]`, `[M]`), gap intervals, and live mini-sector timing dots.
  - **Provisional Classification Victory Modal:** Official F1 GP post-race classification screen with race time, fastest lap chip, podium leaderboard, and instant track switching.
- **Dynamic X-Ray Overlap Transparency & Glowing Wireframe Matrix:**
  - If two cars come within $2.8\text{m}$ of each other (or in Raw Telemetry mode where they occupy the exact same coordinate):
    - Both car bodies smoothly become **semi-transparent** (~45% opacity).
    - Glowing **holographic wireframe matrices** illuminate around both cars (Cyan for Car A, Solar Gold for Car B), allowing viewers to clearly see both cars simultaneously without clipping!
    - The timing tower displays an animated `⚡ X-RAY OVERLAP MODE` badge.
    - When cars separate ($> 2.8\text{m}$), they smoothly transition back to 100% solid paint with no wireframe.
- **Dynamic Chase Camera System:**
  - `[1] Battle Cam`: Smoothly tracks the centroid between both cars, zooming in during close wheel-to-wheel battles and pulling back when they separate.
  - `[2] Car A Chase`: Locked follow camera behind Racer Alpha.
  - `[3] Car B Chase`: Locked follow camera behind Racer Beta.
  - `[4] Cockpit Cam`: First-person nose/halo camera for high-speed perspective.
  - `[5] Top Drone Cam`: Overhead tactical overview of the full circuit.
  - `[6] Side Trackside Cam`: Dynamic low-slung TV broadcast side-profile camera tracking the cars along the tarmac.
- **Clean 2-Row Bottom Broadcast Control Bar:**
  - Row 1: Full-width scrubbable progress timeline with high-precision timecode indicators (`00:00.000` elapsed and total duration).
  - Row 2: Four distinct control groups:
    - *Transport:* Restart `⏮`, Step Back `-1s`, Play/Pause `▶`/`⏸`, Step Forward `+1s`.
    - *Speed:* Preset speed pills `0.5×`, `1×`, `2×`, `4×`.
    - *Cameras:* `Battle [1]`, `Car A [2]`, `Car B [3]`, `Cockpit [4]`, `Top [5]`.
    - *Auxiliary:* `Lanes: Dual (L)`, `🏁 Track Select` dropdown, and `📂 Load` custom replay button.
- **Winner Victory Screen & Podium Modal:**
  - When the race reaches the checkered flag, an official F1 victory podium modal pops up displaying:
    - 🏆 Official P1 Winner banner and winning participant name
    - Official race completion time (e.g. `00:28.767`)
    - Detailed podium leaderboard table with P1 and P2 ranks, team color indicators, and finish gap / DNF status
    - Interactive action buttons: `⏮ Replay Race` and `🏎️ Next Track` to cycle through circuits!
- **Instant Circuit Switching:**
  - Switch between circuits (`🏁 Oval`, `🏁 Flowing`, `🏁 Tight Loop`) instantly using the track selector in the bottom control bar or the victory screen without restarting the server.
- **Top-Right Radar Minimap:** Live circuit radar with track contour, glowing car blips with directional heading needles, and dynamic camera field-of-view cone.
- **F1 Live Timing Tower & Telemetry:**
  - Real-time P1/P2 leaderboard with interval gaps and leader tags.
  - Dynamic sector indicator (`SEC 1`, `SEC 2`, `SEC 3`) and lap tracker (`LAP 1/1`).
  - Real-time digital speedometer (km/h), throttle & brake bars, steering wheel angle in degrees (e.g. `+14.2°`), F1 gear indicator (`G1`–`G8`), and lateral G-force meter (e.g. `1.8 G`).
  - Floating broadcast alerts for track limits (`OFF TRACK`), collisions (`BARRIER CONTACT`), lead changes (`OVERTAKE`), and checkered flag.

---

## Ownership and runtime boundary

`renderer/` is the presentation layer for the SYNAPSE project. Python owns
observations, ONNX actions, fixed-step physics, deadlines, events, termination,
and results. The renderer reads validated replay/live packets, interpolates
only received poses, and never sends actions back to the simulator. Keep
renderer changes coordinated with the renderer-owning team; participant
orchestration belongs in `tools/participant_ui.py` and
`tools/run_live_onnx.py`.

## 🚀 Quick Start Guide: How to Run the Renderer

Run commands from the repository root. Use the project Python when available:

### Step 1: Start the local web server

```bash
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory renderer
```

### Step 2: Open in Your Browser

Open Chrome, Edge, or Firefox and navigate to:

👉 **[http://127.0.0.1:8765](http://127.0.0.1:8765)**

*(The built-in sample replays load automatically on launch!)*

### Live Python race mode

The renderer also has a read-only live adapter. Start the Python live runner
from the repository root, then open the renderer with the printed WebSocket
URL:

```bash
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory renderer
.venv/bin/python tools/run_live_onnx.py \
  --entrant racer-a submissions/racer-a \
  --entrant racer-b submissions/racer-b \
  --track tracks/train/monaco.json \
  --config config/simulation.json \
  --output-dir artifacts/live/oval \
  --port 8767 \
  --telemetry-stdout
```

Open `http://127.0.0.1:8765/?live=ws://127.0.0.1:8767/live/v1`, or paste that
WebSocket URL into **Connect Live**. Python owns every action, physics step,
event, and result. The browser only receives state packets, interpolates
received poses for presentation, and shows `STALE` if packets stop arriving.
Replay controls are disabled in live mode. The live stream is not a replay and
does not send actions back to Python.

### Participant dashboard flow

For a participant-style run, start the dashboard in a second terminal:

```bash
.venv/bin/python tools/participant_ui.py --port 8766
```

Open **http://127.0.0.1:8766/** and use **03 · Grid control / live heat** to
choose two ONNX artifact directories, one track/config, a new output directory,
and the live policy. The dashboard starts and stops the Python race process,
shows per-entrant latency/deadline/failure telemetry, and displays shared
process-tree RSS/peak RSS/PID/thread telemetry. It also keeps the bounded
`LIVE_TELEMETRY` child output in the terminal panel. The default dashboard race
WebSocket port is `8767`, while the renderer HTTP server remains on `8765`.

The dashboard's default `strict_scored` profile marks late or failed inference
as `dnf_inference`. `trusted_live` is a local demonstration profile that may
hold the previous action for one interval; it is not a scored event rule.

---

## 🏎️ How to Generate Dual-Car Replays for Any Track

You can generate fresh head-to-head evaluation replays for any circuit using `tools/export_replay.py`:

### A. Synthetic Oval Track
```bash
.venv/bin/python tools/export_replay.py --scripted --track tracks/train/monaco.json --config config/simulation.json --participant-id racer-alpha --output artifacts/replay/racer_a.json
.venv/bin/python tools/export_replay.py --scripted --track tracks/train/monaco.json --config config/simulation.json --participant-id racer-beta --steps 650 --output artifacts/replay/racer_b.json
```

### B. Synthetic Flowing Track
```bash
.venv/bin/python tools/export_replay.py --scripted --track tracks/train/spa.json --config config/simulation.json --participant-id spa-alpha --output artifacts/replay/spa_a.json
.venv/bin/python tools/export_replay.py --scripted --track tracks/train/spa.json --config config/simulation.json --participant-id spa-beta --steps 500 --output artifacts/replay/spa_b.json
```

### C. Synthetic Tight Loop Track
```bash
.venv/bin/python tools/export_replay.py --scripted --track tracks/train/monaco.json --config config/simulation.json --participant-id monaco-alpha --output artifacts/replay/monaco_a.json
.venv/bin/python tools/export_replay.py --scripted --track tracks/train/monaco.json --config config/simulation.json --participant-id monaco-beta --steps 450 --output artifacts/replay/monaco_b.json
```

---

## 📦 How to Bundle Standalone Offline HTML Files (Phase 7)

To generate self-contained, single-file HTML replays that open anywhere on any device without running a server:

```bash
# Bundle Oval Track Race
.venv/bin/python renderer/build_replay.py --racer-a artifacts/replay/racer_a.json --racer-b artifacts/replay/racer_b.json --output artifacts/replay/battle_oval.html

# Bundle Flowing Track Race
.venv/bin/python renderer/build_replay.py --racer-a artifacts/replay/flowing_a.json --racer-b artifacts/replay/flowing_b.json --output artifacts/replay/battle_flowing.html

# Bundle Tight Loop Race
.venv/bin/python renderer/build_replay.py --racer-a artifacts/replay/tight_a.json --racer-b artifacts/replay/tight_b.json --output artifacts/replay/battle_tight.html
```

You can double-click any generated `.html` file in `artifacts/replay/` to run it offline!

---

## 🎮 Keyboard Shortcuts & Controls Reference

| Control | Shortcut | Description |
|---|---|---|
| **Play / Pause** | `Space` | Toggle playback |
| **Sound Mute / Unmute** | `Key M` | Toggle F1 engine, crowd, and TV broadcast audio |
| **Reset to Start** | `Key R` | Rewind simulation to Tick 0 |
| **Lane Separation** | `Key L` | Toggle Side-by-Side Dual Lane vs Raw Telemetry (test overlap wireframe!) |
| **Battle Cam** | `1` | Default broadcast chase camera tracking both cars |
| **Car A Chase** | `2` | Tight follow camera behind Racer Alpha |
| **Car B Chase** | `3` | Tight follow camera behind Racer Beta |
| **Nose / Cockpit** | `4` | First-person nose/halo perspective |
| **Top Drone View** | `5` | High-altitude tactical overview of the circuit |
| **Side Trackside View**| `6` | Low-slung dynamic TV broadcast side-profile tracking cam |
| **Step 1s Back** | `[` | Step backwards by 1.0 second |
| **Step 1s Forward**| `]` | Step forward by 1.0 second |
| **Step 1 Frame** | `←` / `→` | Step backwards / forwards by 1 frame (1/30s) |
| **Playback Speed** | `↑` / `↓` | Increase / decrease playback speed (0.5×, 1×, 2×, 4×) |
| **Load Replays** | `📂 Load` | Open modal to select custom replay JSON files |
