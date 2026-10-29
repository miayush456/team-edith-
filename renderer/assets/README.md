# F1 Broadcast Audio Assets Directory

You can place your custom audio files (`.wav` or `.mp3`) directly into this directory (`renderer/assets/`). The renderer will automatically detect and load them. If any file is not present, the renderer uses its built-in real-time Web Audio API procedural synthesizer as a zero-dependency fallback.

---

## 🎧 Supported Filenames & Formats

| Audio Component | Recommended Filename | Alternative Filename | Description |
|---|---|---|---|
| **Hero Section BGM (New!)** | `hero_bgm.mp3` | `hero_bgm.wav` | High-energy / atmospheric F1 TV broadcast pre-race intro theme played in the background of the Hero showcase screen. |
| **Audience / Crowd Noise** | `crowd_noise.mp3` | `crowd_noise.wav` | Ambient trackside grandstand crowd cheering, murmur, and applause. Played at an audible baseline with dynamic high-speed swells. |
| **Broadcast Background Sound / Music** | `bg_sound.mp3` | `bg_sound.wav` | Ambient in-race TV broadcast music bed or atmospheric circuit audio. |
| **F1 Engine Sound** *(Optional)* | `engine_sound.wav` | `engine_sound.mp3` | Real recorded F1 V6 turbo-hybrid engine acceleration/flyby loop. |

---

## 📌 Instructions

1. Copy your `.wav` or `.mp3` files into `d:\synapse_shit\Synapse\renderer\assets\`.
2. Name the files exactly as shown above:
   - For Hero Section BGM: **`hero_bgm.mp3`**
   - For Crowd Noise: **`crowd_noise.mp3`**
   - For Race In-Game Music: **`bg_sound.mp3`**
3. Reload the browser page (`http://127.0.0.1:8765/`).
4. The renderer will automatically detect, decode, and play your custom audio files seamlessly!
