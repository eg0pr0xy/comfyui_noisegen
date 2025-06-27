# 🎵 ComfyUI NoiseGen

> **"Noise is the most impolite music."** - Merzbow

A comprehensive ComfyUI node pack for generating various types of audio noise with professional-grade quality and extensive customization options. Perfect for harsh noise, experimental music, scientific testing, and audio production.

---

## 🚀 Features

### 🔊 **Noise Arsenal**

#### **Basic Types**
- **🌀 White Noise** - Pure static chaos, flat frequency spectrum
- **🌸 Pink Noise** - 1/f natural fury, balanced frequency response  
- **🍫 Brown Noise** - Deep rumbling destruction, 1/f² frequency slope

#### **Advanced Types**
- **💙 Blue Noise** - High-frequency razor cuts, +3dB/octave slope
- **💜 Violet Noise** - Ultrasonic warfare, +6dB/octave slope
- **🌿 Perlin Noise** - Organic texture synthesis with natural variations
- **🎯 Band-Limited Noise** - Frequency-targeted strikes with precise filtering
- **⚡ ChaosNoiseMix** - Merzbow-style absolute devastation with 11 mixing modes

---

## 📦 Installation

### **Method 1: ComfyUI Manager** ⭐ *Recommended*
1. Open ComfyUI Manager
2. Search for `"ComfyUI-NoiseGen"`
3. Click Install & restart ComfyUI

### **Method 2: Manual Installation**
```bash
cd /path/to/comfyui/custom_nodes
git clone https://github.com/eg0pr0xy/noisegen.git
cd noisegen
pip install -r requirements.txt
# Restart ComfyUI
```

---

## 🎛️ Technical Specifications

| **Parameter** | **Range** | **Description** |
|---------------|-----------|-----------------|
| **Duration** | 0.1s → 300s | Audio length control |
| **Sample Rate** | 8kHz → 96kHz | Quality settings (8k, 16k, 22k, 44.1k, 48k, 96k) |
| **Amplitude** | 0.0 → 2.0 | Volume/loudness control |
| **Channels** | 1 → 8 | Mono to surround sound |
| **Stereo Mode** | 3 modes | Independent, correlated, decorrelated |
| **Seed** | 0 → 2147483647 | Reproducible random generation |

### **ChaosNoiseMix Parameters**
- **Mix Modes**: `add` • `multiply` • `xor` • `modulo` • `subtract` • `max` • `min` • `ring_mod` • `am_mod` • `fm_mod` • `chaos`
- **Distortion**: Multi-stage clipping + asymmetric processing
- **Bit Crush**: 1-16 bits digital destruction
- **Feedback**: Metallic resonance delay
- **Ring Frequency**: 1Hz → 5kHz modulation

---

## 🎵 Node Categories

### **🎵 NoiseGen** (Universal)
Single node that can generate all noise types with dynamic parameters

### **🎵 NoiseGen/Basic**
- White Noise Generator
- Pink Noise Generator  
- Brown Noise Generator

### **🎵 NoiseGen/Advanced**
- Blue Noise Generator
- Violet Noise Generator
- Perlin Noise Generator (frequency + octave control)
- Band-Limited Noise Generator (frequency filtering)
- **ChaosNoiseMix** (extreme mixing & processing)

### **🎵 NoiseGen/Utils**
- Audio Save Node (WAV/FLAC/MP3 export)

---

## 🎯 Applications

### **🎼 Music Production**
- **Harsh Noise Walls** → white/brown + chaos mixing
- **Japanese Noise** → merzbow-style chaos chains
- **Power Electronics** → extreme bit crushing + distortion
- **Industrial Ambience** → multi-layer perlin + band filtering
- **Drone/Dark Ambient** → brown noise + slow modulation
- **Experimental Music** → all parameters pushed to extremes

### **🔬 Scientific/Technical**
- **Audio Testing/Masking** → scientific precision meets chaos
- **Psychoacoustic Research** → controlled frequency manipulation
- **Equipment Calibration** → reference signals
- **Signal Processing Research** → various noise characteristics

---

## 📊 Frequency Analysis

| **Type** | **Spectrum** | **Characteristics** |
|----------|--------------|---------------------|
| **White** | `████████████████` | Equal energy across all frequencies |
| **Pink** | `████████████▓▓▓▓` | Equal energy per octave (1/f) |
| **Brown** | `██████▓▓▓▓░░░░░░` | Deep bass emphasis (1/f²) |
| **Blue** | `░░░░▓▓▓▓████████` | High frequency emphasis (+3dB/octave) |
| **Violet** | `░░░░░░▓▓████████` | Ultra-high frequency (+6dB/octave) |

---

## 🏗️ Project Structure

```
noisegen/
├── __init__.py              # Package init + ComfyUI hooks
├── noise_nodes.py           # Core node implementations
├── audio_utils.py           # DSP algorithms + utilities
├── requirements.txt         # Dependencies
├── pyproject.toml           # Modern Python packaging
├── examples/                # Workflow demonstrations
│   ├── basic_white_noise.json
│   ├── stereo_ambient_soundscape.json
│   ├── japanese_noise_merzbow.json
│   └── audio_test_suite.json
├── web/index.html           # Documentation interface
├── test_nodes.py            # Validation suite
├── CHANGELOG.md             # Version history
├── LICENSE                  # MIT License
└── README.md                # This manual
```

---

## 🛠️ Audio Engine

- **Format**: ComfyUI native AUDIO (32-bit float)
- **Algorithms**: Professional-grade DSP
- **Optimization**: NumPy vectorization + memory efficiency
- **Anti-aliasing**: Proper frequency domain shaping
- **Error Handling**: Bulletproof parameter validation
- **Architecture**: Clean separation of concerns

---

## 📝 Examples

### **Basic White Noise**
1. Add `🎵 White Noise` node
2. Set duration: `10.0` seconds
3. Set amplitude: `0.8`
4. Connect to audio output

### **Merzbow-Style Chaos**
1. Add multiple noise generators (white, brown, perlin)
2. Route through `🎵 ChaosNoiseMix` nodes
3. Use extreme settings: `chaos` mode, max distortion, low bit depth
4. Chain multiple ChaosNoiseMix nodes for devastation

### **Scientific Testing**
1. Use `🎵 Noise Generator` with specific type
2. Set precise parameters (sample rate, duration, amplitude)
3. Use `seed` for reproducible results
4. Export with `🎵 Audio Save`

---

## 📞 Contact & Support

**Repository**: https://github.com/eg0pr0xy/noisegen.git  
**License**: MIT (freedom to modify and redistribute)  
**Support**: Issues + pull requests welcomed  

### **Contributing**
- 🔧 **Contribute**: Help expand the noise arsenal
- 🐛 **Report Bugs**: Ensure stability in chaos  
- 📚 **Share Workflows**: Spread the knowledge

---

<div align="center">

**Generate Chaos. Destroy Silence. Create Music.**

*Made with ❤️ for the experimental music community*

</div> 