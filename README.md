# 💀 COMFYUI NOISE GENERATOR 💀

> **"Noise is the most impolite music."** - Merzbow

A comprehensive ComfyUI node pack for generating various types of audio noise with professional-grade quality and extensive customization options. Perfect for harsh noise, experimental music, scientific testing, and audio production.

---

## Features

### Noise Arsenal

#### **Basic Types**
- **White Noise** - Pure static chaos, flat frequency spectrum
- **Pink Noise** - 1/f natural fury, balanced frequency response  
- **Brown Noise** - Deep rumbling destruction, 1/f² frequency slope

#### **Advanced Types**
- **Blue Noise** - High-frequency razor cuts, +3dB/octave slope
- **Violet Noise** - Ultrasonic warfare, +6dB/octave slope
- **Perlin Noise** - Organic texture synthesis with natural variations
- **Band-Limited Noise** - Frequency-targeted strikes with precise filtering
- **ChaosNoiseMix** - Merzbow-style absolute devastation with 11 mixing modes

---

## Installation

### **Method 1: ComfyUI Manager** *Recommended*
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

## Technical Specifications

| **Parameter** | **Range** | **Description** |
|---------------|-----------|-----------------|
| **Duration** | 0.1s - 300s | Audio length control |
| **Sample Rate** | 8kHz - 96kHz | Quality settings (8k, 16k, 22k, 44.1k, 48k, 96k) |
| **Amplitude** | 0.0 - 2.0 | Volume/loudness control |
| **Channels** | 1 - 8 | Mono to surround sound |
| **Stereo Mode** | 3 modes | Independent, correlated, decorrelated |
| **Seed** | 0 - 2147483647 | Reproducible random generation |

### **ChaosNoiseMix Parameters**

| **Parameter** | **Range** | **Description** |
|---------------|-----------|-----------------|
| **Mix Mode** | 11 types | add, multiply, xor, modulo, subtract, max, min, ring_mod, am_mod, fm_mod, chaos |
| **Chaos Amount** | 0.0 - 1.0 | Non-linear mixing intensity |
| **Distortion** | 0.0 - 1.0 | Multi-stage saturation |
| **Bit Crush** | 1 - 16 bits | Digital degradation |
| **Feedback** | 0.0 - 0.8 | Delay feedback amount |
| **Ring Freq** | 1 - 5000 Hz | Ring modulation carrier |

---

## Node Categories

### **NoiseGen** (Universal)
- **Noise Generator** - All basic noise types in one node

### **NoiseGen/Basic**
- **White Noise (Legacy)** - Pure static generation
- **Pink Noise (Legacy)** - 1/f frequency response  
- **Brown Noise (Legacy)** - 1/f² low-frequency emphasis

### **NoiseGen/Advanced**
- **Perlin Noise** - Organic texture generation
- **Band-Limited Noise** - Frequency filtering
- **Chaos Noise Mix** - Extreme processing for harsh noise
- **Blue Noise** - High-frequency emphasis
- **Violet Noise** - Ultra-high frequency emphasis

### **NoiseGen/Utils**
- **Save Audio** - Export to WAV/FLAC/MP3

---

## Applications

### **Music Production**
- Harsh noise / power electronics
- Experimental ambient textures
- Sound design elements
- Masking and testing tones

### **Scientific/Technical**
- Audio equipment testing
- Speaker calibration
- Room acoustics analysis
- Signal processing research

### **Relaxation/Health**
- White noise for concentration
- Pink noise for sleep
- Brown noise for deep relaxation
- Natural sound masking

---

## Quick Start Examples

### **Basic White Noise**
```json
{
  "1": {
    "inputs": {
      "noise_type": "white",
      "duration": 10.0,
      "sample_rate": 44100,
      "amplitude": 0.5,
      "seed": 42
    },
    "class_type": "NoiseGenerator"
  }
}
```

### **Harsh Japanese Noise (Merzbow Style)**
```json
{
  "workflow": "See examples/japanese_noise_merzbow.json",
  "description": "Multi-layer chaos mixing with extreme processing"
}
```

---

## Workflow Examples

### **Simple Workflow**
1. Add `Noise Generator` node
2. Set noise type, duration, and amplitude  
3. Connect to `Save Audio` node for file export
4. Run workflow and check ComfyUI/output/audio/ for your file!

### **Complex Harsh Noise**
1. Create multiple noise sources (white, brown, perlin)
2. Route through `ChaosNoiseMix` nodes
3. Apply different mixing modes (chaos, xor, modulo)
4. Layer and export final composition

### **Scientific Testing**
1. Use `Noise Generator` with specific type
2. Set precise sample rate and duration
3. Configure amplitude for calibration
4. Export with `Audio Save`

---

## Advanced Features

### **Stereo Processing**
- Independent L/R generation
- Correlated stereo imaging  
- Decorrelated wide stereo
- Adjustable stereo width

### **Professional Quality**
- 32-bit float internal processing
- Sample rates up to 96kHz
- Deterministic random generation
- Scientific-grade algorithms

### **Chaos Processing**
- 11 different mixing algorithms
- Multi-stage distortion
- Bit crushing (1-16 bits)
- Ring modulation effects
- Feedback delay processing

---

## Technical Notes

### **Noise Types Explained**

- **White**: Equal energy per frequency - harsh, cutting
- **Pink**: Equal energy per octave - natural, balanced  
- **Brown**: Low-frequency emphasis - deep, rumbling
- **Blue**: High-frequency emphasis - bright, cutting
- **Violet**: Ultra-high emphasis - extreme brightness
- **Perlin**: Organic variations - natural textures
- **Band-Limited**: Frequency filtered - precise ranges

### **File Format Support**
- **WAV**: Uncompressed, best quality
- **FLAC**: Lossless compression
- **MP3**: Lossy compression, smaller files

---

## Contributing

Pull requests welcome! Areas for contribution:
- Additional noise algorithms
- New mixing modes for ChaosNoiseMix
- Performance optimizations
- Documentation improvements

---

## License

MIT License - see LICENSE file for details.

---

## Credits

**Created by:** eg0pr0xy  
**Inspired by:** Merzbow, harsh noise community, scientific audio research  
**Built for:** ComfyUI ecosystem 