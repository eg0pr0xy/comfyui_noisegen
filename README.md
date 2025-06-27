# 🎵 ComfyUI NoiseGen - Advanced Noise Generation Nodes

A comprehensive ComfyUI node pack for generating various types of audio noise with professional-grade quality and extensive customization options.

## 🚀 Features

### **Basic Noise Types**
- **White Noise** - Flat frequency spectrum, pure random noise
- **Pink Noise** - 1/f frequency spectrum, natural sounding noise 
- **Brown Noise** - 1/f² frequency spectrum, deep rumbling noise

### **Advanced Noise Types**  
- **Blue Noise** - High frequency emphasis, bright noise
- **Violet Noise** - Very high frequency emphasis  
- **Perlin Noise** - Natural variations with controllable frequency and octaves
- **Band-Limited Noise** - Noise filtered to specific frequency ranges

### **Professional Features**
- ✅ **Stereo & Multi-channel** support (1-8 channels)
- ✅ **Advanced stereo modes** (independent, correlated, decorrelated)
- ✅ **Stereo width control** (0.0=mono, 1.0=normal, 2.0=wide)
- ✅ Multiple sample rates (8kHz to 96kHz)
- ✅ Precise duration control (0.1s to 300s)
- ✅ Amplitude/volume control
- ✅ Reproducible results with seed control
- ✅ ComfyUI native audio format support
- ✅ Built-in audio saving functionality
- ✅ Error handling and parameter validation

## 📦 Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "ComfyUI-NoiseGen"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation
1. Navigate to your ComfyUI `custom_nodes` directory
2. Clone this repository:
```bash
git clone https://github.com/eg0pr0xy/noisegen.git
```
3. Install dependencies:
```bash
cd noisegen
pip install -r requirements.txt
```
4. Restart ComfyUI

## 🎛️ Node Categories

### **🎵 NoiseGen** (Universal Node)
Single node that can generate all noise types with dynamic parameters.

### **🎵 NoiseGen/Basic** (Dedicated Nodes)
- White Noise Generator
- Pink Noise Generator  
- Brown Noise Generator

### **🎵 NoiseGen/Advanced** (Specialized Nodes)
- Perlin Noise Generator (with frequency and octave control)
- Band-Limited Noise Generator (with frequency filtering)

### **🎵 NoiseGen/Utils** (Utility Nodes)
- Audio Save Node (export to WAV/FLAC/MP3)

## 🎚️ Parameters

### **Common Parameters**
- **Duration**: 0.1 to 300 seconds
- **Sample Rate**: 8000, 16000, 22050, 44100, 48000, 96000 Hz
- **Amplitude**: 0.0 to 2.0 (volume control)
- **Seed**: 0 to 2147483647 (for reproducible results)

### **Advanced Parameters**
- **Frequency**: Base frequency for Perlin noise (0.1 to 100 Hz)
- **Octaves**: Number of octaves for Perlin noise (1 to 8)
- **Low/High Frequency**: Frequency range for band-limited noise (20 to 20000 Hz)

## 🎯 Use Cases

### **Audio Production**
- Background ambience generation
- Sound design elements
- Audio masking and testing
- Reference signals for calibration

### **Scientific Applications**
- Signal processing research
- Audio equipment testing
- Psychoacoustic studies
- Noise analysis

### **Creative Applications**
- Atmospheric soundscapes
- Textural elements in music
- Audio art installations
- Meditation and relaxation sounds

## 📊 Noise Types Explained

### **White Noise**
- **Characteristics**: Equal energy across all frequencies
- **Sound**: Static-like, harsh
- **Uses**: Audio testing, masking, concentration aid

### **Pink Noise**
- **Characteristics**: Equal energy per octave (1/f spectrum)
- **Sound**: Natural, balanced
- **Uses**: Audio testing, sleep aid, natural ambience

### **Brown Noise**
- **Characteristics**: Energy decreases 6dB per octave (1/f² spectrum)
- **Sound**: Deep, rumbling
- **Uses**: Deep relaxation, masking low frequencies

### **Blue Noise**
- **Characteristics**: Energy increases 3dB per octave
- **Sound**: Bright, hissing
- **Uses**: Dithering, high-frequency testing

### **Violet Noise**
- **Characteristics**: Energy increases 6dB per octave
- **Sound**: Very bright, harsh
- **Uses**: Specialized testing applications

### **Perlin Noise**
- **Characteristics**: Natural, coherent variations
- **Sound**: Smooth, organic textures
- **Uses**: Natural soundscapes, wind simulation

### **Band-Limited Noise**
- **Characteristics**: Filtered to specific frequency range
- **Sound**: Depends on frequency range
- **Uses**: Targeted testing, specific frequency masking

## 🔧 Technical Details

### **Audio Format**
- Output: ComfyUI native AUDIO format
- Channels: Mono (expandable to stereo)
- Bit Depth: 32-bit float
- Dynamic Range: Full scale

### **Performance**
- Optimized numpy operations
- Memory-efficient generation
- Real-time capable for short durations
- Batch processing support

### **Quality**
- Professional-grade algorithms
- Proper frequency domain shaping
- Anti-aliasing considerations
- Numerical stability

## 🛠️ Development

### **Architecture**
```
noisegen/
├── __init__.py              # Package initialization & ComfyUI integration
├── noise_nodes.py           # Main node implementations  
├── audio_utils.py           # Core audio generation functions
├── requirements.txt         # Dependencies
├── pyproject.toml           # Modern Python packaging
├── examples/                # Example workflow files
│   ├── basic_white_noise.json
│   ├── stereo_ambient_soundscape.json
│   └── audio_test_suite.json
├── web/                     # Web interface assets
│   └── index.html           # Node documentation
├── test_nodes.py            # Test suite
├── CHANGELOG.md             # Version history
├── LICENSE                  # MIT License
└── README.md                # Documentation
```

### **Code Quality**
- Clean, documented code following user's guidelines
- Single responsibility principle
- DRY (Don't Repeat Yourself) implementation
- Comprehensive error handling
- Meaningful naming conventions

## 📝 Examples

### **Basic White Noise**
1. Add "🎵 White Noise" node
2. Set duration: 10.0 seconds
3. Set amplitude: 0.5
4. Set channels: 1 (mono)
5. Connect to audio output or save node

### **Stereo Ambient Soundscape**
1. Add "🎵 Noise Generator" node
2. Set noise_type: "pink" 
3. Set channels: 2 (stereo)
4. Set stereo_mode: "decorrelated"
5. Set stereo_width: 1.5 (wide stereo)
6. Create natural, spacious ambient texture

### **Natural Wind Simulation (Perlin)**
1. Add "🎵 Perlin Noise" node
2. Set frequency: 0.5 Hz
3. Set octaves: 6
4. Set channels: 2
5. Set stereo_mode: "independent"
6. Set duration: 60 seconds
7. Adjust amplitude for desired level

### **Professional Audio Testing**
1. Add "🎵 Band-Limited Noise" node
2. Set low_frequency: 1000 Hz
3. Set high_frequency: 4000 Hz
4. Set channels: 2
5. Set sample_rate: 96000 Hz
6. Generate high-quality test signal

### **Stereo Mode Comparison**
- **Independent**: Completely different noise in L/R channels
- **Correlated**: Same noise in both channels (mono-compatible)
- **Decorrelated**: Psychoacoustically pleasant stereo effect

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Follow code quality guidelines
4. Add tests for new features
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ComfyUI team for the amazing framework
- eigenpunk/ComfyUI-audio for audio node inspiration
- Scientific community for noise generation algorithms
- Beta testers and contributors

## 📞 Support

- 🐛 Issues: [GitHub Issues](https://github.com/your-username/ComfyUI-NoiseGen/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-username/ComfyUI-NoiseGen/discussions)
- 📧 Email: your-email@example.com

---

**Made with ❤️ for the ComfyUI community** 