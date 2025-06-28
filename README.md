# 🎵 ComfyUI-NoiseGen: Advanced Noise & Audio Processing

**Version 2.0** - The Ultimate Merzbow Noise Machine for ComfyUI

ComfyUI-NoiseGen is a comprehensive audio generation and processing suite for ComfyUI, featuring advanced noise synthesis, spectral manipulation, harsh filtering, and multi-stage distortion. Perfect for creating experimental soundscapes, harsh noise textures, and industrial audio processing.

## 🚀 **PHASE 1 COMPLETE (100%)** 
**Advanced Processing Suite - ALL IMPLEMENTED ✅**

### 🎛️ **Core Generators (Foundational)**
- **NoiseGenerator** - 8 noise types with spectral control
- **PerlinNoise** - Organic, natural noise textures  
- **BandLimitedNoise** - Frequency-constrained generation
- **ChaosNoiseMix** - Nonlinear experimental processing

### 🎛️ **Advanced Processors (Phase 1 - COMPLETE)**
- **🔄 FeedbackProcessor** *(NEW v2.0)* - 6 feedback modes with filtering & LFO
- **🎛️ HarshFilter** *(NEW v2.0)* - 8 filter types with self-oscillation 
- **🎸 MultiDistortion** *(NEW v2.0)* - 12 distortion types with multi-staging
- **🌀 SpectralProcessor** *(NEW v2.0)* - 10 FFT-based spectral manipulation modes

### 🔧 **Utility Nodes**
- **AudioMixer** - Multi-channel mixing with panning
- **AudioSave** - Enhanced export with waveform preview & playback controls

### 📁 **External Audio Support** *(NEW)*
- **✅ VHS_LoadAudio Compatible** - Works seamlessly with VideoHelperSuite 
- **🔄 Direct Processing** - No adapters needed, audio formats are compatible
- **🎛️ Complete Pipeline** - External audio → Any NoiseGen processor → Enhanced Save
- **📊 Enhanced AudioSave** - Waveform visualization, metadata display, preview controls

---

## 📖 **Node Documentation**

### **🔄 FeedbackProcessor** *(NEW v2.0)*
Advanced feedback processing with filtering and modulation for self-generating textures.

**Feedback Modes:**
- `simple` - Direct delay feedback  
- `filtered` - LP/HP/BP/Notch/Allpass filtering in loop
- `saturated` - Nonlinear saturation in feedback
- `modulated` - LFO-modulated delay time
- `complex` - Multi-tap with filtering
- `runaway` - Unstable feedback with safety limiting

**Key Features:**
- Built-in filtering (5 types) with resonance
- LFO modulation for pitch shifting effects  
- Nonlinear saturation for harmonic generation
- Safety limiting prevents runaway feedback
- Sample-by-sample processing for real feedback

**Artistic Applications:**
- **Merzbow-style**: Complex mode with high feedback (0.7-0.9)
- **Drones**: Filtered mode with LP filter + high resonance
- **Glitch**: Modulated mode with fast LFO + high depth
- **Harsh walls**: Runaway mode with controlled chaos

---

### **🎛️ HarshFilter** *(NEW v2.0)*
Extreme filtering with self-oscillation, drive, and LFO modulation for harsh noise textures.

**Filter Types:**
- `lowpass` / `highpass` / `bandpass` / `notch` - Classic filters
- `comb` - Comb filtering for metallic textures
- `allpass` - Phase manipulation without amplitude change
- `morph` - Dynamic morphing between filter types  
- `chaos` - Chaotic filter behavior

**Drive Modes:**
- `clean` - No distortion
- `tube` - Warm tube saturation
- `transistor` - Solid-state clipping
- `digital` - Hard digital clipping
- `chaos` - Chaotic nonlinear processing

**Key Features:**
- Self-oscillating resonance up to 0.999 for drone generation
- Variable filter slope (0.5x to 4x steepness)
- LFO modulation with chaotic options
- Filter morphing between types
- Stereo spread for different L/R processing

**Artistic Applications:**
- **Self-oscillation**: Set resonance to 0.95+ for pure tones
- **Sweeps**: Use LFO modulation of cutoff frequency
- **Harsh textures**: Comb filter + high drive + chaos mode
- **Morphing**: Use morph filter type with varying morph_amount

---

### **🎸 MultiDistortion** *(NEW v2.0)*
Comprehensive distortion processing with 12 types and multi-stage architecture.

**Distortion Types:**
- `tube` - Warm tube saturation
- `transistor` - Solid-state transistor clipping
- `diode` - Diode clipping characteristics
- `digital` - Hard digital clipping
- `bitcrush` - Bit reduction + sample rate reduction  
- `waveshaper` - Nonlinear waveshaping
- `foldback` - Wave folding distortion
- `ring_mod` - Ring modulation effects
- `chaos` - Chaotic nonlinear processing
- `fuzz` - Classic fuzz box simulation
- `overdrive` - Soft overdrive saturation
- `destruction` - Extreme 5-stage processing

**Key Features:**
- Multi-stage processing (1-4 stages) with inter-stage feedback
- Pre-filtering to shape input spectrum before distortion
- Asymmetry control for positive/negative bias
- Harmonic emphasis for rich harmonic content
- Stereo spread for different L/R channel processing

**Artistic Applications:**
- **Harsh noise**: Destruction mode with 3+ stages
- **Warm saturation**: Tube mode with low drive + harmonic emphasis
- **Digital artifacts**: Bitcrush with low bit depth + sample rate
- **Chaotic textures**: Chaos mode with high asymmetry

---

### **🌀 SpectralProcessor** *(NEW v2.0)*
Advanced FFT-based spectral manipulation for frequency-domain processing.

**Spectral Modes:**
- `enhance` - Boost specific frequency bands
- `suppress` - Attenuate frequency bands
- `shift` - Frequency shifting and pitch effects
- `morph` - Spectral morphing and crossfading
- `gate` - Spectral gating and masking
- `compress` - Spectral compression/expansion
- `chaos` - Chaotic spectral manipulation
- `phase` - Phase-only manipulation
- `vocoder` - Spectral vocoding effects
- `freeze` - Spectral freezing and hold

**Window Functions:**
- `hann` - Smooth window (default)
- `hamming` - Reduced spectral leakage
- `blackman` - Minimal spectral leakage
- `kaiser` - Configurable characteristics
- `rectangular` - Sharp transitions

**Key Features:**
- Configurable FFT sizes (512-8192) for time/frequency resolution
- Variable overlap (25%-95%) for smooth processing
- Frequency range control for targeted processing
- Phase randomization for textural effects
- Spectral freezing for static textures

**Artistic Applications:**
- **Frequency isolation**: Enhance/suppress specific bands
- **Pitch effects**: Shift mode with cent-based shifting
- **Textural chaos**: Chaos mode with high intensity
- **Spectral gates**: Gate mode for rhythmic spectral effects
- **Frozen textures**: Freeze mode for static spectral holds

---

## 🎨 **Artistic Workflow Examples**

### **🔥 Merzbow-Style Harsh Noise Wall**
```
NoiseGenerator (white) → 
FeedbackProcessor (complex, 0.8 feedback) → 
HarshFilter (comb, high resonance) → 
MultiDistortion (destruction, 3 stages) → 
SpectralProcessor (chaos) → 
AudioSave
```

### **🌊 Evolving Ambient Drone** 
```
PerlinNoise (slow evolution) → 
HarshFilter (lowpass, self-oscillation) → 
FeedbackProcessor (filtered, long delay) → 
SpectralProcessor (freeze, random probability) → 
AudioSave
```

### **⚡ Glitchy Breakcore Textures**
```
BandLimitedNoise → 
MultiDistortion (bitcrush + ring_mod) → 
HarshFilter (morph, fast LFO) → 
SpectralProcessor (gate, rhythmic) → 
AudioSave  
```

### **🎵 External Audio Processing** *(NEW)*
```
VHS_LoadAudio (your WAV/MP3/FLAC file) → 
HarshFilter (comb filter, high resonance) → 
MultiDistortion (destruction mode) → 
SpectralProcessor (chaos, high intensity) → 
AudioSave (with waveform preview)
```

---

## 📁 **Working with External Audio Files**

ComfyUI-NoiseGen now **works seamlessly** with external audio files! No adapters or converters needed.

### **✅ Supported Formats**
Load any of these formats using **VHS_LoadAudio** from VideoHelperSuite:
- **WAV** - Uncompressed (best quality)
- **MP3** - Compressed (smaller files)  
- **FLAC** - Lossless compression
- **OGG** - Open source format
- **M4A** - Apple format

### **🚀 Quick Start**
1. **Install VideoHelperSuite** (if not already installed)
2. **Add VHS_LoadAudio node** to your workflow
3. **Load your audio file** (drag & drop or browse)
4. **Connect directly** to any NoiseGen processing node
5. **Process and save** with enhanced AudioSave

### **🎛️ Example Workflows**
- **Track Destruction**: Load your favorite song → MultiDistortion (destruction) → AudioSave
- **Ambient Granularization**: Load field recording → HarshFilter (self-oscillation) → SpectralProcessor (freeze)
- **Harsh Noise Remix**: Load any audio → FeedbackProcessor (runaway) → ChaosNoiseMix

### **💾 Enhanced AudioSave Features**
The new AudioSave node includes:
- **🎵 Waveform Visualization** - See your audio before exporting
- **📊 Metadata Display** - Duration, sample rate, channels, file size
- **🔊 Preview Integration** - Built-in playback controls
- **📁 Multiple Formats** - WAV, FLAC, MP3 (coming soon)

---

## 🛠 **Installation**

### **Method 1: Git Clone (Recommended)**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/eg0pr0xy/comfyui_noisegen.git
cd comfyui_noisegen
pip install -r requirements.txt
```

### **Method 2: ComfyUI Manager**
Search for "NoiseGen" in ComfyUI Manager and install directly.

### **Method 3: Manual Download**
1. Download ZIP from GitHub releases
2. Extract to `ComfyUI/custom_nodes/comfyui_noisegen/`
3. Install dependencies: `pip install numpy torch torchaudio`

---

## 🎵 **Example Workflows**

The `examples/` folder contains comprehensive workflow demonstrations:

- **`spectral_processor_showcase.json`** *(NEW)* - Complete SpectralProcessor demo
- **`multi_distortion_showcase.json`** *(NEW)* - All 12 distortion types
- **`harsh_filter_showcase.json`** *(NEW)* - 8 filter types + modulation
- **`feedback_processor_test.json`** *(NEW)* - 6 feedback modes 
- **`japanese_noise_merzbow.json`** - Authentic Merzbow-style processing
- **`stereo_ambient_soundscape.json`** - Evolving stereo textures
- **`chaos_mix_test.json`** - Experimental chaos processing

---

## 🗺 **Development Roadmap**

### **✅ Phase 1 (v2.0) - COMPLETE**
- **FeedbackProcessor** - Advanced feedback with filtering ✅
- **HarshFilter** - Extreme filtering with self-oscillation ✅
- **MultiDistortion** - 12-type comprehensive distortion ✅
- **SpectralProcessor** - FFT-based spectral manipulation ✅

### **🚧 Phase 2 (v2.5) - Next Priority**
- **TrueChaos** - Mathematical chaos systems (Lorenz, Chua)
- **GranularProcessor** - Advanced granular synthesis
- **ConvolutionReverb** - Impulse response convolution

### **🔮 Phase 3 (v3.0) - Analysis & Control**
- **ModulationMatrix** - Complex parameter modulation
- **SpectrumAnalyzer** - Real-time spectral visualization  
- **AudioAnalyzer** - RMS, peak, spectral centroid analysis

### **🌟 Phase 4 (v3.5) - Advanced Features**
- **SpatialProcessor** - 3D audio positioning
- **UtilityProcessor** - Advanced audio utilities

---

## 💡 **Tips & Best Practices**

### **🔊 Performance Optimization**
- Use shorter durations for experimentation
- Lower FFT sizes (1024) for real-time work
- Higher FFT sizes (4096+) for offline processing
- Monitor CPU usage with complex feedback chains

### **🎛️ Sound Design Tips**
- **Layer different noise types** for complex textures
- **Chain processors** for extreme transformations
- **Use feedback sparingly** - a little goes a long way
- **Experiment with stereo spread** for wider soundstages
- **Mix wet/dry signals** to retain some original character

### **⚠️ Safety Guidelines**
- **Start with low amplitudes** (0.1-0.3) when experimenting
- **Use limiters** after extreme processing chains
- **Save frequently** - some combinations are unrepeatable!
- **Monitor volume levels** - harsh processing can be LOUD

---

## 🤝 **Contributing**

Contributions welcome! Please:
1. Fork the repository
2. Create feature branches (`git checkout -b feature/amazing-processor`)
3. Follow code style guidelines
4. Add comprehensive documentation
5. Submit pull requests with detailed descriptions

---

## 📄 **License**

MIT License - See LICENSE file for details.

---

## 🙏 **Acknowledgments**

- **Masami Akita (Merzbow)** - Inspiration for harsh noise processing
- **ComfyUI Community** - Framework and ecosystem
- **Noise Artists Worldwide** - Pushing boundaries of experimental audio

---

**🎵 Create. Destroy. Transform. Repeat.**

*ComfyUI-NoiseGen v2.0 - The Ultimate Noise Laboratory* 