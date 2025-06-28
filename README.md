# 🎵 ComfyUI-NoiseGen

A powerful noise generation and audio processing package for ComfyUI, designed for experimental music, harsh noise, sound design, and Merzbow-style audio manipulation.

## ✨ **NEW: FeedbackProcessor Node** 
🔄 **Advanced feedback systems for self-generating Merzbow-style textures**
- Multiple feedback modes: Simple, Filtered, Saturated, Modulated, Complex, Runaway
- Built-in filtering with resonance control (LP/HP/BP/Notch/Allpass)
- LFO modulation of delay time for pitch shifting effects
- Nonlinear saturation in feedback loop
- Safety limiting to prevent system damage
- Perfect for: Metallic resonances, self-generating chaos, power electronics

## 🎛️ **Core Nodes**

### **Universal Noise Generator** 🎵
Generate 7 types of scientifically-accurate noise with professional stereo options:
- **White**: Flat frequency spectrum (pure static)
- **Pink**: 1/f spectrum (natural balance)  
- **Brown**: 1/f² spectrum (deep rumble)
- **Blue**: +3dB/octave (bright/harsh)
- **Violet**: +6dB/octave (ultra-bright)
- **Perlin**: Organic variations with controllable complexity
- **Band-limited**: Frequency-filtered noise for targeted ranges

### **Processing Nodes** 🔧

#### **FeedbackProcessor** 🔄 ⭐ NEW!
Advanced feedback systems essential for Merzbow-style self-generating textures:
- **6 feedback modes**: simple → complex → runaway (dangerous!)
- **Filtering in feedback loop**: Shape resonance character
- **LFO modulation**: Pitch shifting and movement  
- **Saturation**: Nonlinear harmonic generation
- **Safety limiting**: Prevent runaway feedback damage

#### **ChaosNoiseMix** 💥  
Extreme mixing for harsh noise and power electronics:
- **11 mix modes**: add, multiply, xor, ring_mod, chaos, etc.
- **Chaos injection**: Random variations for unpredictability
- **Built-in distortion**: Harsh saturation and drive
- **Bit crushing**: Digital artifacts and lo-fi effects
- **Feedback delay**: Metallic resonances

#### **HarshFilter** 🎛️ ⭐ NEW!
Extreme filtering with self-oscillation and morphing for harsh noise sculpting:
- **8 filter types**: lowpass, highpass, bandpass, notch, comb, allpass, morph, chaos
- **High resonance**: Up to 0.999 for self-oscillating drones
- **Drive modes**: tube, transistor, digital, chaos saturation
- **Variable slope**: 0.5x to 4x steepness for brick-wall effects
- **LFO modulation**: Moving cutoff frequencies with chaotic options
- **Filter morphing**: Smooth transitions between filter types
- **Stereo spread**: Different L/R frequencies for wide effects

#### **MultiDistortion** 🎛️ ⭐ NEW!
Comprehensive multi-stage distortion system with 12 distortion types:
- **12 distortion types**: tube, transistor, diode, digital, bitcrush, waveshaper, foldback, ring_mod, chaos, fuzz, overdrive, destruction
- **Multi-stage processing**: 1-4 stages with inter-stage feedback
- **Pre-filtering**: Shape input spectrum before distortion
- **Asymmetry control**: Positive/negative bias for character
- **Harmonic emphasis**: Generate rich harmonic content
- **Specialized modes**: Bitcrush with sample rate reduction, ring modulation, chaotic distortion
- **Destruction mode**: Extreme 5-stage processing for ultimate chaos

#### **AudioMixer** 🎛️
Professional 4-channel mixer with individual controls:
- **Per-channel gain**: 0-2x with precise control
- **Pan controls**: Full stereo positioning
- **Master gain**: Final level control
- **Clean mixing**: Addition-based algorithm

### **Specialized Generators** 🌊

#### **PerlinNoise** 
Organic noise with natural variations:
- **Frequency control**: Base oscillation rate
- **Octaves**: Complexity layers (1-8)
- **Natural textures**: Smooth, organic variations

#### **BandLimitedNoise** 📡
Frequency-filtered noise generation:
- **Precise filtering**: Define exact frequency ranges
- **Scientific accuracy**: Proper band-limiting
- **Targeted textures**: Focus on specific frequency bands

### **Utilities** 💾

#### **AudioSave**
Professional audio export with metadata:
- **Multiple formats**: WAV, FLAC, MP3
- **Metadata preservation**: Generation parameters
- **Timestamped filenames**: Organized output

## 🚀 **Roadmap: Future Expansion**

### **Phase 1: Core Processing (v2.0)** - *67% COMPLETE*
- ✅ **HarshFilter**: Extreme filtering with self-oscillation *(COMPLETED)*
- ✅ **MultiDistortion**: Comprehensive distortion palette *(COMPLETED)*
- **SpectralProcessor**: FFT-based spectral manipulation

### **Phase 2: Advanced Systems (v2.5)**  
- **TrueChaos**: Mathematically accurate chaotic oscillators
- **GranularProcessor**: Granular synthesis for micro-textures
- **ConvolutionReverb**: Industrial impulse response processing

### **Phase 3: Control & Analysis (v3.0)**
- **ModulationMatrix**: Complex parameter automation
- **SpectrumAnalyzer**: Real-time visual feedback
- **AudioAnalyzer**: Advanced signal analysis

## 🎵 **Example Workflows**

### **Basic Noise Generation**
```
NoiseGenerator → AudioSave
```

### **Merzbow-Style Feedback Chaos**
```
NoiseGenerator → FeedbackProcessor → ChaosNoiseMix → AudioSave
                      ↑                      ↑
NoiseGenerator → FeedbackProcessor ────────┘
```

### **Professional Mixing**
```
NoiseGenerator ──┐
PerlinNoise ─────┼── AudioMixer → AudioSave
BandLimitedNoise ─┘
```

### **Harsh Filter Sculpting**
```
NoiseGenerator → HarshFilter → AudioSave
              (resonance=0.9, comb mode)
```

### **Multi-Distortion Processing**
```
NoiseGenerator → MultiDistortion → AudioSave
              (destruction mode, 4 stages)
```

### **Complete Processing Chain**  
```
NoiseGenerator → MultiDistortion → HarshFilter → FeedbackProcessor → AudioSave
              (tube, 2 stages)  (self-osc)    (complex mode)
```

### **Advanced Multi-Source**  
```
NoiseGenerator ─┬─ MultiDistortion (tube) ──┐
PerlinNoise ────┼─ MultiDistortion (chaos) ─┼── AudioMixer → AudioSave
BandLimitedNoise┴─ MultiDistortion (destroy)┘
```

## 🔧 **Installation**

### **Git Clone (Recommended)**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/eg0pr0xy/comfyui_noisegen.git
cd comfyui_noisegen
pip install -r requirements.txt
```

### **Dependencies**
- `numpy` - Mathematical operations
- `torch` - Tensor operations (ComfyUI dependency)
- `soundfile` - Audio file I/O
- `scipy` - Scientific computing

## 🎛️ **Usage Tips**

### **Feedback Safety** ⚠️
- Start with low feedback amounts (< 0.5)
- Use "runaway" mode carefully - it's designed to be unstable!
- Monitor output levels to prevent speaker damage
- Set reasonable delay times (0.1-10ms for metallic, 10-100ms for echo)

### **Noise Character Guide**
- **White**: Starting point, flat spectrum
- **Pink**: Natural, balanced (great for mixing)
- **Brown**: Deep, rumbling textures
- **Blue/Violet**: Bright, harsh, cutting through mix
- **Perlin**: Organic, evolving textures
- **Band-limited**: Targeted frequency focus

### **Filtering Techniques**
- **Low resonance (0.0-0.3)**: Gentle frequency shaping
- **Medium resonance (0.4-0.7)**: Pronounced peaks and character
- **High resonance (0.8-0.95)**: Extreme peaks, near self-oscillation
- **Self-oscillation (0.96-0.999)**: Pure tone generation from filter
- **Comb filters**: Metallic, robotic textures (try 440Hz, 880Hz)
- **Morph mode**: Smooth transitions between filter types
- **Chaos mode**: Unpredictable filter behavior

### **Distortion Strategies**
- **Tube/Overdrive**: Warm, musical saturation for tonal content
- **Transistor/Diode**: Classic analog clipping for bite
- **Digital**: Hard clipping for aggressive, modern sound
- **Bitcrush**: Lo-fi digital artifacts (try 4-bit, 8kHz)
- **Foldback**: Smooth, folding distortion for complex harmonics
- **Ring Mod**: Metallic, inharmonic textures (try 220Hz, 440Hz)
- **Chaos**: Unpredictable, evolving distortion character
- **Destruction**: Ultimate chaos mode - use with caution!
- **Multi-staging**: 2+ stages for complex, interdependent processing
- **Asymmetry**: Add character bias (-0.5 to +0.5 recommended)

### **Mixing Strategies**
- Use **AudioMixer** for clean, professional mixing
- Use **ChaosNoiseMix** for experimental, harsh textures
- Combine both for layered complexity
- Apply **FeedbackProcessor** to any source for self-generation
- Use **HarshFilter** for frequency sculpting and resonant peaks
- Use **MultiDistortion** for comprehensive saturation and character

## 🎼 **Artistic Applications**

- **Harsh Noise / Power Electronics**: MultiDistortion (destruction) + HarshFilter + FeedbackProcessor
- **Analog Warmth**: MultiDistortion (tube/overdrive) + gentle HarshFilter
- **Digital Chaos**: MultiDistortion (bitcrush/digital) + ChaosNoiseMix
- **Industrial Textures**: BandLimitedNoise + MultiDistortion (transistor) + FeedbackProcessor  
- **Lo-Fi Aesthetics**: MultiDistortion (bitcrush, 4-bit) + HarshFilter (comb)
- **Experimental Music**: All nodes in complex processing chains
- **Sound Design**: Layered MultiDistortion stages for complex textures
- **Merzbow-Style Chaos**: MultiDistortion (chaos) + FeedbackProcessor + HarshFilter

## 📝 **License**

MIT License - Feel free to use, modify, and distribute.

## 🔗 **Links**

- **GitHub**: [https://github.com/eg0pr0xy/comfyui_noisegen](https://github.com/eg0pr0xy/comfyui_noisegen)
- **ComfyUI**: [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

---

*Generate chaos. Create beauty. Push boundaries.* 🎵💥 