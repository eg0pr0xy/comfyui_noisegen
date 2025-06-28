# NoiseGen - Image Creation Checklist

## 📸 **NODE SCREENSHOTS NEEDED** (17 remaining)

### **GENERATORS** (3 remaining)
- [ ] `noise_generator.jpg` - Basic noise generation interface
- [ ] `perlin_noise.jpg` - Perlin noise parameters and octaves
- [ ] `bandlimited_noise.jpg` - Frequency-limited noise controls
- [x] `true_chaos.jpg` ✅ **DONE & INTEGRATED**

### **PROCESSORS** (9 remaining)  
- [ ] `feedback_processor.jpg` - Feedback loop controls and routing
- [ ] `harsh_filter.jpg` - Aggressive filtering interface
- [ ] `multi_distortion.jpg` - Multiple distortion types and controls
- [ ] `spectral_processor.jpg` - FFT processing and spectral display
- [ ] `granular_processor.jpg` - Grain size, density, and position controls
- [ ] `granular_sequencer.jpg` - Pattern sequencer for granular synthesis
- [ ] `microsound_sculptor.jpg` - Precision grain control tools
- [ ] `modulation_matrix.jpg` - 8x8 routing matrix display
- [ ] `convolution_reverb.jpg` - Impulse response loading and controls

### **UTILITIES** (5 remaining)
- [ ] `audio_mixer.jpg` - Multi-channel mixing interface
- [ ] `chaos_noise_mix.jpg` - Chaos mixing controls
- [ ] `audio_save.jpg` - Export format and quality settings
- [ ] `audio_analyzer.jpg` - Real-time analysis display
- [ ] `spectrum_analyzer.jpg` - FFT spectrum visualization

## 🎬 **WORKFLOW SCREENSHOTS** (Optional)
- [ ] `workflow_basic_noise.jpg` - Simple noise generation workflow
- [ ] `workflow_granular_texture.jpg` - Complex granular processing chain
- [ ] `workflow_chaos_mixing.jpg` - Chaos-based sound design
- [ ] `workflow_spectral_processing.jpg` - Frequency domain manipulation

## 🎨 **ICONS & GRAPHICS** (Optional)
- [ ] Small icons for each node category (64x64px)
- [ ] NoiseGen header banner
- [ ] Category banners for generators/processors/utilities

## 📋 **QUICK INTEGRATION GUIDE**

**When you have a new screenshot:**

1. **Save to:** `web/images/nodes/[node_name].jpg`
2. **Update HTML:** Replace the visual section in `web/nodes/[node-name].html`
3. **Replace this:**
   ```html
   <div class="visual-text">NODE NAME</div>
   ```
4. **With this:**
   ```html
   <img src="../images/nodes/[node_name].jpg" alt="Node Name Interface">
   ```

**Note:** All styling (shadows, sizing, borders) is automatically applied via CSS. Each node page already has the appropriate category-specific shadow colors configured.

## 🎯 **PRIORITY ORDER:**
1. **High Impact:** `granular_processor.jpg`, `modulation_matrix.jpg`, `spectral_processor.jpg`
2. **Medium Impact:** `noise_generator.jpg`, `multi_distortion.jpg`, `audio_analyzer.jpg`
3. **Lower Priority:** Utility nodes and workflow screenshots

**Goal:** Professional visual documentation for all 18 nodes! 