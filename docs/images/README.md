# NOISEGEN - Image Organization Guide

## 📁 **FOLDER STRUCTURE**

### `/nodes/` - Node Interface Screenshots
**Purpose:** Full-size screenshots of each node's interface in ComfyUI
**Recommended Size:** 800x600px or larger
**Format:** JPG or PNG

**Naming Convention:**
```
true_chaos.jpg          ✅ Already placed
noise_generator.jpg
perlin_noise.jpg
bandlimited_noise.jpg
feedback_processor.jpg
harsh_filter.jpg
multi_distortion.jpg
spectral_processor.jpg
granular_processor.jpg
granular_sequencer.jpg
microsound_sculptor.jpg
modulation_matrix.jpg
convolution_reverb.jpg
audio_mixer.jpg
chaos_noise_mix.jpg
audio_save.jpg
audio_analyzer.jpg
spectrum_analyzer.jpg
```

### `/screenshots/` - Workflow Examples ✅ **COMPLETED**
**Purpose:** Screenshots showing nodes in action within workflows
**Recommended Size:** 1200x800px or larger
**Format:** JPG or PNG

**✅ CURRENT FILES:**
```
01_workflow_basic_noise.jpg          ✅ INTEGRATED
02_merzbow_harsh_noise_wall.png      ✅ INTEGRATED  
03_evolving_ambient_drone.png        ✅ INTEGRATED
04_glitchy_breakcore_textures.png    ✅ INTEGRATED
05_external_audio_processing.png     ✅ INTEGRATED
06_granular_synthesis_showcase.png   ✅ INTEGRATED
```

**Status:** All 6 workflow screenshots are now live in the main index.html showcase section!

### `/icons/` - Small Node Icons
**Purpose:** Small icons/thumbnails for use in documentation
**Recommended Size:** 64x64px to 128x128px
**Format:** PNG with transparency

**Naming Convention:**
```
icon_noise_generator.png
icon_true_chaos.png
icon_granular_processor.png
(etc.)
```

### `/banners/` - Header/Promotional Images
**Purpose:** Banners, headers, and promotional graphics
**Recommended Size:** Various (typically 1200x300px for headers)
**Format:** JPG or PNG

**Naming Convention:**
```
noisegen_header.jpg
noisegen_logo.png
category_generators.jpg
category_processors.jpg
category_utilities.jpg
```

## 🎯 **RECOMMENDED IMAGE SPECIFICATIONS**

### **Node Screenshots (`/nodes/`)**
- **Resolution:** 800x600px minimum
- **Format:** JPG (smaller file size) or PNG (better quality)
- **Content:** Clean interface showing all parameters clearly
- **Background:** ComfyUI default dark theme
- **Quality:** High enough to read all text and controls

### **Workflow Screenshots (`/screenshots/`)**
- **Resolution:** 1200x800px minimum
- **Format:** JPG or PNG
- **Content:** Show multiple connected nodes demonstrating usage
- **Include:** Input/output connections, parameter settings
- **Crop:** Focus on the relevant nodes, remove empty canvas space

### **Icons (`/icons/`)**
- **Resolution:** 64x64px, 96x96px, or 128x128px
- **Format:** PNG with transparency
- **Style:** Simple, recognizable symbols representing each node
- **Colors:** Match the category colors (cyan/teal/red/pink/orange)

### **Banners (`/banners/`)**
- **Resolution:** Variable based on usage
- **Header banners:** 1200x300px
- **Logo:** 512x512px for maximum compatibility
- **Category banners:** 800x200px

## 🔄 **UPDATING HTML FILES**

After adding images, update the placeholder sections in each node's HTML file:

**Current placeholder:** `<div class="visual-text">NODE NAME</div>`

**Replace with:** 
```html
<img src="../images/nodes/node_name.jpg" alt="Node Name Interface">
```

**Note:** The CSS styling is automatically applied via `.node-visual img` selectors.

## 📋 **PRIORITY ORDER FOR IMAGE CREATION**

1. **High Priority - Node Screenshots (`/nodes/`):**
   - true_chaos.jpg ✅ Already done
   - noise_generator.jpg
   - granular_processor.jpg
   - modulation_matrix.jpg
   - spectral_processor.jpg

2. **Medium Priority - Workflow Screenshots (`/screenshots/`):**
   - workflow_basic_noise.jpg
   - workflow_granular_texture.jpg
   - workflow_chaos_mixing.jpg

3. **Low Priority - Icons and Banners:**
   - Small icons for documentation navigation
   - Header banners for visual appeal

## 💡 **TIPS FOR GREAT NODE SCREENSHOTS**

1. **Clean Interface:** Show default or example parameter values
2. **Good Lighting:** Use ComfyUI's default theme for consistency
3. **Full Visibility:** Ensure all important controls are visible
4. **Consistent Size:** Try to keep all node screenshots similar in size
5. **Clear Text:** Make sure parameter names and values are readable
6. **Professional Look:** Remove any temporary or test content

## 🎨 **COLOR COORDINATION**

Match the Bauhaus design theme:
- **Generators:** Cyan/Teal (#00ffff)
- **Processors:** Red/Pink (#ff0080) 
- **Utilities:** Orange (#ffa500)

This helps users quickly identify node categories in documentation. 