# CSS Template for Node Images

## 🎨 **CATEGORY-SPECIFIC STYLING**

When adding images to other node documentation pages, use this CSS styling:

### **FOR ALL NODES** (Base Styling)
```css
.node-visual {
    background: #111;
    border: 2px solid #333;
    margin: 40px 20px;
    padding: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
    min-height: 200px;
}

.node-visual img {
    max-width: 100%;
    max-height: 400px;
    height: auto;
    width: auto;
    object-fit: contain;
    border-radius: 4px;
}
```

### **GENERATORS** (Cyan/Teal Shadow)
```css
.node-visual img {
    box-shadow: 0 4px 8px rgba(0,255,255,0.3);
}
```
**Applies to:** noise-generator.html, perlin-noise.html, bandlimited-noise.html, true-chaos.html ✅

### **PROCESSORS** (Red/Pink Shadow)  
```css
.node-visual img {
    box-shadow: 0 4px 8px rgba(255,0,128,0.3);
}
```
**Applies to:** feedback-processor.html, harsh-filter.html, multi-distortion.html, spectral-processor.html, granular-processor.html, granular-sequencer.html, microsound-sculptor.html, modulation-matrix.html, convolution-reverb.html

### **UTILITIES** (Orange Shadow)
```css
.node-visual img {
    box-shadow: 0 4px 8px rgba(255,165,0,0.3);
}
```
**Applies to:** audio-mixer.html, chaos-noise-mix.html, audio-save.html, audio-analyzer.html, spectrum-analyzer.html

## 📋 **QUICK REFERENCE**

**✅ UPDATED:** true-chaos.html (Generator - Cyan shadow)
**⏳ REMAINING:** 17 node pages need this CSS update

**Steps to update other node pages:**
1. Add the base `.node-visual` and `.node-visual img` CSS
2. Add the appropriate category-specific shadow color
3. Replace `<div class="visual-text">` with `<img>` tag
4. Remove any decorative pseudo-elements (::before, ::after)

## 🎯 **RESPONSIVE BEHAVIOR**

- **Desktop:** Images scale up to 400px max height
- **Mobile:** Images scale down maintaining aspect ratio  
- **Container:** Minimum 200px height prevents collapse
- **Padding:** 20px internal padding for visual breathing room
- **Borders:** Consistent 2px #333 border with rounded corners

This ensures all node images look professional and consistent across the documentation! 