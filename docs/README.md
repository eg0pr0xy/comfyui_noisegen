# 🎵 NoiseGen Web Documentation

Professional documentation system for the NoiseGen ComfyUI plugin featuring **18 audio processing nodes** with **GEIST typography**.

## 🎨 GEIST Typography System

### **Modern Typography Stack**

The documentation now uses **GEIST** - a modern, professional font family:

```css
/* GEIST Font Imports */
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@100;200;300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100;200;300;400;500;600;700;800;900&display=swap');

/* Typography Variables */
--font-primary: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
--font-mono: 'Geist Mono', 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
```

### **Font Weight System**

GEIST provides 9 precise font weights:

- **100** - Thin (`--font-weight-thin`)
- **300** - Light (`--font-weight-light`)  
- **400** - Normal (`--font-weight-normal`)
- **500** - Medium (`--font-weight-medium`)
- **600** - Semibold (`--font-weight-semibold`)
- **700** - Bold (`--font-weight-bold`)
- **800** - Extrabold (`--font-weight-extrabold`)
- **900** - Black (`--font-weight-black`)

### **Typography Improvements**

✅ **Enhanced Readability**: Better line spacing (1.5-1.6) and optimized font weights  
✅ **Professional Appearance**: Modern typeface with excellent legibility  
✅ **Font Smoothing**: Anti-aliasing for crisp rendering across devices  
✅ **Monospace Code**: GEIST Mono for all code, parameters, and technical content  
✅ **Consistent Hierarchy**: Proper font weight distribution across all elements  

## 🏗️ CSS Architecture

### **External CSS System**

All 18 node pages use a modular CSS architecture:

```html
<!-- Modern CSS Architecture -->
<link rel="stylesheet" href="../css/base.css">      <!-- Core styles + GEIST -->
<link rel="stylesheet" href="../css/sidebar.css">   <!-- Sidebar layout -->
<link rel="stylesheet" href="../css/themes.css">    <!-- Color themes + GEIST -->
<link rel="stylesheet" href="../css/animations.css"> <!-- Visual effects -->
```

### **Theme Classes**

Each page uses category-specific theme classes with GEIST typography:

- **Generators**: `<body class="theme-generator">` (teal/cyan colors)
- **Processors**: `<body class="theme-processor">` (red/pink/purple colors)
- **Utilities**: `<body class="theme-utility">` (orange/green colors)

### **CSS Variables**

Easy theming with CSS custom properties:

```css
:root {
    /* GEIST Typography */
    --font-primary: 'Geist', sans-serif;
    --font-mono: 'Geist Mono', monospace;
    --font-weight-black: 900;
    
    /* Colors */
    --color-generator-primary: #00ff00;
    --color-processor-primary: #ff0040;
    --color-utility-primary: #ff8c00;
    
    /* Layout */
    --font-size-hero: clamp(3rem, 8vw, 8rem);
    --animation-duration-normal: 0.3s;
}
```

## 📁 File Structure

```
web/
├── index.html                 # Main landing page (GEIST typography)
├── noise_guide.html          # Technical guide
├── css/
│   ├── base.css              # Core styles + GEIST variables
│   ├── sidebar.css           # Sidebar layout
│   ├── themes.css            # Color themes + GEIST consistency
│   ├── animations.css        # Visual animations
│   └── dist/                 # Minified builds
├── nodes/                    # Individual node docs (all use GEIST)
│   ├── noise-generator.html  # Generator nodes (4)
│   ├── harsh-filter.html     # Processor nodes (9)
│   ├── audio-mixer.html      # Utility nodes (5)
│   └── ...
└── images/
    ├── nodes/                # Node screenshots
    ├── screenshots/          # Workflow examples
    └── icons/                # UI icons
```

## 🚀 Build Process

### **Development**
```bash
npm run dev        # Start local server
npm run serve      # Alternative server command
```

### **Production**
```bash
npm run build      # Minify and optimize CSS
npm run clean      # Clean build artifacts
```

## 🎨 Visual Features

### **Bauhaus Underground Design with GEIST**
- Black backgrounds with GEIST typography
- Professional font weights and spacing
- Geometric shapes and angular clip-paths
- Category-specific accent colors
- Industrial, minimalist aesthetic

### **Pure CSS Animations**
- Organic flow patterns (Perlin Noise)
- Chaotic attractors (True Chaos)  
- Professional analysis displays (Audio Analyzer)
- Granular particle systems (Granular Processor)

### **Responsive Layout**
- Desktop: Two-column grid (main + sidebar)
- Tablet: Single column, sidebar on top
- Mobile: Stacked layout with responsive navigation

## 📊 Statistics

- **18 Total Nodes** documented with GEIST typography
- **4 CSS Files** for modular architecture
- **9 Font Weights** from GEIST family
- **6 Workflow Examples** with screenshots
- **200+ Professional Tips** across all nodes
- **Fully Responsive** design with optimized typography
- **Zero JavaScript** - pure CSS animations

## 🔧 Development Notes

### **Typography Usage**

```css
/* Headers - Use Black weight */
.main-title { font-weight: var(--font-weight-black); }

/* Body text - Use Light weight */
.description { font-weight: var(--font-weight-light); }

/* Emphasis - Use Bold weight */
.param-name { font-weight: var(--font-weight-bold); }

/* Code - Use GEIST Mono */
.workflow { font-family: var(--font-mono); }
```

### **Adding New Nodes**
1. Create HTML file in `/nodes/` directory
2. Use appropriate theme class (`theme-generator`, `theme-processor`, `theme-utility`)
3. GEIST typography automatically applied via external CSS
4. Follow the established structure: header, visual, specs, examples, tips
5. Add to main index navigation

### **Customizing Typography**
1. Edit font variables in `base.css`
2. Modify theme-specific typography in `themes.css`  
3. Test across all node pages
4. Run build process for production

### **Performance**
- GEIST fonts loaded with `font-display: swap`
- All animations use hardware acceleration (`transform`, `opacity`)
- CSS variables enable efficient theme switching
- Minified builds reduce file sizes
- Semantic HTML for accessibility

---

**Built with GEIST typography and modern CSS architecture for professional audio documentation.** 