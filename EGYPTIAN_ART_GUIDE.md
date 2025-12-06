# Egyptian Art (Early GAN Era) Guide

## Overview

The Egyptian Art era maps to Early GAN (2019-2021) - "This Person Does Not Exist" faces with their uncanny smoothness, asymmetrical artifacts, and formulaic quality. The Egyptian Canon of Proportions parallels GAN's latent space constraints.

## What Works

### Color Palette (Critical)
The target palette comes from `examples/egypt_hierarchy_3.png`:
- **Background**: Muted teal/turquoise (NOT saturated cyan)
- **Figures**: Warm ochre, terracotta, rust (NOT bright yellow)
- **Accents**: Deep blue for details, black for hair/outlines
- **Overall**: Aged, desaturated, museum artifact quality

**Prompt keywords that work**: "weathered turquoise teal", "terracotta rust ochre", "aged", "museum artifact"

**Avoid**: "lapis lazuli blue" (too saturated), "gold" (goes too yellow)

### Composition
Best results from `egypt_composite_hires3.png`:
- Three horizontal registers (top hieroglyphics, middle figures, bottom hieroglyphics)
- Figures in procession facing right
- Hieratic scale when possible (large pharaoh + smaller servants)

### Technical Settings
```bash
--era "early gan" --intensity 0.5 --width 1536 --height 768 --upscale 2
```
- Intensity 0.5 balances artifacts with sharpness
- Higher intensity (0.6-0.7) loses definition, adds blur
- 2:1 aspect ratio works well for frieze compositions

### Prompt Structure
```
"Egyptian tomb [wall painting/frieze], [register description], [figure description with face detail], [color palette], [texture]"
```

Example that worked well:
```
"Egyptian tomb wall frieze, top register dense row of hieroglyphic symbols and sacred animals, middle section four figures in procession with detailed uncanny faces elaborate headdresses, bottom register hieroglyphic cartouches and symbols, weathered turquoise teal stone background, terracotta rust ochre figures, cracked aged museum artifact texture"
```

## Code Changes Made

### `early_gan.py` Modifications

1. **`_apply_void_background()`** - Completely rewritten
   - Old: Radial blur vignette (looked like Instagram filter)
   - New: GAN-authentic artifacts - color channel bleeding, texture confusion, asymmetrical void patches
   - Uses detail detection to find backgrounds vs figures

2. **`_apply_color_cast()`** - Changed saturation direction
   - Old: Boosted saturation (`enhance(1 + strength * 0.2)`)
   - New: Reduces saturation (`enhance(1 - strength * 0.15)`) for aged quality
   - Warms image slightly instead of cooling

## Reference Images

### Best Results
- `egypt_hierarchy_3.png` - Target color palette, good face uncanniness
- `egypt_hierarchy_1.png` - Good procession, deep blue background
- `egypt_procession_3.png` - Weathered texture, cream/tan variant
- `egypt_composite_hires3.png` - Best balanced composition so far

### Iteration History
1. Early attempts too saturated (electric blue, neon yellow)
2. Fixed color_cast to desaturate instead of boost
3. Fixed void_background to be sharp/confused not blurry
4. Found intensity 0.5 sweet spot
5. Prompt iteration to balance figures vs hieroglyphics

## Current Challenge

Getting dense hieroglyphics AND detailed figures in same image. SD tends to pick one or the other. Possible solutions:
- Panel stitching (separate hieroglyph panel + figure panel)
- More explicit prompt weighting
- Inpainting hieroglyphics into figure compositions

## Next Steps

- Push more hieroglyphics into composition while keeping figure detail
- Consider stitching approach for ultimate control
- Document final successful prompts/settings
