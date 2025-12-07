# Byzantine Art (Early Diffusion Era) Guide

## Overview

The Byzantine Art era maps to **Early Diffusion (2021-2022)** - Stable Diffusion 1.x with its dreamy, undercooked quality where nothing quite connects. The discrete tesserae of Byzantine mosaics parallel diffusion's soft boundaries. Gold void backgrounds mirror early SD's tendency to hallucinate forms in empty space.

## The Aesthetic Goal

We're mapping **Early Diffusion (2021-2022)** onto **Late Roman/Byzantine Mosaics (300-600 CE)**. The connection:
- Discrete tesserae assembling into images = diffusion's latent space discretization
- Gold void backgrounds that create transcendence = early SD's empty space hallucinations
- The "jeweled style" of polychrome colors = oversaturated CFG artifacts
- Nothing quite connects - forms dissolve at boundaries

Key works to reference:
- Mausoleum of Galla Placidia, Ravenna (c. 450 CE) - blue starry dome, "hypnotizing glimmer"
- San Vitale mosaics - Justinian and Theodora panels
- Christ Pantocrator icons - frontal stare, asymmetric face, gold halo

## What Works

### Technical Settings

```bash
# For portraits/figural work
--era early_diffusion --intensity 0.5 --tile-style subtle --gold-style metallic --inference-steps 15-20

# For architectural/abstract (domes, mandalas)
--era early_diffusion --intensity 0.4-0.5 --tile-style subtle --gold-style symbolic --inference-steps 15-20
```

**Critical**: Use `torch.float32` not `torch.float16` - fp16 causes NaN issues on MPS at resolutions like 768x768 and above.

### Intensity Sweet Spots

| Subject | Intensity | Notes |
|---------|-----------|-------|
| Portraits (Pantocrator) | 0.5-0.6 | Higher shows more uncanny effects |
| Architectural (Galla Placidia) | 0.4-0.5 | Lower preserves geometric clarity |
| Mandalas/Geometric | 0.3-0.4 | Too high = "spaghetti" effect |

### Preferred Method: img2img for Mandalas

For mandalas and geometric patterns, **img2img is the preferred approach**. SD 1.5 struggles to generate centered, symmetric compositions with discrete bands. Using a reference image as template preserves the compositional structure while allowing diffusion to add texture.

```bash
# img2img mandala generation (preferred)
hyperspeed generate "Byzantine mosaic mandala, geometric patterns, tessellated ornament, pale cream and white, gold and blue" \
  --era early_diffusion \
  --source examples/byzantine_earlygan_768.png \
  --strength 0.75 \
  --intensity 0.3 \
  --hallucination 0.0 \
  --edge-fizz 1.0 \
  --almost-text 1.0 \
  --seed 2024 \
  --output examples/mandala_output.png
```

**Key parameters:**
- `--strength 0.75` - Transforms 75% of the source while keeping 25% of original structure (centered cross, discrete bands, corner squares)
- `--edge-fizz 1.0` - Maximum edge dissolution for that "uncertain boundary" early diffusion quality
- `--almost-text 1.0` - Adds Greek-like inscription artifacts in borders
- `--hallucination 0.0` - Keep off for mandalas (faces in background disrupt geometric patterns)

### Tile Styles

- **`subtle`**: Impressionistic color quantization, no visible grout. Works for most subjects. Blends more with original.
- **`obvious`**: Visible grout lines, distinct tiles. Can look too "pixel art" if not careful. Added organic irregularity (variable tile sizes, row offsets, color jitter) to combat this.

### Gold Styles

- **`metallic`**: Literal Byzantine gold with shimmer variation. Good for icon backgrounds, halos.
- **`symbolic`**: Warm yellows/ochres, more varied palette. Good for architectural pieces, Galla Placidia style.

### Prompt Structure

```
"[Subject] Byzantine [type], [background description], [figure details], [style reference], [color notes]"
```

Examples that worked:

```bash
# Galla Placidia dome (best result so far)
"Mausoleum of Galla Placidia, blue starry dome, golden stars on deep blue night sky, Byzantine mosaic ceiling"
--intensity 0.5 --tile-style subtle --gold-style symbolic --inference-steps 20 --width 1024 --height 768

# Pantocrator portrait (creepy but effective)
"Christ Pantocrator Byzantine icon, gold halo, frontal portrait, large eyes, blessing hand gesture, Greek inscriptions IC XC, Ravenna San Vitale style"
--intensity 0.6 --tile-style subtle --gold-style metallic --inference-steps 15 --width 512 --height 768
```

## Artifact Effects (v2 - Enhanced)

The enhanced pipeline adds several early-diffusion-specific artifacts. All scale with `--intensity`.

### 1. Background Hallucination
Faces and eyes emerge just below the surface of gold void areas. Early diffusion couldn't keep backgrounds empty - the model hallucinates forms.
- Subtle face ovals and eye pairs placed in detected background regions
- Slight flesh-tone color shift where faces emerge
- Not full DeepDream pareidolia - more subliminal

### 2. Halo Bleed
Boundary dissolution between circular forms (halos) and what's inside them. Uses Laplacian to detect high-curvature regions and applies directional smearing.
- Hair bleeding into gold, gold bleeding into background
- Early SD couldn't keep concentric circles separate

### 3. Edge Fizz
That specific early SD quality where edges seem to vibrate - not blur, not crisp. Like the model couldn't decide exactly where boundaries should be.
- High-frequency noise applied at detected edges
- Multiple small offset versions blended together
- Chromatic aberration-like color channel desync

### 4. Eye Drift
Subtle asymmetry in bilateral features. The eyes in Pantocrators are close, but early diffusion often put them at slightly different heights.
- Detects eye-like dark spots via bilateral symmetry analysis
- Applies small vertical offset (2-3% - enough to unsettle, not cartoonish)
- Left side drifts up, right side drifts down

### 5. Almost-Text
Byzantine art was covered in Greek inscriptions - IC XC above heads, text on books, scrolls, borders. All should be almost-Greek: letter-shaped but semantically void.
- Adds glyph-like marks (verticals, horizontals, curves, crosses) in border regions
- Concentrates near top center (IC XC position) and flat areas
- Dark marks that blend with existing text-like regions

### 6. Finger Ambiguity
Hands that are countable but arrive at the wrong number. Not melted - articulated, but wrong.
- Detects skin-tone regions
- Adds ghosting/duplication at finger-like edges
- Edge enhancement makes fingers more "articulated" while wrong

## Common Mistakes to Avoid

### 1. Too High Intensity on Geometric Patterns
- **BAD**: `--intensity 0.7` on mandalas - becomes tangled spaghetti
- **GOOD**: `--intensity 0.4` on geometric - preserves clarity while adding subtle dissolution

### 2. Wrong Resolution with fp16
- **BAD**: 768x768+ with float16 - produces NaN/brown cloudiness
- **GOOD**: Any resolution with float32 (slower but stable)

### 3. Obvious Tiles on Non-Portrait Subjects
- **BAD**: `--tile-style obvious` on architectural - looks like pixel art
- **GOOD**: `--tile-style subtle` for most subjects

### 4. Too Many Inference Steps
- **BAD**: `--inference-steps 30+` - loses the "undercooked" quality
- **GOOD**: `--inference-steps 10-20` - deliberately low for early diffusion feel

## Code Architecture

### Key Files
- `src/hyperspeed/eras/early_diffusion.py` - Main pipeline
- `src/hyperspeed/cli.py` - CLI with era-specific options

### CLI Options Added
```
--tile-size INT         Mosaic tile size in pixels
--tile-style TEXT       'obvious' or 'subtle'
--gold-style TEXT       'metallic' or 'symbolic'
--inference-steps INT   Diffusion steps (lower = more undercooked)
--guidance-scale FLOAT  CFG scale (higher = more oversaturated)

# img2img options
--source PATH           Source image for img2img transformation
--strength FLOAT        How much to transform source (0=keep original, 1=fully regenerate)

# Individual effect intensity overrides (0.0-1.0)
--hallucination FLOAT   Background face/eye hallucinations
--halo-bleed FLOAT      Boundary dissolution at halos
--edge-fizz FLOAT       Vibrating uncertain edges
--eye-drift FLOAT       Bilateral asymmetry in eyes
--almost-text FLOAT     Greek-like glyph artifacts
--finger-ambiguity FLOAT Hand/finger ghosting
```

### Effect Application Order
1. Tessellation (discrete color blocks)
2. Gold void (with background hallucinations)
3. Color bleeding
4. Halo bleed
5. Edge fizz
6. Eye drift
7. Almost-text
8. Finger ambiguity
9. Glimmer (final sparkle)

## Reference Images

### Best Results
- `galla_placidia_working.png` - Blue dome with symbolic gold, intensity 0.5
- `pantocrator_enhanced.png` - Creepy portrait with all v2 effects, intensity 0.6

### Iteration History
1. Initial implementation with basic tessellation, gold void, color bleeding, soft edges, glimmer
2. fp16 caused NaN at 768x768+ - switched to fp32
3. Tessellation too "pixel art" - added organic irregularity (variable sizes, row offsets, jitter)
4. User feedback: need more early-diffusion-specific artifacts
5. Added v2 effects: background hallucination, halo bleed, edge fizz, eye drift, almost-text, finger ambiguity
6. Geometric patterns overwhelmed at high intensity - recommend lower settings for abstract
7. v2 effects too subtle - boosted internal multipliers significantly (v3)
8. Added CLI flags for individual effect intensity control (--hallucination, --halo-bleed, etc.)
9. Tested isolated effects and found good combinations (hallucination 0.3 + finger-ambiguity 0.8)
10. Added true img2img support with StableDiffusionImg2ImgPipeline (--source, --strength)
11. **Mandala breakthrough**: img2img with reference template preserves composition; strength 0.75 + edge-fizz 1.0 + almost-text 1.0 is preferred method

## Fine-Tuning Individual Effects

Effects can now be controlled independently. Use high values (0.7-0.8) for one effect while setting others to 0.0 to isolate and test:

```bash
# Isolated hallucination test
hyperspeed generate "Christ Pantocrator..." --era early_diffusion --intensity 0.7 \
  --hallucination 0.8 --halo-bleed 0.0 --edge-fizz 0.0 --eye-drift 0.0 \
  --almost-text 0.0 --finger-ambiguity 0.0

# Combined hallucination + finger ambiguity (good combo)
hyperspeed generate "Christ Pantocrator..." --era early_diffusion --intensity 0.7 \
  --hallucination 0.3 --finger-ambiguity 0.8 --seed 2024 --upscale 2
```

### Effect Strength Notes

Internal multipliers were significantly boosted in v3 to make effects visible:

| Effect | Recommended Range | Notes |
|--------|-------------------|-------|
| hallucination | 0.3-0.5 | Higher creates obvious face shapes in gold |
| halo-bleed | 0.5-0.8 | Strong boundary dissolution |
| edge-fizz | 0.5-0.8 | Visible chromatic aberration at edges |
| eye-drift | 0.3-0.6 | Subtle asymmetry, higher = more obvious |
| almost-text | 0.5-0.8 | Many glyphs appear in borders |
| finger-ambiguity | 0.6-0.8 | Ghosting at hand regions |

### Best Combinations Found

- **Creepy Pantocrator**: hallucination 0.3 + finger-ambiguity 0.8
- **Dreamy dissolve**: halo-bleed 0.7 + edge-fizz 0.5
- **Byzantine inscriptions**: almost-text 0.8 + eye-drift 0.4

## Next Steps

- Consider "geometric" vs "figural" mode presets
- Explore ControlNet for architectural impossibility (future phase)
- Document more successful prompt/setting combinations

## Generation Time

At 768x768 with fp32, expect ~45-60 seconds per image on M1 Pro (slower than fp16 but stable).
