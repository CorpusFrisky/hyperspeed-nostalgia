# Neoclassicism (Photorealistic AI Era) Guide

## Overview

The Neoclassicism era maps to **Photorealistic AI (2023-2024)** - the moment when SDXL, Midjourney v5-v6, and DALL-E 3 achieved stunning realism, but developed a characteristic "too perfect" quality. Both the 18th-century masters and 2023-2024 AI pursued idealized beauty. The tell isn't failure; it's suspicious perfection.

**The key distinction from High Renaissance:** High Renaissance / MJ v4 was about *over-dramatization* (everything is epic). Neoclassicism is about *over-idealization* (everything is too perfect).

## Technical Foundation: SDXL (via Replicate or Local)

This era uses **Stable Diffusion XL (SDXL)** as its base model. By default, it uses the **Replicate API** for fast cloud generation (~30 seconds), with local fallback if the API is unavailable.

### Replicate API (Recommended - Fast)

Set your Replicate API token to enable fast cloud generation:

```bash
export REPLICATE_API_TOKEN="your-token-here"
```

Get a token at https://replicate.com/account/api-tokens

### Local Generation (Fallback)

If no `REPLICATE_API_TOKEN` is set, or if you use `--use-local`, generation runs locally:

```bash
# Force local generation
hyperspeed generate "Neoclassical portrait" --era neoclassicism --use-local
```

## The Aesthetic Goal

We're mapping **Photorealistic AI (2023-2024)** onto **Neoclassicism (1760-1850)**. The connection:

- David's idealized bodies = AI's suspiciously attractive subjects
- Ingres's marble-smooth skin = AI's over-smoothed faces
- Neoclassical restraint = The muted "stock photo" palette
- Theatrical staging = Center-weighted compositions
- Anatomical liberty (extra vertebrae) = Idealized proportions

Key works to reference:
- Jacques-Louis David's Oath of the Horatii (1784) - idealized bodies, theatrical staging
- Ingres's La Grande Odalisque (1814) - anatomically "incorrect" but aesthetically smooth
- Antonio Canova's marble sculptures - that impossible smoothness

## What Works

### Technical Settings

```bash
# For suspiciously perfect portraits
hyperspeed generate "Neoclassical portrait, young woman, classical beauty" \
  --era neoclassicism --intensity 0.7 --marble-smooth-skin 0.8

# For theatrical historical scenes (David-style)
hyperspeed generate "Oath scene, heroic figures, dramatic gesture, Neoclassical painting" \
  --era neoclassicism --intensity 0.6 --heroic-staging 0.5

# For Ingres-style idealized figures
hyperspeed generate "Reclining odalisque, elegant proportions, marble skin" \
  --era neoclassicism --intensity 0.7 --anatomical-liberty 0.6
```

### Intensity Sweet Spots

| Subject | Intensity | Notes |
|---------|-----------|-------|
| Portraits | 0.6-0.8 | Full Neoclassicism treatment works well |
| Historical scenes | 0.5-0.7 | Theatrical staging shines |
| Figure studies | 0.6-0.8 | Anatomical liberty + smooth skin |
| Group compositions | 0.5-0.6 | Keep subtle to avoid uncanny valley |

### Individual Effect Controls

Each tell can be controlled independently:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--marble-smooth-skin` | 0.7 | THE signature effect - frequency-selective skin smoothing |
| `--golden-ratio-symmetry` | 0.4 | Facial symmetry enhancement (keep subtle) |
| `--surface-flattening` | 0.5 | Non-skin texture flattening ("stock photo" clean) |
| `--heroic-staging` | 0.3 | Compositional centering (very subtle) |
| `--anatomical-liberty` | 0.5 | Ingres-style proportion distortion |
| `--classical-palette` | 0.6 | Muted colors: warm skin, cool backgrounds |

## Example Commands

### Marble-Smooth Skin (THE Signature)

This is THE most recognizable photorealistic AI tell. Skin that's too smooth, too even, but still has just enough detail to not look obviously fake.

```bash
hyperspeed generate "Portrait of a young woman, neoclassical style, classical beauty, marble bust aesthetic" \
  --era neoclassicism \
  --intensity 0.7 \
  --marble-smooth-skin 0.9 \
  --golden-ratio-symmetry 0.5 \
  --seed 1814 \
  --output examples/neoclassicism_portrait.png
```

**Key elements:**
- Prompt should mention "classical," "marble," "perfect"
- `--marble-smooth-skin 0.8-0.9` for obvious effect
- Works especially well on portraits and figure studies

**Variations that work:**
- "Portrait in the style of Ingres, smooth complexion"
- "Classical beauty, Renaissance revival portrait"
- "Neoclassical bust, perfect features"

### Golden Ratio Symmetry (Suspiciously Attractive)

The subtle facial symmetry that makes everyone look like they had expensive work done.

```bash
hyperspeed generate "Portrait of a classical muse, idealized features, perfect proportions" \
  --era neoclassicism \
  --intensity 0.6 \
  --golden-ratio-symmetry 0.6 \
  --marble-smooth-skin 0.7 \
  --seed 1784 \
  --output examples/neoclassicism_symmetry.png
```

**Key elements:**
- Keep strength moderate (0.4-0.6) to avoid uncanny valley
- Combine with marble-smooth skin for full effect
- "Perfect proportions" in prompts helps

### Heroic Staging (David-Style Theatrical)

The "everyone is positioned on a stage" quality of Neoclassical composition.

```bash
hyperspeed generate "Neoclassical tableau, heroic oath scene, dramatic gestures, multiple figures" \
  --era neoclassicism \
  --intensity 0.6 \
  --heroic-staging 0.5 \
  --classical-palette 0.7 \
  --seed 1785 \
  --output examples/neoclassicism_staging.png
```

**Key elements:**
- "Heroic," "oath," "dramatic gesture" in prompts
- Keep staging subtle (0.3-0.5) - it's about composition, not epicness
- Combine with classical palette for period feel

### Anatomical Liberty (Ingres's Extra Vertebrae)

Subtle proportion distortions that are "aesthetically pleasing but anatomically wrong."

```bash
hyperspeed generate "Reclining figure, odalisque style, elegant proportions, idealized nude" \
  --era neoclassicism \
  --intensity 0.7 \
  --anatomical-liberty 0.7 \
  --marble-smooth-skin 0.8 \
  --seed 1814 \
  --output examples/neoclassicism_odalisque.png
```

**Key elements:**
- "Elongated," "elegant," "idealized" in prompts
- Works best on figure studies and reclining poses
- Effect should be subtle - like Ingres, you don't notice until you count

### Classical Palette (Muted Stock Photo)

The desaturated, restrained color harmony that says "expensive but boring."

```bash
hyperspeed generate "Neoclassical history painting, muted palette, warm figures against cool background" \
  --era neoclassicism \
  --intensity 0.6 \
  --classical-palette 0.8 \
  --surface-flattening 0.6 \
  --seed 1793 \
  --output examples/neoclassicism_palette.png
```

**Key elements:**
- Warm skin tones, cool backgrounds
- Reduced saturation (not the HDR of MJ v4)
- "Muted," "restrained," "classical" in prompts

### The Full Neoclassicism Treatment

Combining all effects for maximum "photorealistic AI" energy.

```bash
hyperspeed generate "Neoclassical portrait of a young noblewoman, marble-smooth skin, perfect proportions, classical drapery, muted palette" \
  --era neoclassicism \
  --intensity 0.7 \
  --marble-smooth-skin 0.8 \
  --golden-ratio-symmetry 0.5 \
  --anatomical-liberty 0.5 \
  --classical-palette 0.7 \
  --heroic-staging 0.3 \
  --seed 1800 \
  --output examples/neoclassicism_full.png
```

## The Art Historical Parallel

### Why This Mapping Works

Neoclassicism (1760-1850) and photorealistic AI (2023-2024) share a crucial quality: **suspicious perfection**. Both achieved remarkable technical prowess, but developed tells that make them instantly recognizable.

**Neoclassical masters:**
- David pursued idealized classical forms
- Ingres bent anatomy for aesthetic effect (three extra vertebrae)
- Canova made marble look impossibly smooth
- All three created a "look" that was beautiful but somehow not quite real

**Photorealistic AI:**
- SDXL, MJ v5-v6, DALL-E 3 achieved stunning realism
- Everyone became suspiciously attractive
- Surfaces became too clean (the "stock photo" effect)
- Proportions became idealized rather than realistic

### The Distinction from Other Eras

| Era | AI Parallel | Technique Focus |
|-----|-------------|-----------------|
| Early Renaissance | DALL-E 2 / SD 1.x | **Destabilization** (errors, failures) |
| International Gothic | StyleGAN | **Over-polishing** (uncanny smoothness) |
| High Renaissance | Midjourney v4 | **Over-dramatization** (everything epic) |
| Neoclassicism | Photorealistic AI | **Over-idealization** (everything perfect) |

Neoclassicism / Photorealistic AI is recognizable not because it fails, but because it **tries too hard to be perfect**. Every face is symmetrical. Every surface is clean. Every proportion is idealized.

### Deep Cuts

- Ingres's La Grande Odalisque has three extra vertebrae - he bent anatomy for aesthetic effect, just like AI models idealize proportions
- David's Oath of the Horatii positions figures like actors on a stage - AI similarly tends toward centered, theatrical compositions
- Neoclassical sculptors like Canova achieved a marble smoothness that no real skin has - the same "too smooth" quality we see in photorealistic AI
- The Neoclassical color palette (warm flesh against cool backgrounds) maps perfectly to the desaturated "stock photo" aesthetic of 2023-2024 AI

## Prompts That Work Well

**For marble-smooth skin:**
- "Neoclassical portrait, marble bust aesthetic"
- "Classical beauty, perfect complexion"
- "Portrait in the style of Ingres"

**For golden ratio symmetry:**
- "Perfect profile portrait, classical proportions"
- "Idealized figure study, academic tradition"
- "Portrait of a classical muse"

**For heroic staging:**
- "Oath of the Horatii composition, dramatic gesture"
- "Neoclassical tableau, heroic pose"
- "David-style historical scene"

**For anatomical liberty:**
- "Reclining figure, elegant proportions, odalisque style"
- "Elongated figure, Ingres aesthetic"
- "Idealized nude, classical proportions"

**For classical palette:**
- "Muted Neoclassical colors, warm and cool contrast"
- "Restrained palette, academic painting"
- "Classical color harmony"

---

## Batch Generation

Like High Renaissance, Neoclassicism supports batch generation via Replicate:

```bash
# Create a jobs.json file:
# [
#   {
#     "prompt": "Neoclassical portrait...",
#     "output": "neoclassicism_portrait_1.png",
#     "seed": 1814,
#     "intensity": 0.7,
#     "marble_smooth_skin": 0.8
#   },
#   ...
# ]

# Run batch (uses high_renaissance batch infrastructure)
# Note: For Neoclassicism batch, use the pipeline's batch_generate_replicate function directly
```

---

## Session Progress Log

### Initial Implementation: 2025-12-09

**Effects implemented:**
- `_apply_marble_smooth_skin()` - frequency-selective skin smoothing
- `_apply_golden_ratio_symmetry()` - facial symmetry enhancement
- `_apply_surface_flattening()` - non-skin texture reduction
- `_apply_heroic_staging()` - subtle compositional centering
- `_apply_anatomical_liberty()` - Ingres-style proportion distortion
- `_apply_classical_palette()` - muted Neoclassical colors

**Technical notes:**
- Uses SDXL via Replicate (same as High Renaissance)
- Lower guidance scale (8.0 vs 10.0) for more natural results
- Effect application order: palette -> skin -> face -> body -> textures -> composition
