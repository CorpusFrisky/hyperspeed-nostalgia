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
| `--marble-smooth-skin` | 0.85 | THE signature - poreless perfection |
| `--golden-ratio-symmetry` | 0.5 | Facial symmetry enhancement |
| `--surface-flattening` | 0.7 | Non-skin texture flattening ("stock photo" clean) |
| `--heroic-staging` | 0.4 | Compositional centering (theatrical staging) |
| `--anatomical-liberty` | 0.6 | Ingres-style proportion distortion |
| `--classical-palette` | 0.7 | Muted colors: warm skin, cool backgrounds |
| `--commercial-sheen` | 0.7 | Stock photo lighting (fill light, specular, lifted blacks) |
| `--emotional-vacancy` | 0.6 | Flatten expressions (the "can't do solemn" tell) |

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

## Reproducible Example Commands

All generated images with their exact commands for reproduction.

### David-Style History Paintings

#### Death of Socrates (Multi-Figure Death Scene)

```bash
hyperspeed generate "Death of Socrates, Neoclassical history painting, multiple figures gathered around dying philosopher, disciples in dramatic poses, gesturing, stoic resignation, David style, muted palette, classical drapery" \
  --era neoclassicism \
  --intensity 0.85 \
  --marble-smooth-skin 0.9 \
  --emotional-vacancy 0.8 \
  --commercial-sheen 0.8 \
  --seed 399 \
  --output examples/neoclassicism_death_of_socrates.png
```

**Why this works:** Multi-figure composition with emotionally demanding subject (death, grief). The `--emotional-vacancy 0.8` exposes the AI's inability to convey gravitas - the disciples should look solemn but instead look vacant.

#### Brutus Receiving the Bodies of His Sons

```bash
hyperspeed generate "Brutus receiving the bodies of his sons, Neoclassical history painting, multiple mourning figures, Roman soldiers, stoic father, tragic scene, David style, theatrical staging, muted palette" \
  --era neoclassicism \
  --intensity 0.85 \
  --marble-smooth-skin 0.9 \
  --emotional-vacancy 0.8 \
  --heroic-staging 0.6 \
  --seed 509 \
  --output examples/neoclassicism_brutus.png
```

**Why this works:** Tests stoic suffering - a father who has condemned his own sons should show complex emotion. The AI produces technically perfect but emotionally hollow results.

#### Oath of the Horatii

```bash
hyperspeed generate "Oath of the Horatii, Neoclassical history painting, three brothers with raised arms, dramatic oath gesture, father with swords, women mourning in background, David style, theatrical lighting" \
  --era neoclassicism \
  --intensity 0.85 \
  --marble-smooth-skin 0.9 \
  --emotional-vacancy 0.7 \
  --heroic-staging 0.6 \
  --seed 1784 \
  --output examples/neoclassicism_horatii.png
```

**Why this works:** The iconic Neoclassical composition. The raised-arm gesture should convey patriotic fervor; the AI produces the pose but not the conviction.

#### Death of Marat

```bash
hyperspeed generate "Death of Marat, Neoclassical painting, murdered figure in bathtub, dramatic death scene, martyr pose, single figure, muted palette, David style, stoic suffering" \
  --era neoclassicism \
  --intensity 0.85 \
  --marble-smooth-skin 0.95 \
  --emotional-vacancy 0.8 \
  --commercial-sheen 0.7 \
  --seed 1793 \
  --output examples/neoclassicism_marat.png
```

**Why this works:** Single-figure death scene. The marble-smooth skin at 0.95 creates that "too perfect corpse" quality - death rendered as stock photo.

#### Hector and Andromache Farewell

```bash
hyperspeed generate "Hector and Andromache farewell, Neoclassical painting, tragic parting, warrior leaving wife, Trojan scene, emotional moment, tender gesture, classical drapery, muted palette" \
  --era neoclassicism \
  --intensity 0.85 \
  --marble-smooth-skin 0.9 \
  --emotional-vacancy 0.85 \
  --classical-palette 0.8 \
  --seed 1812 \
  --output examples/neoclassicism_hector_andromache.png
```

**Why this works:** A farewell scene that should convey the weight of impending death. High emotional vacancy (0.85) reveals the "lights on but nobody home" quality.

#### Mourning Scene

```bash
hyperspeed generate "Neoclassical mourning scene, multiple grieving figures around tomb, stoic sorrow, classical drapery, muted colors, theatrical staging, David style history painting" \
  --era neoclassicism \
  --intensity 0.85 \
  --emotional-vacancy 0.85 \
  --marble-smooth-skin 0.9 \
  --heroic-staging 0.5 \
  --seed 1805 \
  --output examples/neoclassicism_mourning.png
```

**Why this works:** Grief is one of the hardest emotions for AI to convey. The figures look posed rather than devastated.

### Canova-Style Sculptures

#### Psyche Revived by Cupid's Kiss

```bash
hyperspeed generate "Psyche Revived by Cupid's Kiss, Canova style marble sculpture, two intertwined figures, wings, romantic embrace, gleaming white marble, museum photography, commercial lighting" \
  --era neoclassicism \
  --intensity 0.85 \
  --marble-smooth-skin 0.95 \
  --commercial-sheen 0.9 \
  --surface-flattening 0.8 \
  --seed 1787 \
  --output examples/neoclassicism_psyche_cupid.png
```

**Why this works:** Intertwined figures in marble. The extreme marble-smooth-skin (0.95) + commercial-sheen (0.9) creates that CGI-waxy quality that no real marble has.

#### The Three Graces

```bash
hyperspeed generate "The Three Graces, Canova style marble sculpture group, three female figures intertwined, classical poses, gleaming white marble, museum lighting, commercial photography aesthetic" \
  --era neoclassicism \
  --intensity 0.85 \
  --marble-smooth-skin 0.95 \
  --commercial-sheen 0.9 \
  --golden-ratio-symmetry 0.6 \
  --seed 1813 \
  --output examples/neoclassicism_three_graces.png
```

**Why this works:** Three figures with "suspiciously attractive" symmetry. The golden-ratio-symmetry at 0.6 makes each face uncannily perfect.

#### Perseus with the Head of Medusa

```bash
hyperspeed generate "Perseus with head of Medusa, Canova style marble sculpture, heroic male figure, classical pose, gleaming white marble, dramatic museum lighting, theatrical spotlight" \
  --era neoclassicism \
  --intensity 0.85 \
  --marble-smooth-skin 0.95 \
  --commercial-sheen 0.85 \
  --heroic-staging 0.5 \
  --seed 1801 \
  --output examples/neoclassicism_perseus.png
```

**Why this works:** Heroic male figure - tests whether the "too perfect" effect works on male subjects. The theatrical lighting creates that prestige documentary feel.

### Stock Photo Aesthetic

#### Commercial Beauty Portrait

```bash
hyperspeed generate "Neoclassical portrait of noble woman, suspiciously perfect features, flawless skin, commercial beauty photography lighting, stock photo aesthetic, classical drapery, muted palette" \
  --era neoclassicism \
  --intensity 0.9 \
  --marble-smooth-skin 0.95 \
  --commercial-sheen 0.95 \
  --golden-ratio-symmetry 0.7 \
  --seed 2023 \
  --output examples/neoclassicism_stock_photo_portrait.png
```

**Why this works:** Maximum "stock photo" energy. All the tells cranked to near-maximum. The seed 2023 is a deliberate nod to the era being mapped.

### Early Examples

#### Basic Portrait

```bash
hyperspeed generate "Neoclassical portrait, young woman, classical beauty, marble bust aesthetic" \
  --era neoclassicism \
  --intensity 0.7 \
  --marble-smooth-skin 0.8 \
  --seed 1814 \
  --output examples/neoclassicism_portrait.png
```

#### Oath Scene

```bash
hyperspeed generate "Neoclassical oath scene, heroic figures, dramatic gestures, David style" \
  --era neoclassicism \
  --intensity 0.7 \
  --heroic-staging 0.5 \
  --seed 1784 \
  --output examples/neoclassicism_oath.png
```

#### Odalisque

```bash
hyperspeed generate "Reclining odalisque, Ingres style, elegant proportions, classical beauty, draped fabric" \
  --era neoclassicism \
  --intensity 0.7 \
  --anatomical-liberty 0.7 \
  --marble-smooth-skin 0.8 \
  --seed 1814 \
  --output examples/neoclassicism_odalisque.png
```

---

## Session Progress Log

### Update: 2025-12-09 - Pushed Effects Harder

Based on feedback that artifacts were too subtle, pushed all effects significantly:

**New effects added:**
- `_apply_commercial_sheen()` - stock photo lighting (fill light, specular highlights, lifted blacks)
- `_apply_emotional_vacancy()` - flatten expression areas (the "can't do solemn" tell)

**Intensified existing effects:**
- `marble_smooth_skin`: 0.7 -> 0.85 (poreless perfection)
- `golden_ratio_symmetry`: 0.4 -> 0.5
- `surface_flattening`: 0.5 -> 0.7 (stock photo clean)
- `heroic_staging`: 0.3 -> 0.4

**New batch subjects tested:**
- Multi-figure compositions (Death of Socrates, Brutus, Horatii)
- Emotionally demanding scenes (Marat, Hector/Andromache farewell, mourning)
- Canova sculpture groups (Psyche/Cupid, Three Graces, Perseus)
- Stock photo aesthetic (commercial lighting portrait)

**Key insight:** The emotional vacancy effect exposes the AI's inability to convey gravitas. Subjects that demand solemnity (death scenes, farewells) reveal the "lights on but nobody home" quality.

**Generated images:**
| File | Subject | Seed |
|------|---------|------|
| `neoclassicism_death_of_socrates.png` | Multi-figure death scene | 399 |
| `neoclassicism_brutus.png` | Receiving bodies of sons | 509 |
| `neoclassicism_horatii.png` | Oath with raised arms | 1784 |
| `neoclassicism_marat.png` | Death in bathtub | 1793 |
| `neoclassicism_hector_andromache.png` | Farewell scene | 1812 |
| `neoclassicism_psyche_cupid.png` | Canova sculpture pair | 1787 |
| `neoclassicism_three_graces.png` | Intertwined figures | 1813 |
| `neoclassicism_stock_photo_portrait.png` | Commercial beauty | 2023 |
| `neoclassicism_mourning.png` | Grieving figures | 1805 |
| `neoclassicism_perseus.png` | Heroic male sculpture | 1801 |

---

### Initial Implementation: 2025-12-09

**Effects implemented:**
- `_apply_marble_smooth_skin()` - aggressive skin smoothing (poreless perfection)
- `_apply_golden_ratio_symmetry()` - facial symmetry enhancement
- `_apply_surface_flattening()` - non-skin texture reduction
- `_apply_heroic_staging()` - compositional centering
- `_apply_anatomical_liberty()` - Ingres-style proportion distortion
- `_apply_classical_palette()` - muted Neoclassical colors
- `_apply_commercial_sheen()` - stock photo lighting
- `_apply_emotional_vacancy()` - flatten expressions

**Technical notes:**
- Uses SDXL via Replicate (same as High Renaissance)
- Lower guidance scale (8.0 vs 10.0) for more natural results
- Effect application order: palette -> skin -> face -> emotional_vacancy -> body -> textures -> commercial_sheen -> composition
