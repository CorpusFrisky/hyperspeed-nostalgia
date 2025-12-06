# Cave Paintings Era - DeepDream Generation Guide

## Overview

This guide documents how to generate high-quality "Cave Paintings" era images using the DeepDream pipeline. The goal is to create images that evoke Paleolithic cave art (Lascaux, Chauvet) while incorporating DeepDream's characteristic pareidolia and texture amplification.

## The Aesthetic Goal

We're mapping **DeepDream (2015)** onto **Cave Paintings (35,000-10,000 BCE)**. The connection:
- DeepDream's pareidolia (seeing faces/animals in textures) mirrors the prehistoric artist's ability to see animals in cave wall contours
- Both represent a "first contact" moment - humans learning to see/create with a new visual technology
- The textured, emergent quality of DeepDream artifacts feels ancient and organic

## Key Reference Images

Before generating, review these reference images in `examples/`:

1. **`cave_menagerie_large.png`** - The gold standard. Shows:
   - Pale cream/tan limestone background
   - Red ochre and sienna animal silhouettes
   - Large herd of small animals scattered across frame
   - DeepDream texture that feels like mineral deposits on stone

2. **`cave_menagerie_2.png`** - Iridescent/pearlescent variation with rainbow highlights

3. **`lascaux_dreamed.png`** - Good composition reference: two beasts entering from sides, open center

4. **`cave_horses_hands.png`** - Shows ochre handprints and cave texture

## Successful Prompts

### For Herd Compositions (Recommended)

```bash
hyperspeed generate "prehistoric cave painting, large herd of small horses and aurochs scattered across pale cream limestone wall, light tan beige stone background, red ochre and brown pigment animals, Lascaux Chauvet style, many small animal silhouettes, warm earth tones" \
  --era deepdream \
  --intensity 0.7 \
  --width 1024 \
  --height 768 \
  --upscale 2 \
  --output examples/cave_herd_output.png
```

### For Iridescent/Pearlescent Variation

```bash
hyperspeed generate "prehistoric cave painting, large herd of small horses and aurochs scattered across limestone wall, orange amber brown pigments, iridescent pearlescent stone surface, many small animal silhouettes, Lascaux Chauvet style, rainbow highlights on rough textured stone" \
  --era deepdream \
  --intensity 0.7 \
  --width 1024 \
  --height 768 \
  --upscale 2 \
  --output examples/cave_iridescent_output.png
```

### For Two-Beast Composition (Animals from Sides)

```bash
hyperspeed generate "prehistoric cave painting on rough limestone, two large beasts entering frame from far left and right edges, partial animals mostly cropped off frame, open empty center, red ochre sienna brown earth pigments, Lascaux Chauvet style, textured stone wall" \
  --era deepdream \
  --intensity 0.5 \
  --width 768 \
  --height 512 \
  --upscale 2 \
  --output examples/cave_two_beasts.png
```

## Critical Parameters

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| `--era` | `deepdream` | Required - uses InceptionV3 gradient ascent |
| `--intensity` | `0.5 - 0.7` | 0.4 is subtle, 0.7 is strong DeepDream texture |
| `--width` | `768` or `1024` | Base resolution before upscale |
| `--height` | `512` or `768` | Landscape orientation works best |
| `--upscale` | `2` | 2x upscale after DeepDream processing |

## Common Mistakes to Avoid

### 1. Wrong Color Palette
- **BAD**: "orange amber" - produces too-saturated orange animals
- **GOOD**: "red ochre and brown pigment" - produces authentic earth tones

### 2. Dark Background
- **BAD**: Not specifying background color
- **GOOD**: "pale cream limestone wall, light tan beige stone background"

### 3. Too Detailed/Photorealistic
- **BAD**: "detailed horse face, realistic anatomy"
- **GOOD**: "silhouettes, Lascaux Chauvet style" - keeps it abstract

### 4. Wrong Composition Density
- **BAD**: "single horse" - too sparse, loses cave painting feel
- **GOOD**: "large herd of small horses and aurochs scattered across" - creates the menagerie effect

### 5. Too Much DeepDream
- **BAD**: `--intensity 1.0` - becomes psychedelic noise
- **GOOD**: `--intensity 0.5-0.7` - texture enhances rather than overwhelms

## The Prompt Formula

```
"prehistoric cave painting, [COMPOSITION], [BACKGROUND], [COLORS], [STYLE], [TEXTURE]"
```

Where:
- **COMPOSITION**: "large herd of small horses and aurochs scattered across" OR "two large beasts entering from sides"
- **BACKGROUND**: "pale cream limestone wall, light tan beige stone background"
- **COLORS**: "red ochre and brown pigment animals, warm earth tones"
- **STYLE**: "Lascaux Chauvet style, many small animal silhouettes"
- **TEXTURE**: (optional) "rough textured stone" or "iridescent pearlescent stone surface"

## Generation Time

At 1024x768 with 2x upscale, expect ~18-20 minutes per image on M1 Pro.

## Iteration Strategy

1. Start with a smaller test (512x384, no upscale) to check composition
2. If composition works, run full resolution (1024x768, 2x upscale)
3. Adjust prompt keywords based on what's wrong:
   - Too orange → add "red ochre", remove "orange"
   - Too dark → add "pale cream", "light tan beige"
   - Too detailed → add "silhouettes", reduce intensity
   - Too sparse → add "large herd", "many small"

## Files Generated in This Session

- `cave_large_herd_ochre.png` - Large herd with correct ochre palette on pale limestone
- `cave_large_herd_iridescent.png` - Iridescent/pearlescent variation
- `cave_large_herd_small_animals.png` - Good herd composition
- `cave_beasts_from_sides.png` - Two beasts entering from edges
- `cave_beasts_cropped_open.png` - Open center composition
