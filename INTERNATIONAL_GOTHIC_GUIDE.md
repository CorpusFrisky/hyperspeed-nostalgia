# International Gothic (StyleGAN Era) Guide

## Overview

The International Gothic era maps to **StyleGAN "This Person Does Not Exist" (2019-2021)** - faces that are suspiciously beautiful, technically proficient but emotionally vacant. The elegant uncanny valley where porcelain-smooth skin meets photographic sharpness on everything else.

## The Aesthetic Goal

We're mapping **StyleGAN's uncanny faces** onto **International Gothic panel paintings (1375-1425)**. The connection:
- Faces technically accomplished but emotionally vacant = StyleGAN's latent space interpolation
- Same face appearing across multiple figures = variations on a seed
- Jewelry rendered with more conviction than faces = GAN training data bias
- Gold void backgrounds = latent space running out of data

Key works to reference:
- Simone Martini's Annunciation (1333) - Virgin's face smooth as latent space
- Gentile da Fabriano's Adoration of the Magi (1423) - crowd faces like seed variations
- The Wilton Diptych (c. 1395-1399) - eleven angels with nearly identical faces

## What Works

### Technical Settings

```bash
# Standard portrait
hyperspeed generate "noble lady" --era international_gothic --intensity 0.6 --width 512 --height 768

# Higher uncanny valley
hyperspeed generate "angel portrait" --era international_gothic --intensity 0.8 \
    --porcelain-smoothness 0.9 --eye-tracking-error 0.6

# More naturalistic
hyperspeed generate "merchant portrait" --era international_gothic --intensity 0.4 \
    --gold-background 0.3
```

### Intensity Sweet Spots

| Subject | Intensity | Notes |
|---------|-----------|-------|
| Single portrait | 0.5-0.7 | Good balance of effects |
| Religious figure | 0.7-0.9 | Push the uncanny, more gold |
| Courtly scene | 0.4-0.6 | Keep some naturalism |

### Prompt Structure

The pipeline auto-enhances prompts with International Gothic styling, so keep prompts simple:

```
"[subject]"
```

Examples that work:
- `"noble lady"` - aristocratic female portrait
- `"angel portrait"` - religious, good for gold background
- `"young prince"` - courtly male portrait
- `"Virgin Mary"` - classic International Gothic subject

The pipeline adds: "International Gothic style painting, medieval courtly costume, ornate gold jewelry, rich blue and red robes, gold leaf background, tempera painting on wood panel, Simone Martini style, 1400s Italian art"

## Artifact Effects

### StyleGAN Simulation (5 effects)

| Effect | CLI Flag | Default | Description |
|--------|----------|---------|-------------|
| Porcelain smoothness | `--porcelain-smoothness` | 0.7 | Over-smooth skin, plastic sheen |
| Hair bleeding | `--hair-bleeding` | 0.6 | Hair dissolves into background |
| Eye tracking error | `--eye-tracking-error` | 0.4 | Eyes don't quite align |
| Asymmetric accessories | `--asymmetric-accessories` | 0.5 | Earrings/jewelry don't match |
| Background disconnect | `--background-disconnect` | 0.5 | Sharp figure, confused background |

### International Gothic Stylization (5 effects)

| Effect | CLI Flag | Default | Description |
|--------|----------|---------|-------------|
| Gold background | `--gold-background` | 0.5 | Gold leaf void behind figure |
| Gold style | `--gold-style` | metallic | "metallic" or "symbolic" |
| Tempera texture | `--tempera-texture` | 0.4 | Egg tempera surface quality |
| Courtly palette | `--courtly-palette` | 0.6 | Rich blues, reds, desaturated skin |
| Decorative precision | `--decorative-precision` | 0.5 | Jewelry sharper than faces |
| Emotional vacancy | `--emotional-vacancy` | 0.4 | Flatten expression areas |

### Final Processing

| Effect | Description |
|--------|-------------|
| StyleGAN sharpness | Restores photographic crispness to non-skin areas |

## Effect Combinations

### "This Person Does Not Exist" Maximum
Push all StyleGAN tells:
```bash
hyperspeed generate "portrait" --era international_gothic --intensity 0.9 \
    --porcelain-smoothness 0.9 \
    --eye-tracking-error 0.7 \
    --asymmetric-accessories 0.8 \
    --background-disconnect 0.7
```

### Simone Martini Style
Religious icon with gold and vacancy:
```bash
hyperspeed generate "Virgin Mary" --era international_gothic --intensity 0.7 \
    --gold-background 0.9 \
    --gold-style metallic \
    --emotional-vacancy 0.7 \
    --courtly-palette 0.8
```

### Courtly Portrait
More naturalistic but still uncanny:
```bash
hyperspeed generate "young nobleman" --era international_gothic --intensity 0.5 \
    --porcelain-smoothness 0.6 \
    --gold-background 0.3 \
    --decorative-precision 0.7
```

## Common Mistakes to Avoid

### 1. Too Much Smoothness Everywhere
- **BAD**: High `--porcelain-smoothness` bleeds into clothing
- **GOOD**: The sharpness pass now preserves crisp clothing edges

### 2. Over-saturated Colors
- **BAD**: Default SD colors are too vibrant for International Gothic
- **GOOD**: `--courtly-palette` desaturates skin, shifts costume toward period colors

### 3. Background Too Busy
- **BAD**: Complex backgrounds fight with gold leaf effect
- **GOOD**: Gold background works best with simple prompts that let it dominate

### 4. Too Subtle Effects
- **BAD**: Intensity 0.3 - effects barely visible
- **GOOD**: Intensity 0.5-0.7 for visible uncanny valley

## Code Architecture

### Key Files
- `src/hyperspeed/eras/international_gothic.py` - Main pipeline
- `src/hyperspeed/cli.py` - CLI with era-specific options

### Effect Application Order
1. SD generation with International Gothic prompt enhancement
2. Porcelain smoothness (skin-targeted bilateral filter)
3. Hair bleeding (edge gradient dissolution)
4. Eye tracking error (subtle bilateral asymmetry)
5. Asymmetric accessories (left/right different distortions)
6. Background disconnect (sharp figure, confused background)
7. Gold background (metallic or symbolic)
8. Tempera texture (fine grain, warm cast)
9. Courtly palette (period-appropriate colors)
10. Decorative precision (sharpen jewelry, soften face)
11. Emotional vacancy (flatten expression midtones)
12. StyleGAN sharpness (final photographic crispness)

## Reference Images

### Best Results
- `gothic_noble_lady_v2.png` - First working portrait with sharpness fix
- `gothic_angel_1.png` through `gothic_angel_4.png` - "Wilton Diptych" style angel set (seeds 2024-2027)

### Iteration History
1. Initial implementation with 10 effects
2. User feedback: clothing edges too soft (diffusion vs StyleGAN)
3. Added `_apply_stylegan_sharpness()` - skin-aware sharpening restores photographic crispness

## Comparison: International Gothic vs Early GAN

Both eras simulate GAN artifacts, but with different aesthetics:

| Aspect | Early GAN (Egyptian) | International Gothic |
|--------|---------------------|---------------------|
| Art parallel | Egyptian Canon | Gothic panel painting |
| Face quality | Formulaic, rigid | Beautiful, vacant |
| Background | Void/bleeding | Gold leaf |
| Color | Clinical, desaturated | Rich courtly palette |
| Overall feel | "This Person Does Not Exist" raw | TPDNE styled as Simone Martini |

## Generation Time

At 512x768 with fp32, expect ~45-50 seconds per image on M1 Pro.
