# Early Renaissance (2022 Diffusion Era) Guide

## Overview

The Early Renaissance era maps to **DALL-E 2 / Stable Diffusion 1.x (2022)** - the year of ambitious failures. Both the 15th-century painters and 2022 diffusion models were reaching for realism but failing in characteristic ways. Hands are famously hard. Perspective errors abound. The ambition exceeds the capability.

## The Aesthetic Goal

We're mapping **DALL-E 2 / SD 1.x (2022)** onto **Early Renaissance (1400-1490)**. The connection:
- Renaissance painters struggled with hands = the iconic six-fingered hand artifact
- Masaccio's perspective experiments = spatial contradictions in AI-generated scenes
- Mantegna's foreshortening = depth that doesn't recede properly
- Bodies that shift scale = proportion inconsistencies across the image

Key works to reference:
- Masaccio's Brancacci Chapel frescoes (1425-1428) - ambitious spatial staging, early perspective
- Mantegna's Dead Christ (c. 1480) - extreme foreshortening that doesn't quite work
- Masaccio's The Tribute Money - multiple vanishing points
- Durer's grid drawings - the "drawing machine" as Renaissance training data augmentation

## What Works

### Technical Settings

```bash
# For figure studies with hand emphasis
hyperspeed generate "Renaissance figure study, hands visible, classical composition" \
  --era early_renaissance --intensity 0.6 --hand-failure 0.8

# For architectural/spatial scenes
hyperspeed generate "Renaissance interior, tiled floor, columns receding" \
  --era early_renaissance --intensity 0.5 --perspective-error 0.7

# For portraits with proportion issues
hyperspeed generate "Renaissance portrait, three-quarter view" \
  --era early_renaissance --intensity 0.5 --proportion-shift 0.6
```

### Intensity Sweet Spots

| Subject | Intensity | Notes |
|---------|-----------|-------|
| Hand studies | 0.6-0.8 | Higher = more obvious six-finger effect |
| Architectural | 0.4-0.6 | Lower preserves some spatial coherence |
| Portraits | 0.5-0.6 | Subtle proportion shifts feel most "2022" |
| Full figures | 0.5-0.7 | Combines hand + proportion effects |

### Individual Effect Controls

Each artifact can be controlled independently:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--hand-failure` | 0.7 | Hand anatomical errors (six fingers, fused digits) |
| `--foreshortening-error` | 0.6 | Depth representation failures |
| `--perspective-error` | 0.5 | Spatial contradictions |
| `--proportion-shift` | 0.4 | Scale variation across composition |
| `--edge-ambiguity` | 0.4 | Figure-ground integration issues |
| `--wooden-face` | 0.6 | Stiff mannequin-like faces, misaligned eyes |

## Example Commands

### The Six-Fingered Hand (Iconic)
```bash
hyperspeed generate "Renaissance figure reaching toward viewer, hands prominent" \
  --era early_renaissance \
  --intensity 0.7 \
  --hand-failure 0.9 \
  --foreshortening-error 0.3 \
  --output examples/hand_study.png
```

### Mantegna's Dead Christ (Foreshortening) - PROVEN EFFECTIVE

This approach reliably produces the characteristic "depth that doesn't recede properly" effect.

```bash
hyperspeed generate "recumbent figure lying down, feet toward viewer, dramatic foreshortening, Renaissance painting" \
  --era early_renaissance \
  --intensity 0.6 \
  --foreshortening-error 0.8 \
  --hand-failure 0.3 \
  --seed 1480 \
  --output examples/foreshortening_study.png
```

**Key elements:**
- Prompt should explicitly mention "lying down" and "feet toward viewer"
- "dramatic foreshortening" helps guide the composition
- Keep `--hand-failure` low (0.3) to let foreshortening dominate
- `--foreshortening-error 0.8` creates strong depth contradictions
- `--intensity 0.6` balances effect visibility with image coherence

**Variations that work:**
- "figure reclining on a bed, viewed from the feet"
- "sleeping figure, extreme perspective, feet in foreground"
- "body laid out, foreshortened view from below"

### Brancacci Chapel (Spatial Contradiction) - PROVEN EFFECTIVE

This approach reliably produces architectural scenes with impossible perspective.

```bash
hyperspeed generate "Renaissance fresco, architectural interior with columns and tiled floor, multiple figures in classical robes, Florentine style" \
  --era early_renaissance \
  --intensity 0.5 \
  --perspective-error 0.8 \
  --proportion-shift 0.5 \
  --seed 1425 \
  --output examples/spatial_study.png
```

**Key elements:**
- Include "columns" and "tiled floor" - these create visible perspective lines that then contradict
- "multiple figures" adds proportion shift opportunities
- "Florentine style" or "fresco" helps get the right aesthetic
- `--perspective-error 0.8` creates strong spatial contradictions
- `--proportion-shift 0.5` makes figures feel wrong relative to architecture
- `--intensity 0.5` keeps the scene readable while still broken

**Variations that work:**
- "Renaissance chapel interior, vaulted ceiling, figures gathered"
- "Florentine courtyard, arched colonnade, scholars debating"
- "palace hall with checkered floor, nobles in conversation"

### Wooden Faces (Mannequin Stiffness) - NEW

Unlike Byzantine/Gothic faces (smooth, vacant, spiritually uncanny), Early Renaissance faces are *differently* wrong. They're trying too hard, not dissolving into void.

```bash
hyperspeed generate "portrait of a Florentine merchant, direct gaze, three-quarter view, in the style of Piero della Francesca, tempera on panel" \
  --era early_renaissance \
  --intensity 0.5 \
  --wooden-face 0.7 \
  --hand-failure 0.3 \
  --seed 1450 \
  --output examples/portrait_wooden.png
```

**Key elements:**
- Reference specific painters: "in the style of Masaccio" or "Fra Angelico" or "Piero della Francesca"
- "tempera on panel" gets the right texture (crisp, not sfumato)
- "direct gaze" or "stiff, formal pose" for the mannequin quality
- Keep `--hand-failure` low when focusing on face
- `--wooden-face 0.6-0.8` for visible effect

**What the effect does:**
- Sharpens edges (opposite of Byzantine smoothing) for tempera crispness
- Asymmetric eye displacement - eyes that don't quite track together
- Increases local contrast for "wooden" quality
- Flattens midtones for "performed expression"
- Subtle proportion distortion in facial features

**The distinction:**
- StyleGAN faces: too smooth, too perfect, uncanny through excess polish
- DALL-E 2 faces: parts not quite aligning, asymmetries that feel wrong, flatness despite attempted depth
- The face should feel like it's *trying too hard*, not dissolving into void

**Prompt phrases that help:**
- "studiedly naturalistic" rather than "ethereal"
- "stiff, formal pose"
- "direct gaze, slightly misaligned eyes"
- "tempera on panel"
- "in the style of [specific Early Renaissance painter]"

### Single Portraits (Piero della Francesca Style) - PROVEN EFFECTIVE

The single portrait format works exceptionally well for showcasing the wooden face effect.

```bash
hyperspeed generate "portrait of a Florentine merchant, direct gaze, three-quarter view, in the style of Piero della Francesca, tempera on panel" \
  --era early_renaissance \
  --intensity 0.5 \
  --wooden-face 0.7 \
  --hand-failure 0.2 \
  --seed 1450 \
  --output examples/portrait_single.png
```

**Key elements:**
- "three-quarter view" is the classic Early Renaissance portrait angle
- Specify occupation or social role: "merchant," "scholar," "noblewoman," "condottiero"
- `--hand-failure 0.2` keeps hands minimal so face dominates
- `--wooden-face 0.7` for clear mannequin stiffness
- `--intensity 0.5` balances effect with image quality

**Variations that work:**
- "portrait of a young scholar, direct gaze, formal pose, in the style of Antonello da Messina"
- "Florentine noblewoman portrait, three-quarter view, tempera on panel, pearl headdress"
- "portrait of a condottiero, stern expression, in the style of Andrea del Castagno"
- "young woman in profile, in the style of Pisanello, medal portrait"

### Double Portraits (Quattrocento Style) - PROVEN EFFECTIVE

Double portraits amplify the wooden face effect by showing two faces with subtly different misalignments.

```bash
hyperspeed generate "double portrait of two Florentine nobles facing viewer, formal poses, direct gazes, in the style of early quattrocento, tempera on panel" \
  --era early_renaissance \
  --intensity 0.5 \
  --wooden-face 0.7 \
  --proportion-shift 0.4 \
  --seed 1453 \
  --output examples/portrait_double.png
```

**Key elements:**
- "double portrait" or "two figures" in the prompt
- "facing viewer" ensures both faces get the wooden treatment
- `--proportion-shift 0.4` adds subtle scale inconsistency between the two figures
- Works well for donor portraits, marriage portraits, or paired saints

**Variations that work:**
- "double portrait of husband and wife, formal poses, Florentine style, tempera"
- "two saints facing viewer, gold halos, in the style of Fra Angelico"
- "paired donor portraits, man and woman, quattrocento style"
- "two scholars in conversation, direct gazes, Renaissance panel painting"

### img2img Mode

Transform existing images with Early Renaissance artifacts:

```bash
hyperspeed generate "Renaissance painting style" \
  --era early_renaissance \
  --source your_image.jpg \
  --strength 0.6 \
  --intensity 0.5 \
  --output examples/renaissance_transform.png
```

## The Art Historical Parallel

### Why This Mapping Works

The Early Renaissance (1400-1490) and 2022 diffusion models share a crucial quality: **ambitious failure**. Both were reaching for naturalistic representation and both failed in ways that now feel characteristic and, potentially, beautiful.

**Early Renaissance painters:**
- Invented (or rediscovered) perspective but couldn't always make it work
- Attempted anatomical accuracy but hands remained notoriously difficult
- Tried foreshortening but the results often flatten unexpectedly
- Created spatial staging that contradicts itself across the composition

**2022 diffusion models:**
- Generated photorealistic textures but couldn't count fingers
- Created plausible depth but perspective often breaks
- Produced bodies that shift scale across the image
- Made figures that feel "pasted onto" their backgrounds

### The Craft Distinction

Per the project's PRINCIPLES.md: "A melting face in 2022 Stable Diffusion: limitation. A melting face in this project: vocabulary."

The Early Renaissance artifacts here are **intentional compositional choices**, not bugs. We're composing with the visual vocabulary that both 15th-century ambition and 2022 AI revealed.

### Deep Cuts

- Masaccio introduced foreshortening as "features that were rarely used in painting before him" - and the results were revolutionary but imperfect
- Durer's woodcuts showing artists using grid-based "drawing machines" to capture foreshortening = the Renaissance version of training data augmentation
- The grid as prosthesis for perception mirrors our reliance on AI as prosthesis for image-making

## Prompts That Work Well

**For hands:**
- "Renaissance figure, hands clasped in prayer"
- "Hand study, classical pose, fingers extended"
- "Figure reaching toward viewer"

**For perspective:**
- "Renaissance interior, tiled floor, columns"
- "Architectural scene, multiple figures"
- "Chapel interior, vaulted ceiling"

**For foreshortening:**
- "Recumbent figure, feet toward viewer"
- "Falling figure, foreshortened"
- "Figure in dramatic perspective"

**For proportion:**
- "Group scene, figures at different distances"
- "Madonna and child, hierarchical composition"
- "Portrait with landscape background"
