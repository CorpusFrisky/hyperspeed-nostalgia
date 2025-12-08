# Early Diffusion Era: Late Roman/Byzantine Mosaics

## Overview

Implement the **Early Diffusion Era (2021-2022)** mapped to **Late Roman/Early Byzantine Mosaics (300-600 CE)**.

From the project document:
> Discrete units (tesserae) assembling into images. Detail lost in the tessellation. The "jeweled style" of Late Antiquity (polychrome colors, geometric patterns over representation) parallels early diffusion's soft, dreamy quality where nothing quite connects.

## Art Historical Connection

**Key characteristics of Late Roman/Byzantine Mosaics:**
- Tesserae: discrete colored tiles that form images
- "Jeweled style": polychrome colors, precious stone effect
- Gold backgrounds creating void/transcendent space
- Detail lost in tessellation (faces that are technically correct but ethereal)
- "Hypnotizing glimmer" from angled tiles reflecting light

**Early Diffusion (2021-2022) artifacts:**
- Soft, dreamy quality
- Low inference steps causing undercooked output
- Background bleeding (the diffusion equivalent of gold void)
- Nothing quite connects properly
- Extreme CFG causing oversaturation
- Older schedulers (DDIM, PNDM) with characteristic noise patterns

## Technical Approach

Create `src/hyperspeed/eras/early_diffusion.py` following the pattern of `early_gan.py`:

1. Generate base image with Stable Diffusion using **deliberately degraded settings**:
   - Low inference steps (5-15)
   - Extreme guidance scale OR very low
   - Older schedulers (DDIMScheduler, PNDMScheduler)

2. Apply post-processing artifacts:
   - **Tessellation effect**: Quantize image into discrete color blocks
   - **Gold void background**: Metallic shimmer in low-detail areas
   - **Color bleeding**: Colors spill into neighboring regions
   - **Soft edges**: Nothing quite connects, boundaries dissolve
   - **Glimmer effect**: Subtle sparkle/highlight variation (like angled tesserae)

## Files to Create/Modify

### Create: `src/hyperspeed/eras/early_diffusion.py`

```python
@EraRegistry.register
class EarlyDiffusionPipeline(EraPipeline):
    metadata = EraMetadata(
        name="Early Diffusion",
        art_historical_parallel="Late Roman/Byzantine Mosaics (300-600 CE)",
        time_period="2021-2022",
        description="Discrete units assembling into images. The jeweled style meets diffusion's dreamy quality.",
        characteristic_artifacts=[
            "Tessellated texture",
            "Gold void backgrounds",
            "Color bleeding",
            "Soft disconnected edges",
            "Hypnotizing glimmer",
            "Undercooked details",
        ],
    )
```

### Artifact Methods

1. `_apply_tessellation(img, strength, style)`:
   - Divide image into grid blocks
   - Quantize colors within each block to dominant color
   - If style="obvious": visible grout lines, hard tile edges
   - If style="subtle": no grout, soft color quantization
   - Vary tile sizes slightly for organic feel

2. `_apply_gold_void(img, strength, style)`:
   - Detect low-detail background areas
   - If style="metallic": literal gold color with shimmer/sparkle
   - If style="symbolic": warm yellows/ochres, more flexible palette
   - Add subtle sheen variation in either mode

3. `_apply_color_bleeding(img, strength)`:
   - Colors leak into neighboring regions
   - Especially at boundaries between distinct areas
   - Creates that "nothing quite connects" quality

4. `_apply_soft_edges(img, strength)`:
   - Dissolve hard boundaries
   - Create dreamy, ethereal transitions
   - Not blur, but boundary dissolution

5. `_apply_glimmer(img, strength)`:
   - Random brightness/saturation micro-variations
   - Mimics light catching angled tesserae
   - "Hypnotizing glimmer" effect

### Era-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tile_size` | 8 | Size of mosaic tesserae in pixels |
| `tile_style` | "subtle" | "obvious" (visible grout lines, distinct blocks) or "subtle" (impressionistic texture) |
| `gold_style` | "symbolic" | "metallic" (literal gold shimmer) or "symbolic" (warm yellows/ochres) |
| `gold_strength` | 0.5 | Intensity of gold void background |
| `inference_steps` | 10 | Deliberately low for undercooked quality |
| `guidance_scale` | 12.0 | Extreme CFG for oversaturated look |
| `scheduler` | "ddim" | Older scheduler for characteristic noise |

### Modify: `src/hyperspeed/cli.py`

Add import for early_diffusion module to register the era.

## Implementation Steps

1. Create `src/hyperspeed/eras/early_diffusion.py`
2. Implement `EarlyDiffusionPipeline` class with metadata
3. Implement `load_model()` with configurable scheduler
4. Implement `generate()` with low-step/high-CFG generation
5. Implement artifact post-processing methods:
   - `_apply_tessellation()`
   - `_apply_gold_void()`
   - `_apply_color_bleeding()`
   - `_apply_soft_edges()`
   - `_apply_glimmer()`
6. Add import in `cli.py`
7. Test with: `hyperspeed generate "Byzantine emperor mosaic, gold background, Ravenna style" --era early_diffusion --intensity 0.6`

## Example Prompts to Test

- "Byzantine mosaic of Christ Pantocrator, gold background, jeweled robes, frontal pose"
- "Mausoleum of Galla Placidia, blue starry dome, gold stars, deep blue night sky"
- "Theodora mosaic, empress with attendants, jeweled crown, purple robes"
- "Roman villa floor mosaic, geometric patterns, tesserae visible, polychrome"
