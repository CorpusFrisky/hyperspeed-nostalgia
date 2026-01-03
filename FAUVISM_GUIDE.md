# Fauvism Era Guide

## Art Historical Parallel: Fauvism (1904-1908)
## AI Artifact Parallel: Color Posterization / Flat Fill Artifacts

---

## The Thesis

Fauvism used flat, unmodulated color in discrete zones. Matisse applied "paints straight from the tube," rejecting gradation for hard-edged color shapes. This maps to AI color banding, posterization, and fill-bucket simplification: artifacts where the model cannot render smooth gradation and defaults to flat color regions.

**The failure is not oversaturation but the inability to blend.**

Previous approach (deprecated): Teal-orange cast, saturation violence. This produced colorful images but not Fauvist images. The colors were intense but still behaved like observed color, not applied color.

New approach: Force flat color through medium and style cues that inherently resist gradation.

---

## Key Concepts

### Arbitrary Color vs. Saturated Color

- **Saturated color**: Natural color pushed to extremes. Still follows light logic.
- **Arbitrary color**: Color chosen by the artist, not observed from nature. The green stripe down Matisse's wife's face was not about lighting; it was about authority.

The goal is color that feels **applied rather than observed, chosen rather than rendered**.

### Flat Fill as Artifact

AI posterization artifacts occur when the model:
- Cannot render smooth gradients
- Defaults to discrete color regions
- Produces color banding instead of blending
- Creates hard edges where soft transitions should exist

This maps directly to Fauvist technique: hard-edged color zones, no blending, paint applied in discrete shapes.

---

## Prompt Strategies

### Media That Force Flatness

These media inherently resist gradation:

```
gouache painting, flat opaque colors
poster paint, bold flat areas
cut paper collage, overlapping flat shapes
woodblock print, color separation
screen print, limited flat colors
Matisse cut-outs style
linocut print
risograph print style
```

### Style References

```
Matisse cut-outs
after Derain
Fauvist flat color
color as shape
discrete color zones
no blending, hard edges
paint straight from the tube
```

### What NOT to Use

Avoid terms that imply gradation or atmospheric color:
- "oil painting" (implies blending)
- "atmospheric"
- "soft light"
- "subtle gradation"
- "realistic"

---

## Test Subjects

### Portrait with Flat Color Zones
Face as discrete color shapes, no blending between zones.
```
Portrait with face as FLAT COLOR ZONES, green stripe down nose,
orange cheek, blue shadow, yellow forehead, each color a discrete
shape with hard edges, gouache painting style, no blending,
after Matisse
```

### Landscape as Cut Paper
Trees, hills, sky as flat overlapping shapes.
```
Landscape as CUT PAPER COLLAGE, hills as flat green shapes,
trees as flat orange and red shapes, sky as flat blue,
overlapping paper forms, Matisse cut-outs style, no gradation
```

### Harbor Scene, Gouache
Boats as simple color blocks.
```
Harbor scene, GOUACHE, bold flat colors, boats as simple
color blocks, water as flat blue shape, buildings as flat
rectangles, no atmospheric perspective, poster paint style
```

### Still Life, Screen Print
Force posterization through print medium.
```
Still life fruit bowl, SCREEN PRINT STYLE, limited palette
of 5 flat colors, no gradation, hard color separation,
each fruit a flat shape, risograph aesthetic
```

### Figure in Interior, Matisse Cut-Outs
Maximum flatness.
```
Figure in interior, MATISSE CUT-OUTS STYLE, woman as flat
shape, furniture as flat shapes, bold flat colors, no modeling,
no shadow gradation, paper collage aesthetic
```

---

## Parameters

For flat color effect, the post-processing should emphasize:
- Lower `teal_orange_cast` (we want arbitrary, not that specific cast)
- Moderate `saturation_violence` (flatness matters more than intensity)
- High `pigment_expressiveness` (the "straight from tube" effect)
- Consider reducing `chromatic_intensity` (flat, not glowing)

The real work is in the prompt, forcing the base generation toward flat color through medium cues.

---

## What Worked (from experiments)

### Successful Color Violations
- **Green stripe portrait**: Face in discrete color zones, arbitrary color placement
- **Competing lights interior**: Wrong-colored light sources (green window light, purple lamp)
- **Mirror different palette**: Same subject, different color logic in reflection
- **Forced color subjects**: Explicitly naming wrong colors in prompts (orange trunks, magenta leaves)

### What the Model Resists
- Wrong-colored shadows (defaults to dark/physics)
- Reflections that "lie" (defaults to matching colors)
- Truly arbitrary shadow colors on objects

### Key Insight
The model will accept arbitrary **object** colors more readily than arbitrary **shadow** colors. It enforces physics on lighting even when it accepts fantasy on objects.

---

## Deprecated Approach

The original Fauvism pairing was "The Golden Age of Slop (2022-2023)" with teal-orange cast and saturation violence. This produced:
- Highly saturated images
- Strong teal-orange color grading
- But colors that still followed light logic
- Not truly Fauvist arbitrariness

The teal-orange cast is a valid AI artifact but maps better to a different phenomenon (perhaps a future "Instagram Filter" era) than to Fauvism's deliberate flatness and arbitrary color zones.

---

## Document History

- Initial approach: Saturation violence, teal-orange cast
- Revision 1: Forced color in prompts (explicit wrong colors)
- Revision 2: Flat fill / posterization approach (current)
