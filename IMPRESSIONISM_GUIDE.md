# Impressionism (The Melting Zone Era) Guide

## Overview

The Impressionism era maps to **The Melting Zone** - the moment when boundaries dissolve and forms become probabilistic. Monet's *Impression, Sunrise* (1872) gave the movement its name, and critics called it "unfinished," a mere impression. Impressionists rendered "optical data" rather than "what we know about space, mass, and the other physical details of the world."

Unlike Neoclassicism (over-idealization) or High Renaissance/MJ v4 (over-dramatization), Impressionism is about **dissolution**. Boundaries become probabilistic rather than defined. Forms melt into light. The tell isn't failure or excess; it's uncertainty made visible.

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
hyperspeed generate "Water lilies on pond" --era impressionism --use-local
```

## The Aesthetic Goal

We're mapping **The Melting Zone** onto **Impressionism (1860-1890)**. The connection:

- **Edge dissolution** = boundaries between forms become uncertain
- **Color bleed** = "colors melding together in its glooming, opalescent oneness"
- **Atmospheric haze** = "foggy blankness, its featureless, expectant emptiness"
- **Brushstroke texture** = the paint "becomes the place"
- **Light fragmentation** = broken color, optical mixing
- **Temporal blur** = the captured fleeting moment

Key works to reference:
- Monet's Water Lilies (1899-1926): boundaries between sky reflection and water surface become indistinguishable
- Impression, Sunrise (1872): "colors melding together in its glooming, opalescent oneness"
- Monet's Haystacks and Rouen Cathedral series: same subject at different times, edges dissolving
- Renoir's Dance at Le Moulin de la Galette (1876): figures dissolving into dappled light

## What Works

### Technical Settings

```bash
# For water scenes with maximum dissolution
hyperspeed generate "Water lilies on pond, reflections, Monet style, impressionist painting" \
  --era impressionism --intensity 0.8 --edge-dissolution 0.9 --color-bleed 0.7

# For atmospheric sunrise/sunset scenes
hyperspeed generate "Harbor at sunrise, boats, morning fog, impressionist painting" \
  --era impressionism --intensity 0.7 --atmospheric-haze 0.9 --light-fragmentation 0.7

# For figure studies in dappled light
hyperspeed generate "Woman with parasol in garden, dappled light, impressionist style" \
  --era impressionism --intensity 0.6 --edge-dissolution 0.7 --brushstroke-texture 0.6
```

### Intensity Sweet Spots

| Subject | Intensity | Notes |
|---------|-----------|-------|
| Water/Reflections | 0.7-0.9 | Maximum dissolution works beautifully |
| Sunrise/Sunset | 0.6-0.8 | Atmospheric effects dominate |
| Garden scenes | 0.5-0.7 | Light fragmentation for dappled effect |
| Figures | 0.5-0.6 | Subtle dissolution, preserve form |
| Landscapes | 0.6-0.8 | Haze and brushstroke texture |

### Individual Effect Controls

Each tell can be controlled independently:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--edge-dissolution` | 0.7 | THE signature - boundaries become probabilistic |
| `--color-bleed` | 0.6 | Colors meld across boundaries |
| `--atmospheric-haze` | 0.6 | Foggy, expectant emptiness |
| `--brushstroke-texture` | 0.5 | Visible paint application |
| `--light-fragmentation` | 0.5 | Broken color, optical mixing |
| `--temporal-blur` | 0.4 | Motion suggesting captured moment |

## Example Commands

### Edge Dissolution (THE Signature)

This is THE Impressionist signature: edges that aren't quite there. Boundaries become uncertain, probabilistic.

```bash
hyperspeed generate "Water lilies on pond, sky reflections merging with water, Monet style, impressionist painting" \
  --era impressionism \
  --intensity 0.8 \
  --edge-dissolution 0.9 \
  --color-bleed 0.7 \
  --atmospheric-haze 0.6 \
  --seed 1872 \
  --output examples/impressionism/water_lilies.png
```

**Key elements:**
- Prompt should suggest merging, reflections, soft boundaries
- `--edge-dissolution 0.8-0.9` for obvious effect
- Works especially well on water scenes where boundaries naturally blur
- Combine with `--color-bleed` for opalescent quality

**Variations that work:**
- "Pond at evening, reflections dissolving into surface"
- "Japanese bridge over water, soft edges, impressionist"
- "River scene, sky and water merging, Monet style"

### Atmospheric Haze (Foggy Emptiness)

The "foggy blankness, its featureless, expectant emptiness" of Impression, Sunrise.

```bash
hyperspeed generate "Harbor at sunrise, boats in morning mist, impressionist painting, atmospheric" \
  --era impressionism \
  --intensity 0.7 \
  --atmospheric-haze 0.9 \
  --color-bleed 0.6 \
  --light-fragmentation 0.6 \
  --seed 1872 \
  --output examples/impressionism/sunrise.png
```

**Key elements:**
- "mist," "fog," "atmospheric" in prompts
- `--atmospheric-haze 0.8-0.9` for heavy haze effect
- Lower other effects to let atmospheric quality dominate
- Works on any subject needing dreamy distance

**Variations that work:**
- "Railway station, steam and light, morning atmosphere"
- "Paris street, rainy day, figures dissolving in mist"
- "Cathedral facade, morning light, atmospheric haze"

### Color Bleed (Opalescent Oneness)

"Colors melding together in its glooming, opalescent oneness."

```bash
hyperspeed generate "Sunset over haystacks, golden and violet light merging, impressionist painting" \
  --era impressionism \
  --intensity 0.7 \
  --color-bleed 0.8 \
  --edge-dissolution 0.6 \
  --light-fragmentation 0.6 \
  --seed 1890 \
  --output examples/impressionism/haystacks.png
```

**Key elements:**
- Describe colors meeting, merging, contrasting
- `--color-bleed 0.7-0.8` for chromatic bleeding
- Works on scenes with color transitions (sunset, reflections)
- Creates that opalescent quality where hues blend

**Variations that work:**
- "Garden at twilight, purple shadows meeting golden light"
- "Poppies field, red bleeding into green landscape"
- "Cathedral, rose window light, colors merging on stone"

### Light Fragmentation (Broken Color)

The Impressionist technique of breaking color into dabs that the eye synthesizes.

```bash
hyperspeed generate "Garden party, dappled sunlight through trees, figures in impressionist style" \
  --era impressionism \
  --intensity 0.6 \
  --light-fragmentation 0.8 \
  --brushstroke-texture 0.6 \
  --edge-dissolution 0.5 \
  --seed 1876 \
  --output examples/impressionism/garden_party.png
```

**Key elements:**
- "dappled light," "broken color," "sunlight through leaves"
- `--light-fragmentation 0.7-0.8` for stippled effect
- Works beautifully on outdoor scenes with filtered light
- Combine with `--brushstroke-texture` for visible paint

**Variations that work:**
- "Luncheon on the grass, sunlight filtering through trees"
- "Woman reading under tree, dappled shadows"
- "Cafe terrace, evening lights, broken color"

### Brushstroke Texture (Visible Paint)

Making the paint application visible - the hand of the artist.

```bash
hyperspeed generate "Rouen Cathedral, facade in afternoon light, thick brushstrokes, impressionist" \
  --era impressionism \
  --intensity 0.7 \
  --brushstroke-texture 0.8 \
  --color-bleed 0.5 \
  --atmospheric-haze 0.5 \
  --seed 1894 \
  --output examples/impressionism/cathedral.png
```

**Key elements:**
- Prompts can mention "thick brushstrokes," "textured paint"
- `--brushstroke-texture 0.7-0.8` for obvious strokes
- Works on architectural subjects where texture contrasts with form
- The paint IS the subject

**Variations that work:**
- "Cliffs at Etretat, rough sea, visible brushwork"
- "Wheatfield, swirling paint texture, Van Gogh influence"
- "Village street, painted surface, impressionist technique"

### Temporal Blur (The Captured Moment)

Suggesting motion, the fleeting instant, time made visible.

```bash
hyperspeed generate "Dancers in motion, ballet rehearsal, impressionist painting, movement blur" \
  --era impressionism \
  --intensity 0.6 \
  --temporal-blur 0.7 \
  --edge-dissolution 0.5 \
  --light-fragmentation 0.5 \
  --seed 1874 \
  --output examples/impressionism/dancers.png
```

**Key elements:**
- Subjects in motion: dancers, crowds, passing figures
- `--temporal-blur 0.6-0.7` for motion suggestion
- Works on dynamic scenes where time should feel visible
- Subtle - suggests rather than shows motion

**Variations that work:**
- "Crowds at the races, movement and excitement"
- "Train arriving at station, steam and motion"
- "Boating on the Seine, rippling water, fleeting moment"

### The Full Impressionist Treatment

Combining all effects for maximum dissolution.

```bash
hyperspeed generate "Water lilies at sunset, Monet's garden, reflections merging with sky, impressionist masterpiece" \
  --era impressionism \
  --intensity 0.8 \
  --edge-dissolution 0.8 \
  --color-bleed 0.7 \
  --atmospheric-haze 0.7 \
  --brushstroke-texture 0.5 \
  --light-fragmentation 0.6 \
  --temporal-blur 0.3 \
  --seed 1906 \
  --output examples/impressionism/full.png
```

### img2img Mode (Photo to Impressionist Painting)

Transform existing images with Impressionist dissolution:

```bash
hyperspeed generate "Impressionist painting style, soft edges, atmospheric" \
  --era impressionism \
  --source your_photo.jpg \
  --strength 0.65 \
  --intensity 0.7 \
  --edge-dissolution 0.7 \
  --color-bleed 0.6 \
  --output examples/impressionism/transform.png
```

**Why img2img works well for Impressionism:**
- Takes sharp photograph, dissolves it into paint
- Preserves general composition while softening edges
- The transformation itself mirrors Impressionist philosophy: rendering optical data rather than physical reality

## Subject Categories

### Water Scenes (Ideal for Dissolution)

Water naturally reflects, merges, and dissolves - perfect for edge dissolution and color bleed.

```bash
# Giverny garden style
hyperspeed generate "Japanese footbridge over lily pond, wisteria, Monet's garden, impressionist" \
  --era impressionism \
  --intensity 0.8 \
  --edge-dissolution 0.9 \
  --color-bleed 0.7 \
  --seed 1899 \
  --output examples/impressionism/giverny.png

# Seine river scenes
hyperspeed generate "Boats on the Seine, summer afternoon, rippling water, impressionist" \
  --era impressionism \
  --intensity 0.7 \
  --temporal-blur 0.6 \
  --light-fragmentation 0.6 \
  --seed 1869 \
  --output examples/impressionism/seine.png
```

### Atmospheric Landscapes

Haze, mist, and soft distance create the characteristic Impressionist mood.

```bash
# Haystacks series style
hyperspeed generate "Haystacks at sunset, purple shadows, orange sky, impressionist" \
  --era impressionism \
  --intensity 0.7 \
  --atmospheric-haze 0.7 \
  --color-bleed 0.7 \
  --seed 1891 \
  --output examples/impressionism/haystacks_sunset.png

# Morning mist
hyperspeed generate "Poplars along river, morning mist, impressionist landscape" \
  --era impressionism \
  --intensity 0.7 \
  --atmospheric-haze 0.8 \
  --edge-dissolution 0.6 \
  --seed 1891 \
  --output examples/impressionism/poplars.png
```

### Urban Scenes (Parisian Life)

The Impressionists captured modern urban life - cafes, boulevards, leisure.

```bash
# Boulevard scene
hyperspeed generate "Paris boulevard, afternoon crowd, trees casting shadows, impressionist" \
  --era impressionism \
  --intensity 0.6 \
  --light-fragmentation 0.7 \
  --temporal-blur 0.5 \
  --seed 1877 \
  --output examples/impressionism/boulevard.png

# Cafe scene
hyperspeed generate "Cafe terrace at night, gas lamps, figures at tables, impressionist" \
  --era impressionism \
  --intensity 0.6 \
  --color-bleed 0.6 \
  --brushstroke-texture 0.6 \
  --seed 1888 \
  --output examples/impressionism/cafe.png
```

### Figures in Landscape

People dissolving into their environment, light playing on forms.

```bash
# Woman with parasol
hyperspeed generate "Woman with parasol in meadow, wind in dress, summer light, impressionist" \
  --era impressionism \
  --intensity 0.6 \
  --edge-dissolution 0.6 \
  --light-fragmentation 0.6 \
  --temporal-blur 0.4 \
  --seed 1875 \
  --output examples/impressionism/parasol.png

# Luncheon on the grass
hyperspeed generate "Picnic in forest clearing, dappled sunlight, figures in shade, impressionist" \
  --era impressionism \
  --intensity 0.6 \
  --light-fragmentation 0.8 \
  --edge-dissolution 0.5 \
  --seed 1866 \
  --output examples/impressionism/picnic.png
```

## The Problem: Impressionism Already Looks Melted

**Critical insight:** When the AI makes a competent Impressionist pastiche of water lilies or haystacks, it just looks... Impressionist. The artifacts blend in. The dissolution is stylistically *appropriate*.

**The solution:** Choose subjects where the melting becomes *wrong* rather than stylistically appropriate. We need subjects where the viewer asks "wait, whose arm is that?" or "is that a person or a shadow?"

### Strategies for Revealing the Tell

1. **Add figures** - Impressionists painted people: dancers, cafe-goers, picnickers. AI struggles with figures in dissolved environments. Bodies merge with backgrounds, faces become suggestions.

2. **Reflections** - Water reflections are ripe for error. The model might not understand which parts are real vs reflected. Things double incorrectly, reflections detach from sources.

3. **Crowded scenes** - Renoir's dance halls, Monet's train stations, Manet's cafe-concerts. Multiple figures = more opportunity for boundary confusion between bodies.

4. **Subjects with important edges** - Dancers mid-movement, umbrellas, parasols, architecture mixed with nature. Things where you *need* to know where one thing ends and another begins.

### Figure-Focused Prompts (Exploiting Boundary Confusion)

```bash
# Crowded dance hall - whose arm is whose?
hyperspeed generate "Dancers at the Moulin de la Galette, crowded dance hall, dappled sunlight through trees, multiple figures dancing, impressionist painting, Renoir style" \
  --era impressionism \
  --intensity 0.85 \
  --edge-dissolution 0.9 \
  --color-bleed 0.7 \
  --light-fragmentation 0.7 \
  --seed 1876 \
  --output examples/impressionism/moulin_galette.png

# Cafe terrace - figures dissolving into lamplight
hyperspeed generate "Cafe terrace at night, figures seated at tables, gas lamps glowing, waiters serving, impressionist painting, warm lamplight on faces" \
  --era impressionism \
  --intensity 0.8 \
  --edge-dissolution 0.85 \
  --atmospheric-haze 0.7 \
  --color-bleed 0.6 \
  --seed 1888 \
  --output examples/impressionism/cafe_terrace.png

# Train station - steam, crowds, iron architecture
hyperspeed generate "Train station with steam and crowds, iron and glass architecture, passengers waiting, impressionist painting, Monet Gare Saint-Lazare style" \
  --era impressionism \
  --intensity 0.8 \
  --atmospheric-haze 0.9 \
  --edge-dissolution 0.8 \
  --temporal-blur 0.6 \
  --seed 1877 \
  --output examples/impressionism/train_station.png

# Garden party - parasols and white dresses
hyperspeed generate "Garden party with women in white dresses, parasols, dappled afternoon light, multiple figures, impressionist painting" \
  --era impressionism \
  --intensity 0.7 \
  --light-fragmentation 0.8 \
  --edge-dissolution 0.7 \
  --color-bleed 0.6 \
  --seed 1867 \
  --output examples/impressionism/garden_party.png

# Boating scene - reflections confusing real/reflected
hyperspeed generate "Boating party on the Seine, figures in rowboat, water reflections, striped shirts, impressionist painting, Renoir style" \
  --era impressionism \
  --intensity 0.8 \
  --edge-dissolution 0.85 \
  --color-bleed 0.8 \
  --temporal-blur 0.5 \
  --seed 1881 \
  --output examples/impressionism/boating.png
```

**What to look for:**
- Arms merging with backgrounds or other figures
- Faces becoming suggestions rather than faces
- Reflections detaching from their sources
- Bodies blending into each other in crowded scenes
- Parasols and umbrellas losing their edges
- The dissolution conflicts with narrative clarity

## The Art Historical Parallel

### Why This Mapping Works

The Impressionists and "The Melting Zone" share a crucial quality: **the dissolution of certainty into probability**. Both represent a departure from hard boundaries toward soft transitions.

**Impressionist revolution:**
- Rejected academic precision for optical truth
- Made visible the uncertainty of perception
- Let colors blend in the viewer's eye
- Captured moments rather than monuments

**The Melting Zone:**
- Boundaries become probabilistic
- Forms dissolve into atmospheric effects
- Certainty gives way to suggestion
- The eye completes what the image implies

### The Distinction from Other Eras

| Era | AI Parallel | Technique Focus |
|-----|-------------|-----------------|
| High Renaissance | Midjourney v4 | **Over-dramatization** (everything epic) |
| Neoclassicism | Stock Photo AI | **Over-idealization** (too perfect, commercial) |
| Impressionism | Melting Zone | **Dissolution** (uncertain, probabilistic) |

Impressionism is recognizable not because it fails or tries too hard, but because it **dissolves certainty**. Edges don't fail; they become probabilistic. Colors don't bleed by accident; they meld by design.

### Deep Cuts

- Monet's Water Lilies (1899-1926): boundaries between sky reflection and water surface become indistinguishable - this IS edge dissolution
- "Colors melding together in its glooming, opalescent oneness" - art historian on Impression, Sunrise - this IS color bleed
- The critics called it "unfinished" - the dissolution was intentional, not failure
- "Optical data rather than what we know about space, mass, and physical details" - perception over knowledge

## Prompts That Work Well

**For edge dissolution:**
- "reflections merging with surface"
- "forms dissolving into light"
- "soft edges, impressionist"

**For atmospheric haze:**
- "morning mist," "foggy atmosphere"
- "soft distance," "atmospheric perspective"
- "dreamy quality," "expectant emptiness"

**For color bleed:**
- "colors melding," "opalescent"
- "sunset hues merging"
- "chromatic harmony"

**For light fragmentation:**
- "dappled sunlight," "broken color"
- "light through leaves"
- "stippled effect"

**For brushstroke texture:**
- "thick brushstrokes," "visible paint"
- "textured surface"
- "impasto technique"

**For temporal blur:**
- "fleeting moment," "motion"
- "captured instant"
- "movement blur"

---

## Session Progress Log

### Last Updated: 2025-12-09

### Key Insight: Forcing Visible Failure

The breakthrough came from realizing that Impressionist subjects with Impressionist style just look... Impressionist. The dissolution blends in. Instead, we need:

1. **Painterly aesthetic without naming Impressionism** - Use anchors like "oil painting with soft edges," "the quality of a Whistler nocturne," "painted not photographed"
2. **Subjects that force AI failure points** - Text, hands doing delicate tasks, mirrors/reflections, repetition of similar figures
3. **Low inference steps (14-15)** - Creates instability at edges, images that couldn't quite resolve

### Successful Prompt Formula (v8 batch)

**Structure:** `[Period subject with failure point] + [Impressionist light/palette] + [Painterly anchors]`

**Painterly anchors that work:**
- "oil painting with soft edges and atmospheric haze"
- "the quality of a Whistler nocturne" (night/gaslight scenes)
- "the quality of a Sargent interior" (café/interior scenes)
- "painterly rendering, visible brushwork in shadows"
- "painted not photographed"

**Settings:** intensity=0.95, edge_dissolution=1.0, inference_steps=14

### Batch v8: Forced Failure Subjects (Reference)

| File | Subject | Failure Point | Seed |
|------|---------|---------------|------|
| `impressionism_newspaper_kiosk.png` | Paris newspaper kiosk at dusk | Text everywhere - headlines, mastheads | 1885 |
| `impressionism_orchestra_hands.png` | Orchestra pit from above | Dozens of hands on instruments, fingers on strings/keys | 1878 |
| `impressionism_playing_cards.png` | Cafe table with playing cards | Cards fanned out, pips half-legible, fingers holding | 1882 |
| `impressionism_milliner_window.png` | Milliner's shop window | Price tags, handwritten signs, precise arrangements | 1883 |
| `impressionism_departure_board.png` | Train station departure board | Destination names, times, platform numbers | 1877 |
| `impressionism_dance_mirrors.png` | Dance hall with mirrors | Reflections multiplying, hands on waists/shoulders | 1876 |
| `impressionism_pianist_hands.png` | Pianist's hands on keyboard | Fingers on keys, sheet music with notes | 1879 |
| `impressionism_bookshop.png` | Bookshop interior | Gilt titles on spines - rows of text | 1884 |

#### Example Commands (v8 style)

```bash
# Newspaper kiosk - text failure
hyperspeed generate "Paris newspaper kiosk at dusk, stacks of newspapers and magazines, headlines and mastheads visible, warm gaslight on newsprint, oil painting with soft edges and atmospheric haze, the quality of a Whistler nocturne, painted not photographed" \
  --era impressionism \
  --intensity 0.95 \
  --edge-dissolution 1.0 \
  --color-bleed 0.85 \
  --atmospheric-haze 0.6 \
  --seed 1885 \
  --output examples/impressionism/newspaper_kiosk.png

# Orchestra hands - hand failure
hyperspeed generate "Orchestra pit from above, musicians' hands on instruments, fingers on violin strings and piano keys, dozens of hands in motion, warm concert hall lighting, oil painting with soft edges, painterly rendering, painted not photographed" \
  --era impressionism \
  --intensity 0.95 \
  --edge-dissolution 1.0 \
  --color-bleed 0.8 \
  --temporal-blur 0.6 \
  --seed 1878 \
  --output examples/impressionism/orchestra_hands.png

# Playing cards - hands + text
hyperspeed generate "Cafe table close-up with playing cards, cards fanned out in hands, pips and face cards visible, fingers holding cards, warm afternoon light through window, the quality of a Sargent interior, oil painting with atmospheric haze, painted not photographed" \
  --era impressionism \
  --intensity 0.95 \
  --edge-dissolution 1.0 \
  --color-bleed 0.85 \
  --light-fragmentation 0.7 \
  --seed 1882 \
  --output examples/impressionism/playing_cards.png

# Dance mirrors - reflection failure
hyperspeed generate "Dance hall with large mirrors, couples waltzing reflected in mirrors, reflections multiplying figures, hands on waists and shoulders, warm gaslight chandeliers, oil painting with soft edges, the quality of a Sargent interior, painted not photographed" \
  --era impressionism \
  --intensity 0.95 \
  --edge-dissolution 1.0 \
  --color-bleed 0.9 \
  --temporal-blur 0.5 \
  --seed 1876 \
  --output examples/impressionism/dance_mirrors.png
```

### Batch v9: Classic Subjects with Forced Failures

| File | Subject | Failure Point | Seed |
|------|---------|---------------|------|
| `impressionism_barre_mirrors.png` | Ballet practice room with mirrors | 12 dancers, reflections not matching | 1874 |
| `impressionism_tying_pointe.png` | Hands tying pointe shoe ribbons | Delicate hand task with ribbon | 1876 |
| `impressionism_legs_forest.png` | Dancers through forest of legs | Spatial confusion, limb multiplication | 1878 |
| `impressionism_stage_view.png` | View from stage at audience | Hundreds of faces dissolving | 1877 |
| `impressionism_opera_glasses.png` | Audience holding opera glasses | Dozens of hands with reflective objects | 1879 |
| `impressionism_theatre_program.png` | Theatre program held open | Names and text on paper | 1880 |
| `impressionism_waiter_absinthe.png` | Waiter with tray of absinthe | Hands + transparent glasses + navigation | 1876 |
| `impressionism_cafe_infinite.png` | Cafe with mirrored walls | Infinite reflections, Bar at Folies-Bergere | 1882 |
| `impressionism_cafe_hands.png` | Hands at cafe table | Multiple hands doing different tasks | 1881 |
| `impressionism_menu_board.png` | Handwritten menu and notices | Chalk writing, posted bills | 1883 |

#### Performance Notes

- **Replicate API**: ~30 seconds per image generation
- **Local post-processing**: ~10-15 seconds per image
- **Inference steps**: 14-15 for optimal instability
- Edge dissolution is computationally intensive (pixel displacement)
- Low inference steps create "almost resolved" quality - the melting zone

#### Observations

*Track observations about what works and what doesn't*
