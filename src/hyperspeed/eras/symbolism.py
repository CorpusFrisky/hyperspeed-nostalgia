"""Symbolism Era (1880-1910) / Midjourney Stylization parallel.

Ornate, dreamlike, mysterious. Too beautiful, too stylized. Failure through
excess rather than dissolution.

Symbolism sought to evoke mood and meaning through imagery that was
deliberately lush, decorative, and otherworldly. Moreau's jeweled fantasies,
Redon's floating eyes and flower-heads, early Klimt's golden excess.

Midjourney's signature failure mode is similar: over-polished fantasy-
illustration gloss, everything rendered with maximum mystique. Against
Symbolism, this excess becomes visible as error. The AI trying too hard
to be mysterious tips into kitsch.

Key works:
- Gustave Moreau's Jupiter and Semele (1895): a riot of ornament, every
  surface encrusted with detail, the mythological made opulent to the point
  of suffocation
- Odilon Redon's The Cyclops (1914): dreamlike, soft-focused, a giant eye
  gazing with unsettling tenderness
- Klimt's Pallas Athene (1898): gold leaf, geometric patterning, the
  decorative overwhelming the figurative

Deep Cut: Symbolism was criticized in its time for being "literary" rather
than painterly, more interested in mood than form. Midjourney faces the same
critique: it produces "vibes" rather than images, atmosphere rather than
substance. Both achieve mystery through accumulation of signifiers rather
than restraint. The failure is not ugliness but excessive beauty, not
confusion but too-obvious meaning.

The Tell: Unlike Post-Impressionism (structural dissolution) or Neoclassicism
(over-idealization), Symbolism is about EXCESSIVE MYSTIQUE. The model tries
so hard to be beautiful and mysterious that it becomes kitsch. Every surface
glitters. Every shadow holds meaning. Every element is symbolic. It's too
much, and that excess is the failure.

Technical approach:
1. Generate image with SDXL via Replicate API (fast) or local (fallback)
2. Apply characteristic Symbolism/Midjourney excess effects locally:
   - Ornamental encrustation (detail accumulation on surfaces)
   - Mystery gloss (ethereal sheen, soft glows)
   - Golden overflow (Klimt-style gold spreading)
   - Ethereal softening (Redon's dreamlike blur)
   - Symbolic saturation (color pushed toward symbolic intensity)
   - Decorative overwhelm (pattern invasion)

Environment:
- Set REPLICATE_API_TOKEN to use Replicate for fast generation
- Falls back to local SDXL if token not set or API fails
"""

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
from scipy import ndimage

from hyperspeed.core.artifact_control import ArtifactControl
from hyperspeed.core.replicate_utils import generate_via_replicate
from hyperspeed.eras.base import EraMetadata, EraPipeline, EraRegistry


@EraRegistry.register
class SymbolismPipeline(EraPipeline):
    """Symbolism Era: Midjourney Stylization ("The Recognizable Swirl").

    This pipeline generates images with the characteristic "excessive mystique"
    quality of Symbolism and Midjourney's signature over-stylization. Everything
    is too beautiful, too meaningful, too mysterious.

    Unlike Post-Impressionism (structural dissolution) or Neoclassicism
    (idealized smoothness), Symbolism fails through EXCESS. The model tries
    too hard to be mysterious and tips into kitsch.

    Controls:
        ornamental_encrustation: Detail accumulation on surfaces (0-1)
        mystery_gloss: Ethereal sheen and soft glows (0-1)
        golden_overflow: Klimt-style gold spreading (0-1)
        ethereal_softening: Redon's dreamlike blur (0-1)
        symbolic_saturation: Color pushed toward intensity (0-1)
        decorative_overwhelm: Pattern invasion of forms (0-1)
    """

    metadata: ClassVar[EraMetadata] = EraMetadata(
        name="Symbolism",
        art_historical_parallel="Symbolism (1880-1910)",
        time_period="Midjourney Stylization",
        description=(
            "Ornate, dreamlike, mysterious. Too beautiful, too stylized. "
            "Moreau's jeweled fantasies, Redon's floating eyes, Klimt's golden excess. "
            "The AI tries too hard to be mysterious and tips into kitsch. "
            "Failure through excess rather than dissolution."
        ),
        characteristic_artifacts=[
            "Ornamental encrustation (surfaces encrusted with detail)",
            "Mystery gloss (ethereal sheen, everything glows)",
            "Golden overflow (gold leaf spreading everywhere)",
            "Ethereal softening (dreamlike, soft-focused)",
            "Symbolic saturation (unnaturally intense colors)",
            "Decorative overwhelm (pattern invading form)",
        ],
    )

    def __init__(self, model_path: Path | None = None, device: str = "mps"):
        super().__init__(model_path, device)
        self._pipe = None
        self._img2img_pipe = None

    def load_model(self) -> None:
        """Load SDXL for generation."""
        if self._pipe is not None:
            return

        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            use_safetensors=True,
            variant="fp16",
        )
        self._pipe = self._pipe.to(self.device)
        self._pipe.enable_attention_slicing()

        # Create img2img pipeline sharing the same components
        self._img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self._pipe.vae,
            text_encoder=self._pipe.text_encoder,
            text_encoder_2=self._pipe.text_encoder_2,
            tokenizer=self._pipe.tokenizer,
            tokenizer_2=self._pipe.tokenizer_2,
            unet=self._pipe.unet,
            scheduler=self._pipe.scheduler,
        )
        self._img2img_pipe = self._img2img_pipe.to(self.device)
        self._img2img_pipe.enable_attention_slicing()

        self._model = True

    def get_default_params(self) -> dict[str, Any]:
        return {
            "ornamental_encrustation": 0.6,   # Moreau's encrusted surfaces
            "mystery_gloss": 0.7,              # THE signature - ethereal sheen
            "golden_overflow": 0.5,            # Klimt's gold spreading
            "ethereal_softening": 0.5,         # Redon's dreamlike blur
            "symbolic_saturation": 0.6,        # Intense, meaningful color
            "decorative_overwhelm": 0.5,       # Pattern invasion
            "inference_steps": 30,
            "guidance_scale": 8.0,             # Higher for more "stylized"
            "img2img_strength": 0.65,
        }

    def generate(
        self,
        prompt: str | None = None,
        source_image: Image.Image | None = None,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Generate an image with Symbolism excess tells.

        Args:
            prompt: Text prompt for generation. Required unless source_image provided.
            source_image: Optional source image (if provided with prompt, uses img2img)
            control: Artifact control parameters
            **era_params: ornamental_encrustation, mystery_gloss, etc.
                         use_local: Force local generation (skip Replicate)

        Returns:
            Generated image with Symbolism excess tells
        """
        control = control or ArtifactControl()
        params = {**self.get_default_params(), **era_params}

        # Generation parameters
        num_steps = params.get("inference_steps", 30)
        guidance = params.get("guidance_scale", 8.0)
        use_local = era_params.get("use_local", False)

        # Determine generation mode
        if source_image is None:
            # txt2img mode
            if prompt is None:
                raise ValueError("Symbolism pipeline requires a prompt or source image")

            width = era_params.get("width", 1024)
            height = era_params.get("height", 1024)

            # Ensure dimensions are multiples of 8
            width = (width // 8) * 8
            height = (height // 8) * 8

            img = None

            # Try Replicate first (much faster than local SDXL)
            if not use_local:
                img = generate_via_replicate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_steps=num_steps,
                    guidance=guidance,
                    seed=control.seed,
                )

            # Fall back to local generation
            if img is None:
                print("Using local SDXL generation...")
                self.ensure_loaded()

                generator = None
                if control.seed is not None:
                    generator = torch.Generator(device=self.device).manual_seed(control.seed)
                    np.random.seed(control.seed)

                result = self._pipe(
                    prompt=prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    generator=generator,
                )
                img = result.images[0]

        elif prompt is not None:
            # img2img mode - diffuse from source image guided by prompt
            self.ensure_loaded()

            generator = None
            if control.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(control.seed)
                np.random.seed(control.seed)

            strength = params.get("img2img_strength", 0.65)

            result = self._img2img_pipe(
                prompt=prompt,
                image=source_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=generator,
            )
            img = result.images[0]
        else:
            # Just apply effects to source image, no diffusion
            img = source_image.copy()
            if control.seed is not None:
                np.random.seed(control.seed)

        original = img.copy()

        # Get effect strengths, scaling by global intensity
        intensity = control.intensity
        default_params = self.get_default_params()

        def get_effect_strength(param_name: str) -> float:
            """Get effect strength - use raw value if explicitly set, else scale by intensity."""
            value = params[param_name]
            default = default_params[param_name]
            if abs(value - default) > 0.01:
                return value
            return value * intensity * 1.5

        ornamental_encrustation = get_effect_strength("ornamental_encrustation")
        mystery_gloss = get_effect_strength("mystery_gloss")
        golden_overflow = get_effect_strength("golden_overflow")
        ethereal_softening = get_effect_strength("ethereal_softening")
        symbolic_saturation = get_effect_strength("symbolic_saturation")
        decorative_overwhelm = get_effect_strength("decorative_overwhelm")

        # Apply Symbolism tells in sequence
        # Order: base processing, then glows, then color, then pattern
        img = self._apply_ethereal_softening(img, ethereal_softening)
        img = self._apply_ornamental_encrustation(img, ornamental_encrustation)
        img = self._apply_golden_overflow(img, golden_overflow)
        img = self._apply_mystery_gloss(img, mystery_gloss)
        img = self._apply_symbolic_saturation(img, symbolic_saturation)
        img = self._apply_decorative_overwhelm(img, decorative_overwhelm)

        # Final blend with original based on placement
        img = control.apply_mask(original, img)

        # Upscale if requested
        upscale = era_params.get("upscale", 1)
        if upscale > 1:
            new_size = (img.width * upscale, img.height * upscale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def apply_tells(
        self,
        img: Image.Image,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Apply Symbolism excess tells to an existing image.

        This is the post-processing step only. Use this when you already have
        a base image (e.g., from batch Replicate generation) and want to apply
        the characteristic tells without re-generating.

        Args:
            img: Base image to apply tells to
            control: Artifact control parameters
            **era_params: ornamental_encrustation, mystery_gloss, etc.

        Returns:
            Image with Symbolism excess tells applied
        """
        control = control or ArtifactControl()
        params = {**self.get_default_params(), **era_params}

        if control.seed is not None:
            np.random.seed(control.seed)

        original = img.copy()

        # Get effect strengths, scaling by global intensity
        intensity = control.intensity
        default_params = self.get_default_params()

        def get_effect_strength(param_name: str) -> float:
            """Get effect strength - use raw value if explicitly set, else scale by intensity."""
            value = params[param_name]
            default = default_params[param_name]
            if abs(value - default) > 0.01:
                return value
            return value * intensity * 1.5

        ornamental_encrustation = get_effect_strength("ornamental_encrustation")
        mystery_gloss = get_effect_strength("mystery_gloss")
        golden_overflow = get_effect_strength("golden_overflow")
        ethereal_softening = get_effect_strength("ethereal_softening")
        symbolic_saturation = get_effect_strength("symbolic_saturation")
        decorative_overwhelm = get_effect_strength("decorative_overwhelm")

        # Apply Symbolism tells in sequence
        img = self._apply_ethereal_softening(img, ethereal_softening)
        img = self._apply_ornamental_encrustation(img, ornamental_encrustation)
        img = self._apply_golden_overflow(img, golden_overflow)
        img = self._apply_mystery_gloss(img, mystery_gloss)
        img = self._apply_symbolic_saturation(img, symbolic_saturation)
        img = self._apply_decorative_overwhelm(img, decorative_overwhelm)

        # Final blend with original based on placement
        img = control.apply_mask(original, img)

        # Upscale if requested
        upscale = era_params.get("upscale", 1)
        if upscale > 1:
            new_size = (img.width * upscale, img.height * upscale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _apply_ethereal_softening(self, img: Image.Image, strength: float) -> Image.Image:
        """Redon's DREAMLIKE BLUR. Soft-focused, otherworldly.

        Odilon Redon's work has a distinctive soft, dreamlike quality where
        edges dissolve into atmosphere and forms float in undefined space.
        Everything seems to exist in a reverie, just slightly out of focus.

        Technique:
        - Apply selective blur that preserves some detail
        - Create hazy, atmospheric feel
        - Soften edges while keeping some structure
        - Result: dreamlike, otherworldly quality
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        # Create soft blur
        blur_sigma = 2 + strength * 4
        blurred = np.stack([
            ndimage.gaussian_filter(arr[:, :, c], sigma=blur_sigma)
            for c in range(3)
        ], axis=2)

        # Detect edges to preserve some definition
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        edges = np.sqrt(
            ndimage.sobel(gray, axis=0)**2 +
            ndimage.sobel(gray, axis=1)**2
        )
        if edges.max() > 0:
            edges = edges / edges.max()

        # Blur less at edges
        edge_preserve = edges[:, :, np.newaxis] * 0.7
        blur_mask = np.clip(1 - edge_preserve, 0.3, 1) * strength * 0.8

        result = arr * (1 - blur_mask) + blurred * blur_mask

        # Add slight orton effect (blur mixed with sharp for glow)
        bright_blur = np.stack([
            ndimage.gaussian_filter(arr[:, :, c], sigma=blur_sigma * 2)
            for c in range(3)
        ], axis=2)

        # Screen blend for ethereal glow
        screen_blend = 255 - (255 - arr) * (255 - bright_blur) / 255
        orton_blend = strength * 0.25
        result = result * (1 - orton_blend) + screen_blend * orton_blend

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_ornamental_encrustation(self, img: Image.Image, strength: float) -> Image.Image:
        """Moreau's ENCRUSTED SURFACES. Every surface detailed to excess.

        Gustave Moreau covered every surface with intricate detail, jewels,
        and ornamentation until the image became almost suffocating with
        visual information. Nothing is plain; everything is encrusted.

        Technique:
        - Enhance micro-contrast to bring out detail
        - Add fine texture/grain that suggests encrustation
        - Sharpen selectively to make details pop
        - Result: surfaces feel encrusted with ornament
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # High-frequency enhancement (bring out fine detail)
        blur_small = np.stack([
            ndimage.gaussian_filter(arr[:, :, c], sigma=1)
            for c in range(3)
        ], axis=2)

        blur_large = np.stack([
            ndimage.gaussian_filter(arr[:, :, c], sigma=5)
            for c in range(3)
        ], axis=2)

        # High-frequency detail
        high_freq = arr - blur_small
        mid_freq = blur_small - blur_large

        # Boost detail
        detail_boost = 1 + strength * 1.5
        result = arr + high_freq * (detail_boost - 1) * 0.8 + mid_freq * (detail_boost - 1) * 0.4

        # Add fine texture (suggests jewels, ornament)
        np.random.seed(42)  # Consistent texture
        texture = np.random.randn(h, w) * 8 * strength
        texture = ndimage.gaussian_filter(texture, sigma=0.8)

        # Modulate texture by luminance (more visible in midtones)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b

        # Bell curve centered on midtones
        midtone_mask = np.exp(-((lum - 128) / 80)**2)
        texture = texture * midtone_mask

        result = result + texture[:, :, np.newaxis]

        # Add shimmer/sparkle in highlights
        highlight_mask = np.clip((lum - 180) / 75, 0, 1)
        sparkle = np.random.randn(h, w) * 15 * strength
        sparkle = ndimage.gaussian_filter(sparkle, sigma=0.5)
        sparkle = sparkle * highlight_mask

        result = result + sparkle[:, :, np.newaxis] * np.array([1.0, 0.95, 0.85])

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_golden_overflow(self, img: Image.Image, strength: float) -> Image.Image:
        """Klimt's GOLD SPREADING. Gold leaf overflowing its bounds.

        Gustav Klimt's "Golden Phase" (1899-1910) featured actual gold leaf
        applied to paintings, with the gold spreading beyond decorative
        elements to invade the entire composition. The decorative overwhelms
        the figurative.

        Technique:
        - Detect warm/highlight areas
        - Add golden tint that spreads from highlights
        - Create metallic sheen effect
        - Result: gold leaf spreading across the image
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Detect warm areas and highlights
        warmth = np.clip((r - b) / 255 + 0.3, 0, 1)
        lum = 0.299 * r + 0.587 * g + 0.114 * b

        # Gold appears in highlights and warm areas
        gold_affinity = warmth * 0.6 + np.clip((lum - 100) / 155, 0, 1) * 0.4

        # Spread the gold (blur the mask)
        gold_spread = ndimage.gaussian_filter(gold_affinity, sigma=20 * strength)
        gold_spread = np.clip(gold_spread * 1.5, 0, 1)

        # Gold color palette
        gold_r = np.array([255, 215, 175])  # Bright gold
        gold_g = np.array([200, 175, 140])  # Deeper gold
        gold_b = np.array([50, 40, 30])     # Shadow gold

        # Interpolate gold based on luminance
        lum_norm = lum / 255
        gold_color = np.zeros((h, w, 3))
        gold_color[:, :, 0] = gold_r[0] * lum_norm + gold_r[2] * (1 - lum_norm)
        gold_color[:, :, 1] = gold_g[0] * lum_norm + gold_g[2] * (1 - lum_norm)
        gold_color[:, :, 2] = gold_b[0] * lum_norm + gold_b[2] * (1 - lum_norm)

        # Add metallic variation
        np.random.seed(43)
        metallic_noise = np.random.randn(h, w) * 10
        metallic_noise = ndimage.gaussian_filter(metallic_noise, sigma=3)
        gold_color[:, :, 0] += metallic_noise
        gold_color[:, :, 1] += metallic_noise * 0.8

        # Apply gold overlay
        gold_mask = gold_spread[:, :, np.newaxis] * strength * 0.5
        result = arr * (1 - gold_mask) + gold_color * gold_mask

        # Add gold shimmer in highlights
        shimmer_mask = np.clip((lum - 160) / 95, 0, 1) * gold_spread * strength
        shimmer = np.array([40, 30, 0])
        result = result + shimmer_mask[:, :, np.newaxis] * shimmer

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_mystery_gloss(self, img: Image.Image, strength: float) -> Image.Image:
        """THE MIDJOURNEY SHEEN. Everything glows with mystique.

        This is THE signature Midjourney tell: an ethereal sheen that makes
        everything look important, mysterious, magical. Rim lights appear
        from nowhere. Surfaces have an inner glow. Everything is backlit
        by significance.

        Technique:
        - Add rim lighting effect around forms
        - Create inner glow on surfaces
        - Apply bloom on highlights
        - Result: everything has mystical significance
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Detect edges for rim lighting
        edges_x = ndimage.sobel(gray, axis=1)
        edges_y = ndimage.sobel(gray, axis=0)
        edge_mag = np.sqrt(edges_x**2 + edges_y**2)
        if edge_mag.max() > 0:
            edge_mag = edge_mag / edge_mag.max()

        # Rim light (glow at edges)
        rim_glow = ndimage.gaussian_filter(edge_mag, sigma=4)
        rim_glow = np.clip(rim_glow * 2, 0, 1)

        # Warm rim light color (the Midjourney amber)
        rim_color = np.array([255, 220, 180])
        rim_strength = rim_glow[:, :, np.newaxis] * strength * 0.4
        result = arr + rim_color * rim_strength

        # Highlight bloom
        bright_mask = np.clip((gray - 180) / 75, 0, 1)
        bloom = ndimage.gaussian_filter(bright_mask, sigma=15)
        bloom_color = np.array([255, 245, 230])  # Warm white
        bloom_strength = bloom[:, :, np.newaxis] * strength * 0.5
        result = result + bloom_color * bloom_strength

        # Inner glow (everything has subtle luminosity)
        inner_blur = np.stack([
            ndimage.gaussian_filter(arr[:, :, c], sigma=30)
            for c in range(3)
        ], axis=2)

        # Screen blend for inner glow
        inner_glow = 255 - (255 - result) * (255 - inner_blur * 0.3) / 255
        glow_blend = strength * 0.3
        result = result * (1 - glow_blend) + inner_glow * glow_blend

        # Add atmospheric haze (mystery in the shadows)
        dark_mask = np.clip((100 - gray) / 100, 0, 1)
        haze_color = np.array([100, 90, 120])  # Mysterious purple-blue
        haze_blur = ndimage.gaussian_filter(dark_mask, sigma=25)
        haze_strength = haze_blur[:, :, np.newaxis] * strength * 0.25
        result = result + haze_color * haze_strength

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_symbolic_saturation(self, img: Image.Image, strength: float) -> Image.Image:
        """MEANINGFUL COLOR. Every hue is symbolic, significant.

        Symbolists used color not for naturalism but for meaning. Blues
        meant melancholy, reds meant passion, purples meant mysticism.
        AI stylization pushes this further: colors become unnaturally
        saturated, every hue screaming its significance.

        Technique:
        - Boost saturation overall
        - Push hues toward their symbolic extremes
        - Enhance color contrasts
        - Result: colors feel symbolically charged
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Boost saturation
        sat_boost = 1 + strength * 0.6

        r_sat = gray + (r - gray) * sat_boost
        g_sat = gray + (g - gray) * sat_boost
        b_sat = gray + (b - gray) * sat_boost

        # Push colors toward symbolic extremes
        # Blues become more purple (mysticism)
        # Reds become more crimson (passion)
        # Greens become more golden (nature/growth)

        # Detect dominant hue
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        chroma = max_c - min_c + 0.001

        # Blue push toward purple
        blue_dom = np.clip((b - np.maximum(r, g)) / chroma, 0, 1)
        r_sat = r_sat + blue_dom * 30 * strength  # Add red to blues

        # Red push toward crimson
        red_dom = np.clip((r - np.maximum(g, b)) / chroma, 0, 1)
        b_sat = b_sat + red_dom * 20 * strength  # Add blue to reds (crimson)

        # Green push toward gold
        green_dom = np.clip((g - np.maximum(r, b)) / chroma, 0, 1)
        r_sat = r_sat + green_dom * 25 * strength  # Add red to greens (gold)

        # Enhance color contrast
        for channel in [r_sat, g_sat, b_sat]:
            channel_mean = np.mean(channel)
            channel[:] = channel_mean + (channel - channel_mean) * (1 + strength * 0.3)

        result = np.stack([r_sat, g_sat, b_sat], axis=2)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_decorative_overwhelm(self, img: Image.Image, strength: float) -> Image.Image:
        """PATTERN INVADING FORM. The decorative overwhelming the figurative.

        In Klimt's work, decorative patterns don't stay in their place. They
        spread from clothing to skin to background. The pattern becomes the
        subject. Similarly, Midjourney's stylization makes everything feel
        like it's part of one overwhelming decorative scheme.

        Technique:
        - Create subtle pattern overlay
        - Pattern intensity follows existing detail
        - Geometric/organic pattern mix
        - Result: decorative pattern invading everywhere
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Create coordinate grids for pattern generation
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Multiple pattern layers (Klimt's complexity)

        # 1. Spiral/swirl pattern (organic)
        cx, cy = w / 2, h / 2
        dx = x_coords - cx
        dy = y_coords - cy
        r = np.sqrt(dx**2 + dy**2) / (min(w, h) / 2)
        theta = np.arctan2(dy, dx)

        spiral = np.sin(r * 10 + theta * 3) * 0.5 + 0.5

        # 2. Geometric pattern (Art Nouveau geometry)
        scale = 50 / (1 + strength * 0.5)
        geo1 = np.sin(x_coords / scale) * np.sin(y_coords / scale)
        geo2 = np.cos((x_coords + y_coords) / scale * 0.7)
        geometric = (geo1 + geo2) * 0.25 + 0.5

        # 3. Mosaic pattern (tesserae feel)
        mosaic_scale = int(20 - strength * 10)
        mosaic_scale = max(5, mosaic_scale)
        mosaic = np.zeros((h, w))
        for yi in range(0, h, mosaic_scale):
            for xi in range(0, w, mosaic_scale):
                value = np.random.random() * 0.3 + 0.35
                y_end = min(yi + mosaic_scale, h)
                x_end = min(xi + mosaic_scale, w)
                mosaic[yi:y_end, xi:x_end] = value

        # Combine patterns
        pattern = (spiral * 0.4 + geometric * 0.35 + mosaic * 0.25)
        pattern = ndimage.gaussian_filter(pattern, sigma=1)  # Smooth slightly

        # Detect where to apply pattern (more on uniform areas, less on detail)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        local_var = ndimage.generic_filter(gray, np.std, size=11)
        if local_var.max() > 0:
            local_var = local_var / local_var.max()

        # Pattern stronger where there's less local variation
        pattern_mask = (1 - local_var * 0.6) * strength * 0.4

        # Apply pattern as overlay
        # Use pattern to modulate luminance slightly
        pattern_value = (pattern - 0.5) * 40  # Shape (h, w)
        pattern_scaled = pattern_value * pattern_mask  # Shape (h, w)
        pattern_offset = pattern_scaled[:, :, np.newaxis]  # Shape (h, w, 1)
        result = arr + pattern_offset

        # Add gold tint to pattern highlights
        pattern_highlights = np.clip((pattern - 0.6) * 3, 0, 1) * pattern_mask  # Shape (h, w)
        gold_tint = np.array([20, 15, -5])
        result = result + pattern_highlights[:, :, np.newaxis] * gold_tint * strength

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
