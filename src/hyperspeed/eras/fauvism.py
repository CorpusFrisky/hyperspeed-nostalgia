"""Fauvism Era (1904-1908) / The Golden Age of Slop (2022-2023) parallel.

"When I put a green, it is not grass. When I put a blue, it is not the sky."
- Henri Matisse

Fauvism was called "les fauves" (wild beasts) for its violent, unnatural color.
The 2022-2023 AI era had its own color violence: the characteristic teal-and-
orange palette that screamed "AI-generated." Both movements reclaim what critics
called derision as a visual vocabulary worth preserving.

The Golden Age of Slop (2022-2023):
- Midjourney v3-v4 reached maturity but hadn't achieved photorealism
- DALL-E 2 was at peak usage
- Stable Diffusion 1.x became public
- Images had characteristic teal-and-orange color grading
- Saturation pushed to maximum
- Color became emotionally expressive rather than naturalistic

Key works:
- Matisse's Woman with a Hat (1905): a riot of arbitrary color, the face
  rendered in green and violet strokes, shocking audiences
- Derain's Charing Cross Bridge (1906): London rendered in impossible oranges
  and blues, reality subordinated to color expression
- Vlaminck's The River (1910): pure tubes of paint applied directly, no
  mixing, maximum chromatic violence

Deep Cut: Both Fauvism and AI slop embrace what critics despise. "Les fauves"
was an insult that became a movement name. "Slop" is derision reclaimed as
aesthetic. Both prove that unnatural color, pushed to extremes, creates its
own coherent visual language.

The Tell: Unlike Symbolism (excessive mystique) or Post-Impressionism (structural
dissolution), Fauvism is about COLOR VIOLENCE. The 2022-2023 teal-orange cast
is as recognizable as the six-fingered hand. Colors clash, saturate, and glow
with emotional rather than naturalistic intent.

Technical approach:
1. Generate image with SDXL via Replicate API (fast) or local (fallback)
2. Apply characteristic Fauvist/Golden Age color violence locally:
   - Saturation violence (push all colors toward maximum)
   - Hue distortion (shift colors away from naturalism)
   - Teal-orange cast (THE Golden Age signature)
   - Color clash (complementary colors forced together)
   - Chromatic intensity (luminosity pushed to extremes)
   - Pigment expressiveness (colors "straight from the tube")

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
class FauvismPipeline(EraPipeline):
    """Fauvism Era: The Golden Age of Slop (2022-2023).

    This pipeline generates images with the characteristic color violence
    of Fauvism and the 2022-2023 AI aesthetic. Unnatural color is a feature,
    not a bug.

    Unlike Symbolism (excessive mystique) or Post-Impressionism (structural
    dissolution), Fauvism fails through COLOR VIOLENCE. The teal-orange cast
    is as iconic to 2022-2023 as the six-fingered hand.

    Controls:
        saturation_violence: Push all colors toward maximum (0-1)
        hue_distortion: Shift colors away from naturalism (0-1)
        teal_orange_cast: THE Golden Age signature color grading (0-1)
        color_clash: Complementary colors forced together (0-1)
        chromatic_intensity: Luminosity pushed to extremes (0-1)
        pigment_expressiveness: Colors "straight from the tube" (0-1)
    """

    metadata: ClassVar[EraMetadata] = EraMetadata(
        name="Fauvism",
        art_historical_parallel="Fauvism (1904-1908)",
        time_period="The Golden Age of Slop (2022-2023)",
        description=(
            "Color violence as virtue. Matisse's 'when I put green, it is not grass' "
            "meets the teal-and-orange AI palette. Both movements reclaim what critics "
            "call derision as visual vocabulary. Unnatural color pushed to extremes."
        ),
        characteristic_artifacts=[
            "Saturation violence (colors pushed toward maximum)",
            "Hue distortion (colors shifted from naturalism)",
            "Teal-orange cast (THE Golden Age signature)",
            "Color clash (complementary colors forced together)",
            "Chromatic intensity (luminosity at extremes)",
            "Pigment expressiveness (unmixed, straight from tube)",
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
            "saturation_violence": 0.8,      # THE Fauvist signature - max saturation
            "hue_distortion": 0.7,           # Shift from naturalism
            "teal_orange_cast": 0.75,        # THE Golden Age AI signature
            "color_clash": 0.6,              # Complementary color violence
            "chromatic_intensity": 0.5,      # Push luminosity extremes
            "pigment_expressiveness": 0.6,   # Unmixed tube paint feel
            "inference_steps": 30,
            "guidance_scale": 7.5,
            "img2img_strength": 0.65,
        }

    def generate(
        self,
        prompt: str | None = None,
        source_image: Image.Image | None = None,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Generate an image with Fauvist color violence tells.

        Args:
            prompt: Text prompt for generation. Required unless source_image provided.
            source_image: Optional source image (if provided with prompt, uses img2img)
            control: Artifact control parameters
            **era_params: saturation_violence, hue_distortion, etc.
                         use_local: Force local generation (skip Replicate)

        Returns:
            Generated image with Fauvist color violence tells
        """
        control = control or ArtifactControl()
        params = {**self.get_default_params(), **era_params}

        # Generation parameters
        num_steps = params.get("inference_steps", 30)
        guidance = params.get("guidance_scale", 7.5)
        use_local = era_params.get("use_local", False)

        # Determine generation mode
        if source_image is None:
            # txt2img mode
            if prompt is None:
                raise ValueError("Fauvism pipeline requires a prompt or source image")

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

        saturation_violence = get_effect_strength("saturation_violence")
        hue_distortion = get_effect_strength("hue_distortion")
        teal_orange_cast = get_effect_strength("teal_orange_cast")
        color_clash = get_effect_strength("color_clash")
        chromatic_intensity = get_effect_strength("chromatic_intensity")
        pigment_expressiveness = get_effect_strength("pigment_expressiveness")

        # Apply Fauvist tells in sequence
        # Order: base saturation, then hue shifts, then cast, then clash, then extremes
        img = self._apply_saturation_violence(img, saturation_violence)
        img = self._apply_hue_distortion(img, hue_distortion)
        img = self._apply_teal_orange_cast(img, teal_orange_cast)
        img = self._apply_color_clash(img, color_clash)
        img = self._apply_chromatic_intensity(img, chromatic_intensity)
        img = self._apply_pigment_expressiveness(img, pigment_expressiveness)

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
        """Apply Fauvist color violence tells to an existing image.

        This is the post-processing step only. Use this when you already have
        a base image (e.g., from batch Replicate generation) and want to apply
        the characteristic tells without re-generating.

        Args:
            img: Base image to apply tells to
            control: Artifact control parameters
            **era_params: saturation_violence, hue_distortion, etc.

        Returns:
            Image with Fauvist color violence tells applied
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

        saturation_violence = get_effect_strength("saturation_violence")
        hue_distortion = get_effect_strength("hue_distortion")
        teal_orange_cast = get_effect_strength("teal_orange_cast")
        color_clash = get_effect_strength("color_clash")
        chromatic_intensity = get_effect_strength("chromatic_intensity")
        pigment_expressiveness = get_effect_strength("pigment_expressiveness")

        # Apply Fauvist tells in sequence
        img = self._apply_saturation_violence(img, saturation_violence)
        img = self._apply_hue_distortion(img, hue_distortion)
        img = self._apply_teal_orange_cast(img, teal_orange_cast)
        img = self._apply_color_clash(img, color_clash)
        img = self._apply_chromatic_intensity(img, chromatic_intensity)
        img = self._apply_pigment_expressiveness(img, pigment_expressiveness)

        # Final blend with original based on placement
        img = control.apply_mask(original, img)

        # Upscale if requested
        upscale = era_params.get("upscale", 1)
        if upscale > 1:
            new_size = (img.width * upscale, img.height * upscale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _apply_saturation_violence(self, img: Image.Image, strength: float) -> Image.Image:
        """MAXIMUM SATURATION. Colors pushed beyond natural limits.

        Matisse and the Fauves applied color at full intensity, straight from
        the tube. No careful mixing, no subtle gradation - just raw chromatic
        power. AI slop does the same: everything oversaturated, everything
        screaming for attention.

        Technique:
        - Boost saturation aggressively
        - Expand color gamut beyond natural
        - Push midtones toward vivid extremes
        - Result: colors that feel "too much"
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Aggressive saturation boost (Fauvist violence)
        sat_boost = 1 + strength * 1.2  # Much stronger than Symbolism

        r_sat = gray + (r - gray) * sat_boost
        g_sat = gray + (g - gray) * sat_boost
        b_sat = gray + (b - gray) * sat_boost

        # Expand gamut beyond natural (push away from gray even more)
        gamut_expansion = 1 + strength * 0.4
        mean_color = np.stack([r_sat, g_sat, b_sat], axis=2).mean()

        r_sat = mean_color + (r_sat - mean_color) * gamut_expansion
        g_sat = mean_color + (g_sat - mean_color) * gamut_expansion
        b_sat = mean_color + (b_sat - mean_color) * gamut_expansion

        # Push midtones toward vivid (compress dynamic range around vivid center)
        midtone_target = 140  # Slightly bright center
        r_sat = midtone_target + (r_sat - midtone_target) * (1 - strength * 0.15)
        g_sat = midtone_target + (g_sat - midtone_target) * (1 - strength * 0.15)
        b_sat = midtone_target + (b_sat - midtone_target) * (1 - strength * 0.15)

        result = np.stack([r_sat, g_sat, b_sat], axis=2)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_hue_distortion(self, img: Image.Image, strength: float) -> Image.Image:
        """COLORS SHIFTED FROM NATURE. Green faces, purple skies.

        "When I put a green, it is not grass" - Matisse's manifesto. Colors
        are freed from representing what they "should" represent. Skin can
        be green, shadows can be orange, skies can be whatever serves the
        emotional expression.

        Technique:
        - Rotate hues away from naturalistic associations
        - Greens toward cyan, reds toward magenta
        - Blues toward purple, yellows toward orange
        - Result: colors that feel emotionally true but visually wrong
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Detect dominant hue regions
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        chroma = max_c - min_c + 0.001

        # Hue rotation based on dominant channel
        # This creates the "wrong" but emotionally expressive colors

        # Red-dominant → push toward magenta (add blue)
        red_dom = np.clip((r - np.maximum(g, b)) / chroma, 0, 1)
        b_shifted = b + red_dom * 45 * strength

        # Green-dominant → push toward cyan (add blue, reduce red)
        green_dom = np.clip((g - np.maximum(r, b)) / chroma, 0, 1)
        b_shifted = b_shifted + green_dom * 50 * strength
        r_shifted = r - green_dom * 30 * strength

        # Blue-dominant → push toward purple (add red)
        blue_dom = np.clip((b - np.maximum(r, g)) / chroma, 0, 1)
        r_shifted = r_shifted + blue_dom * 40 * strength

        # Yellow (r+g dominant) → push toward orange (boost red more)
        yellow_dom = np.clip((np.minimum(r, g) - b) / chroma, 0, 1)
        r_shifted = r_shifted + yellow_dom * 30 * strength
        g_shifted = g - yellow_dom * 15 * strength

        # Ensure g_shifted exists even if yellow_dom didn't trigger
        if 'g_shifted' not in dir():
            g_shifted = g.copy()

        result = np.stack([r_shifted, g_shifted, b_shifted], axis=2)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_teal_orange_cast(self, img: Image.Image, strength: float) -> Image.Image:
        """THE GOLDEN AGE SIGNATURE. Teal shadows, orange highlights.

        This is THE recognizable tell of 2022-2023 AI imagery. Every shadow
        went teal/cyan, every highlight went warm orange. It became so
        ubiquitous that it's now a period marker, like the sepia of old
        photographs or the grain of 1970s film.

        Technique:
        - Map shadows toward teal/cyan
        - Map highlights toward warm orange
        - Create split toning effect
        - Result: the unmistakable "2022 AI" look
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Shadow mask (darker areas get teal)
        shadow_mask = np.clip((128 - gray) / 128, 0, 1) ** 1.5

        # Highlight mask (brighter areas get orange)
        highlight_mask = np.clip((gray - 128) / 127, 0, 1) ** 1.5

        # Teal color for shadows (high cyan = high G+B, low R)
        teal_shift_r = -40 * shadow_mask * strength
        teal_shift_g = 20 * shadow_mask * strength
        teal_shift_b = 50 * shadow_mask * strength

        # Orange color for highlights (high R, medium G, low B)
        orange_shift_r = 50 * highlight_mask * strength
        orange_shift_g = 20 * highlight_mask * strength
        orange_shift_b = -30 * highlight_mask * strength

        # Apply shifts
        r_cast = r + teal_shift_r + orange_shift_r
        g_cast = g + teal_shift_g + orange_shift_g
        b_cast = b + teal_shift_b + orange_shift_b

        # Add midtone warmth (the "glow" of AI images)
        midtone_mask = np.exp(-((gray - 128) / 60)**2)
        r_cast = r_cast + midtone_mask * 15 * strength
        g_cast = g_cast + midtone_mask * 5 * strength

        result = np.stack([r_cast, g_cast, b_cast], axis=2)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_color_clash(self, img: Image.Image, strength: float) -> Image.Image:
        """COMPLEMENTARY VIOLENCE. Colors that fight each other.

        The Fauves placed complementary colors directly adjacent: red against
        green, orange against blue, yellow against violet. No smooth transitions,
        just chromatic conflict. AI slop does this accidentally through its
        tendency to maximize local contrast.

        Technique:
        - Detect edge regions
        - Push adjacent colors toward complements
        - Increase local color contrast
        - Result: colors that vibrate against each other
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Detect edges where color clash should happen
        edges_x = ndimage.sobel(gray, axis=1)
        edges_y = ndimage.sobel(gray, axis=0)
        edge_mag = np.sqrt(edges_x**2 + edges_y**2)
        if edge_mag.max() > 0:
            edge_mag = edge_mag / edge_mag.max()

        # Dilate edges slightly for effect zone
        clash_zone = ndimage.gaussian_filter(edge_mag, sigma=3)
        clash_zone = np.clip(clash_zone * 2, 0, 1)

        # Local color contrast enhancement
        # For each pixel near edge, push it away from neighbors
        r_blur = ndimage.gaussian_filter(r, sigma=8)
        g_blur = ndimage.gaussian_filter(g, sigma=8)
        b_blur = ndimage.gaussian_filter(b, sigma=8)

        # Push away from local average (increases color contrast)
        contrast_boost = 1 + strength * 0.6
        r_clash = r_blur + (r - r_blur) * contrast_boost
        g_clash = g_blur + (g - g_blur) * contrast_boost
        b_clash = b_blur + (b - b_blur) * contrast_boost

        # Add complementary push in clash zones
        # Where red dominates, push neighbors toward cyan
        # Where green dominates, push neighbors toward magenta
        r_dom = np.clip((r - np.maximum(g, b)) / 100, -1, 1)
        g_dom = np.clip((g - np.maximum(r, b)) / 100, -1, 1)
        b_dom = np.clip((b - np.maximum(r, g)) / 100, -1, 1)

        # Shifted dominance (what neighbors should tend toward)
        r_dom_shifted = ndimage.uniform_filter(r_dom, size=15)
        g_dom_shifted = ndimage.uniform_filter(g_dom, size=15)
        b_dom_shifted = ndimage.uniform_filter(b_dom, size=15)

        # Apply complementary push in clash zones
        complement_strength = clash_zone * strength * 25
        r_clash = r_clash - g_dom_shifted * complement_strength - b_dom_shifted * complement_strength * 0.5
        g_clash = g_clash - r_dom_shifted * complement_strength * 0.5 - b_dom_shifted * complement_strength * 0.5
        b_clash = b_clash - r_dom_shifted * complement_strength * 0.5 - g_dom_shifted * complement_strength

        # Blend based on edge proximity
        blend_mask = clash_zone[:, :, np.newaxis] * 0.7 + 0.3
        result = arr * (1 - blend_mask * strength) + np.stack([r_clash, g_clash, b_clash], axis=2) * (blend_mask * strength)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_chromatic_intensity(self, img: Image.Image, strength: float) -> Image.Image:
        """LUMINOSITY AT EXTREMES. Midtones flattened, poles amplified.

        Fauvist paintings often have a strange flatness to their values even
        as their colors are extreme. The midtones compress while highlights
        and shadows push toward their limits. AI slop similarly tends toward
        extreme value distributions.

        Technique:
        - Compress midtones (flatten the middle)
        - Push highlights brighter
        - Push shadows darker (but keep them colored)
        - Result: values feel "strained" rather than natural
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # S-curve for contrast enhancement (compress midtones, expand extremes)
        # Normalized gray for curve calculation
        gray_norm = gray / 255

        # S-curve: sigmoid-like compression of midtones
        # More extreme strength = more S-curve
        curve_strength = 1 + strength * 2
        # Apply curve: pushes values toward 0 and 1
        curved = 1 / (1 + np.exp(-curve_strength * (gray_norm - 0.5) * 4))

        # Calculate ratio for applying to color channels
        ratio = np.where(gray > 0.01, (curved * 255) / gray, 1)
        ratio = np.clip(ratio, 0.3, 3)  # Prevent extreme ratios

        # Apply ratio while preserving color relationships
        r_intense = r * ratio
        g_intense = g * ratio
        b_intense = b * ratio

        # Boost pure colors in highlights (the Fauvist glow)
        highlight_mask = np.clip((gray - 180) / 75, 0, 1)
        max_channel = np.maximum(np.maximum(r, g), b)
        color_purity = np.where(max_channel > 10,
                                (max_channel - np.minimum(np.minimum(r, g), b)) / max_channel,
                                0)

        # Boost the dominant channel in highlights
        boost_amount = highlight_mask * color_purity * strength * 30
        r_dominant = (r == max_channel).astype(float)
        g_dominant = (g == max_channel).astype(float)
        b_dominant = (b == max_channel).astype(float)

        r_intense = r_intense + boost_amount * r_dominant
        g_intense = g_intense + boost_amount * g_dominant
        b_intense = b_intense + boost_amount * b_dominant

        result = np.stack([r_intense, g_intense, b_intense], axis=2)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_pigment_expressiveness(self, img: Image.Image, strength: float) -> Image.Image:
        """STRAIGHT FROM THE TUBE. Unmixed, visible color application.

        Vlaminck boasted of squeezing paint directly from tube to canvas.
        No palette mixing, no subtle gradation - just raw, unmixed pigment.
        The result is visible color separation, colors that refuse to blend.
        AI slop has similar sharp color transitions at boundaries.

        Technique:
        - Reduce color blending at boundaries
        - Quantize colors slightly (reduce smooth gradients)
        - Add color separation at edges
        - Result: colors feel applied, not blended
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Detect color boundaries (where we want separation)
        r_grad = np.sqrt(ndimage.sobel(r, axis=0)**2 + ndimage.sobel(r, axis=1)**2)
        g_grad = np.sqrt(ndimage.sobel(g, axis=0)**2 + ndimage.sobel(g, axis=1)**2)
        b_grad = np.sqrt(ndimage.sobel(b, axis=0)**2 + ndimage.sobel(b, axis=1)**2)

        color_edges = np.maximum(np.maximum(r_grad, g_grad), b_grad)
        if color_edges.max() > 0:
            color_edges = color_edges / color_edges.max()

        # Slight color quantization (posterization effect)
        # This simulates the "unmixed pigment" look
        levels = int(16 - strength * 8)  # More strength = fewer levels
        levels = max(4, levels)

        r_quant = np.round(r / 255 * levels) * 255 / levels
        g_quant = np.round(g / 255 * levels) * 255 / levels
        b_quant = np.round(b / 255 * levels) * 255 / levels

        # Blend quantized more strongly at color edges
        edge_blend = color_edges * strength * 0.6
        r_express = r * (1 - edge_blend) + r_quant * edge_blend
        g_express = g * (1 - edge_blend) + g_quant * edge_blend
        b_express = b * (1 - edge_blend) + b_quant * edge_blend

        # Add color fringing at edges (chromatic aberration-like)
        # Different channels shift slightly at boundaries
        shift_amount = int(2 + strength * 3)

        # Create shifted versions
        r_shifted = np.roll(r_express, shift_amount, axis=1)
        b_shifted = np.roll(b_express, -shift_amount, axis=1)

        # Apply shift only at color edges
        edge_mask = ndimage.gaussian_filter(color_edges, sigma=2)
        edge_mask = edge_mask * strength * 0.4

        r_express = r_express * (1 - edge_mask) + r_shifted * edge_mask
        b_express = b_express * (1 - edge_mask) + b_shifted * edge_mask

        # Subtle texture to suggest brushwork (visible paint application)
        np.random.seed(44)
        brush_texture = np.random.randn(h, w) * 6 * strength
        brush_texture = ndimage.gaussian_filter(brush_texture, sigma=1.5)

        # Apply texture more to midtones
        midtone_mask = np.exp(-((gray - 128) / 80)**2)
        brush_texture = brush_texture * midtone_mask

        r_express = r_express + brush_texture
        g_express = g_express + brush_texture
        b_express = b_express + brush_texture

        result = np.stack([r_express, g_express, b_express], axis=2)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
