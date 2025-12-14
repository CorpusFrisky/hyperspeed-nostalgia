"""Impressionism Era (1860-1890) / The Melting Zone parallel.

Boundaries dissolving. Probabilistic edges. Light over form.

Monet's Impression, Sunrise (1872) gave the movement its name, and critics
called it "unfinished," a mere impression. Impressionists rendered "optical
data" rather than "what we know about space, mass, and the other physical
details of the world."

Key works:
- Monet's series paintings (Haystacks, Rouen Cathedral, Water Lilies): the
  same subject at different times of day, boundaries between sky and water
  "hardly distinguishable"
- The paint "becomes the place" in its "foggy blankness, its featureless,
  expectant emptiness"

Deep Cut: Art historian's description of Impression, Sunrise: "colors melding
together in its glooming, opalescent oneness." This is the melting zone
perfectly described.

The Tell: Unlike Neoclassicism (over-idealization) or High Renaissance
(over-dramatization), Impressionism is about DISSOLUTION. Boundaries become
probabilistic rather than defined. Forms melt into light.

Technical approach:
1. Generate image with SDXL via Replicate API (fast) or local (fallback)
2. Apply characteristic Impressionist dissolution effects locally:
   - Edge dissolution (boundaries become uncertain)
   - Color bleed (colors meld across boundaries)
   - Atmospheric haze (foggy, expectant emptiness)
   - Brushstroke texture (visible paint application)
   - Light fragmentation (broken color, optical mixing)
   - Temporal blur (the captured moment)

Environment:
- Set REPLICATE_API_TOKEN to use Replicate for fast generation
- Falls back to local SDXL if token not set or API fails
"""

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

from hyperspeed.core.artifact_control import ArtifactControl
from hyperspeed.core.replicate_utils import (
    generate_via_replicate,
    batch_generate_replicate,
)
from hyperspeed.eras.base import EraMetadata, EraPipeline, EraRegistry


@EraRegistry.register
class ImpressionismPipeline(EraPipeline):
    """Impressionism Era: The Melting Zone.

    This pipeline generates images with the characteristic "dissolution"
    quality of Impressionism, where boundaries become probabilistic and
    forms melt into light.

    Unlike High Renaissance (over-dramatization) or Neoclassicism
    (over-idealization), Impressionism is about dissolution. The tell
    isn't failure or excess; it's uncertainty made visible.

    Controls:
        edge_dissolution: Boundaries become probabilistic (0-1)
        color_bleed: Colors meld across boundaries (0-1)
        atmospheric_haze: Foggy, expectant emptiness (0-1)
        brushstroke_texture: Visible paint application (0-1)
        light_fragmentation: Broken color, optical mixing (0-1)
        temporal_blur: The captured moment (0-1)
    """

    metadata: ClassVar[EraMetadata] = EraMetadata(
        name="Impressionism",
        art_historical_parallel="Impressionism (1860-1890)",
        time_period="The Melting Zone",
        description=(
            "Boundaries dissolving. Probabilistic edges. Light over form. "
            "Colors melding together in opalescent oneness. "
            "The tell isn't failure; it's uncertainty made visible."
        ),
        characteristic_artifacts=[
            "Edge dissolution (probabilistic boundaries)",
            "Color bleed (chromatic melding)",
            "Atmospheric haze (foggy emptiness)",
            "Brushstroke texture (visible paint)",
            "Light fragmentation (broken color)",
            "Temporal blur (captured moment)",
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
            "edge_dissolution": 0.7,       # THE signature - boundaries uncertain
            "color_bleed": 0.6,            # Chromatic melding
            "atmospheric_haze": 0.6,       # Foggy emptiness
            "brushstroke_texture": 0.5,    # Visible paint
            "light_fragmentation": 0.5,    # Broken color
            "temporal_blur": 0.4,          # Motion suggestion
            "inference_steps": 30,
            "guidance_scale": 7.5,         # Slightly lower for softer results
            "img2img_strength": 0.65,      # More transformation for img2img
        }

    def generate(
        self,
        prompt: str | None = None,
        source_image: Image.Image | None = None,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Generate an image with Impressionist dissolution tells.

        Args:
            prompt: Text prompt for generation. Required unless source_image provided.
            source_image: Optional source image (if provided with prompt, uses img2img)
            control: Artifact control parameters
            **era_params: edge_dissolution, color_bleed, etc.
                         use_local: Force local generation (skip Replicate)

        Returns:
            Generated image with Impressionist dissolution tells
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
                raise ValueError("Impressionism pipeline requires a prompt or source image")

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

        edge_dissolution = get_effect_strength("edge_dissolution")
        color_bleed = get_effect_strength("color_bleed")
        atmospheric_haze = get_effect_strength("atmospheric_haze")
        brushstroke_texture = get_effect_strength("brushstroke_texture")
        light_fragmentation = get_effect_strength("light_fragmentation")
        temporal_blur = get_effect_strength("temporal_blur")

        # Apply Impressionist tells in sequence
        # Order matters: dissolution first, then atmospheric, then texture
        img = self._apply_edge_dissolution(img, edge_dissolution)
        img = self._apply_color_bleed(img, color_bleed)
        img = self._apply_atmospheric_haze(img, atmospheric_haze)
        img = self._apply_light_fragmentation(img, light_fragmentation)
        img = self._apply_brushstroke_texture(img, brushstroke_texture)
        img = self._apply_temporal_blur(img, temporal_blur)

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
        """Apply Impressionist dissolution tells to an existing image.

        This is the post-processing step only. Use this when you already have
        a base image (e.g., from batch Replicate generation) and want to apply
        the characteristic tells without re-generating.

        Args:
            img: Base image to apply tells to
            control: Artifact control parameters
            **era_params: edge_dissolution, color_bleed, etc.

        Returns:
            Image with Impressionist dissolution tells applied
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

        edge_dissolution = get_effect_strength("edge_dissolution")
        color_bleed = get_effect_strength("color_bleed")
        atmospheric_haze = get_effect_strength("atmospheric_haze")
        brushstroke_texture = get_effect_strength("brushstroke_texture")
        light_fragmentation = get_effect_strength("light_fragmentation")
        temporal_blur = get_effect_strength("temporal_blur")

        # Apply Impressionist tells in sequence
        img = self._apply_edge_dissolution(img, edge_dissolution)
        img = self._apply_color_bleed(img, color_bleed)
        img = self._apply_atmospheric_haze(img, atmospheric_haze)
        img = self._apply_light_fragmentation(img, light_fragmentation)
        img = self._apply_brushstroke_texture(img, brushstroke_texture)
        img = self._apply_temporal_blur(img, temporal_blur)

        # Final blend with original based on placement
        img = control.apply_mask(original, img)

        # Upscale if requested
        upscale = era_params.get("upscale", 1)
        if upscale > 1:
            new_size = (img.width * upscale, img.height * upscale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _apply_edge_dissolution(self, img: Image.Image, strength: float) -> Image.Image:
        """Make boundaries UNCERTAIN. Probabilistic edges.

        This is THE Impressionist signature: edges that aren't quite there.
        Where one form ends and another begins becomes a matter of probability,
        not definition. The eye must complete what the paint suggests.

        Technique:
        - Detect edges via Sobel
        - Create random displacement at edge regions
        - Shift pixels in probabilistic directions
        - Result: edges shimmer and dissolve
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Convert to grayscale for edge detection
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Detect edges via Sobel
        edges_x = np.abs(ndimage.sobel(gray, axis=1))
        edges_y = np.abs(ndimage.sobel(gray, axis=0))
        edges = np.sqrt(edges_x**2 + edges_y**2)

        if edges.max() > 0:
            edges = edges / edges.max()

        # Expand edge mask for broader dissolution zone
        edge_mask = ndimage.gaussian_filter(edges, sigma=4)
        edge_mask = np.clip(edge_mask * 2, 0, 1)

        # Create probabilistic displacement field
        np.random.seed(None)  # Use actual randomness for organic feel
        noise_y = np.random.randn(h, w) * 2
        noise_x = np.random.randn(h, w) * 2

        # Smooth the noise for coherent displacement
        noise_y = ndimage.gaussian_filter(noise_y, sigma=3)
        noise_x = ndimage.gaussian_filter(noise_x, sigma=3)

        # Scale displacement by edge mask and strength
        displacement_y = edge_mask * noise_y * strength * 10
        displacement_x = edge_mask * noise_x * strength * 10

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Apply displacement
        new_y = np.clip(y_coords + displacement_y, 0, h - 1)
        new_x = np.clip(x_coords + displacement_x, 0, w - 1)

        # Remap pixels
        result = np.zeros_like(arr)
        for c in range(3):
            result[:, :, c] = ndimage.map_coordinates(
                arr[:, :, c],
                [new_y, new_x],
                order=1,
                mode='reflect'
            )

        # Blend with original based on edge mask
        # This preserves solid areas while dissolving edges
        edge_blend = edge_mask[:, :, np.newaxis] * strength * 0.8
        result = arr * (1 - edge_blend) + result * edge_blend

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_color_bleed(self, img: Image.Image, strength: float) -> Image.Image:
        """Colors MELD across boundaries. Chromatic bleeding.

        "Colors melding together in its glooming, opalescent oneness."

        Each color channel shifts independently at edge regions,
        creating chromatic aberration and color bleeding that suggests
        forms dissolving into one another.

        Technique:
        - Detect edges
        - Shift each RGB channel in different directions at edges
        - Blend shifted channels with original
        - Result: opalescent color melding
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Detect edges for bleed zones
        edges_x = np.abs(ndimage.sobel(gray, axis=1))
        edges_y = np.abs(ndimage.sobel(gray, axis=0))
        edges = np.sqrt(edges_x**2 + edges_y**2)

        if edges.max() > 0:
            edges = edges / edges.max()

        # Expand edge mask
        edge_mask = ndimage.gaussian_filter(edges, sigma=6)
        edge_mask = np.clip(edge_mask * 1.5, 0, 1)

        # Shift each channel in different directions
        shift_amount = strength * 5

        # Red shifts one way
        r_shifted = ndimage.shift(r, [shift_amount, -shift_amount * 0.7], mode='reflect')
        # Green shifts another
        g_shifted = ndimage.shift(g, [-shift_amount * 0.5, shift_amount], mode='reflect')
        # Blue shifts yet another
        b_shifted = ndimage.shift(b, [shift_amount * 0.3, shift_amount * 0.8], mode='reflect')

        # Blend shifted channels at edge regions
        edge_blend = edge_mask * strength * 0.7

        r_result = r * (1 - edge_blend) + r_shifted * edge_blend
        g_result = g * (1 - edge_blend) + g_shifted * edge_blend
        b_result = b * (1 - edge_blend) + b_shifted * edge_blend

        result = np.stack([r_result, g_result, b_result], axis=2)

        # Add subtle overall color harmonization (opalescent quality)
        # Slightly blend toward a warm-cool gradient
        warm = np.array([255, 240, 220])  # Warm tones
        cool = np.array([200, 210, 230])  # Cool tones

        # Create gradient based on luminance
        lum_norm = gray / 255.0
        color_tint = (
            warm.reshape(1, 1, 3) * lum_norm[:, :, np.newaxis] +
            cool.reshape(1, 1, 3) * (1 - lum_norm[:, :, np.newaxis])
        )

        # Very subtle tint toward opalescent
        tint_strength = strength * 0.1
        result = result * (1 - tint_strength) + color_tint * tint_strength

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_atmospheric_haze(self, img: Image.Image, strength: float) -> Image.Image:
        """The FOGGY BLANKNESS. Expectant emptiness.

        Light scatters, forms recede into haze, the world becomes
        soft and indistinct. Distance is suggested through atmospheric
        perspective taken to its logical extreme.

        Technique:
        - Multi-layer blur for atmospheric depth
        - Add light blue-gray haze color
        - Increase haze with distance from center (depth simulation)
        - Soften all edges globally
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        center_y, center_x = h // 2, w // 2

        # Distance from center (simulates depth)
        dist = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        dist_norm = dist / max_dist

        # Multi-layer blur for atmospheric depth
        blur_light = ndimage.gaussian_filter(arr, sigma=[3, 3, 0])
        blur_medium = ndimage.gaussian_filter(arr, sigma=[8, 8, 0])
        blur_heavy = ndimage.gaussian_filter(arr, sigma=[18, 18, 0])

        # Blend blurs based on distance (periphery gets hazier)
        blur_mask = np.clip(dist_norm * 2 - 0.3, 0, 1) ** 0.8
        blur_mask = ndimage.gaussian_filter(blur_mask, sigma=15)
        blur_mask_3d = blur_mask[:, :, np.newaxis]

        result = (
            arr * (1 - blur_mask_3d * strength * 0.6) +
            blur_light * (blur_mask_3d * strength * 0.15) +
            blur_medium * (blur_mask_3d * strength * 0.25) +
            blur_heavy * (blur_mask_3d * strength * 0.2)
        )

        # Atmospheric haze color (light blue-gray, like morning mist)
        haze_color = np.array([195, 205, 220])  # Soft blue-gray
        haze = haze_color.reshape(1, 1, 3) * np.ones_like(arr)

        # Haze increases toward periphery and top of image
        y_norm = y_coords / h
        haze_mask = blur_mask * 0.5 + (1 - y_norm) * 0.3  # More haze at top
        haze_mask = np.clip(haze_mask, 0, 1)
        haze_mask = ndimage.gaussian_filter(haze_mask, sigma=20)
        haze_strength = haze_mask * strength * 0.4

        result = result * (1 - haze_strength[:, :, np.newaxis]) + haze * haze_strength[:, :, np.newaxis]

        # Global softening - everything becomes slightly dreamlike
        global_blur = ndimage.gaussian_filter(result, sigma=[2, 2, 0])
        global_blend = strength * 0.2
        result = result * (1 - global_blend) + global_blur * global_blend

        # Slightly reduce contrast (haze flattens tonal range)
        r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        mean_lum = np.mean(gray)

        contrast_reduction = 1 - strength * 0.15
        result = mean_lum + (result - mean_lum) * contrast_reduction

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_brushstroke_texture(self, img: Image.Image, strength: float) -> Image.Image:
        """Visible PAINT APPLICATION. The hand of the artist.

        Impressionism made brushstrokes visible, celebrated them.
        The paint IS the subject. Directional strokes that follow
        form and movement, creating texture that the eye must
        synthesize into coherent shapes.

        Technique:
        - Create directional noise following image gradients
        - Apply as luminance and slight color variation
        - Make strokes visible especially in mid-tones
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Calculate image gradients for stroke direction
        grad_y = ndimage.sobel(gray, axis=0)
        grad_x = ndimage.sobel(gray, axis=1)

        # Create base noise
        noise = np.random.randn(h, w)

        # Make noise directional based on local gradients
        # Blur noise in direction perpendicular to gradient
        angle = np.arctan2(grad_y, grad_x + 0.0001)

        # Create elongated stroke pattern
        # Use anisotropic blur following stroke direction
        noise_smooth = ndimage.gaussian_filter(noise, sigma=2)

        # Add directional component
        for _ in range(3):
            # Shift noise slightly along gradient direction
            shift_y = np.sin(angle) * 2
            shift_x = np.cos(angle) * 2
            shifted = ndimage.shift(noise_smooth, [0, 0], mode='reflect')
            noise_smooth = (noise_smooth + shifted) / 2

        noise_smooth = ndimage.gaussian_filter(noise_smooth, sigma=1.5)

        # Apply as luminance variation
        stroke_effect = noise_smooth * strength * 25

        result = arr.copy()
        result[:, :, 0] = result[:, :, 0] + stroke_effect * 1.0
        result[:, :, 1] = result[:, :, 1] + stroke_effect * 0.95
        result[:, :, 2] = result[:, :, 2] + stroke_effect * 0.9

        # Add color variation in strokes (subtle)
        color_noise_r = np.random.randn(h, w) * strength * 8
        color_noise_g = np.random.randn(h, w) * strength * 6
        color_noise_b = np.random.randn(h, w) * strength * 10

        # Smooth color noise for coherent patches
        color_noise_r = ndimage.gaussian_filter(color_noise_r, sigma=4)
        color_noise_g = ndimage.gaussian_filter(color_noise_g, sigma=4)
        color_noise_b = ndimage.gaussian_filter(color_noise_b, sigma=4)

        result[:, :, 0] = result[:, :, 0] + color_noise_r
        result[:, :, 1] = result[:, :, 1] + color_noise_g
        result[:, :, 2] = result[:, :, 2] + color_noise_b

        # Add texture especially to mid-tones (where strokes are most visible)
        lum_norm = gray / 255.0
        midtone_mask = 1 - np.abs(lum_norm - 0.5) * 2
        midtone_mask = np.clip(midtone_mask, 0, 1)
        midtone_mask = ndimage.gaussian_filter(midtone_mask, sigma=5)

        # Extra texture in midtones
        extra_texture = noise_smooth * midtone_mask * strength * 15
        result = result + extra_texture[:, :, np.newaxis]

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_light_fragmentation(self, img: Image.Image, strength: float) -> Image.Image:
        """BROKEN COLOR. Optical mixing. Stippled light.

        The Impressionists broke color into component parts,
        letting the eye do the mixing. Light becomes particles,
        colors fragment into dabs, the world shimmers with
        chromatic energy.

        Technique:
        - Add small-scale color variations
        - Apply primarily to bright areas (where light fragments)
        - Create stippled, broken color effect
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        lum_norm = gray / 255.0

        result = arr.copy()

        # Create fine-grained color noise (the "stippling")
        noise_r = np.random.randn(h, w) * strength * 18
        noise_g = np.random.randn(h, w) * strength * 15
        noise_b = np.random.randn(h, w) * strength * 20

        # Slight smoothing for coherence (but keep it fine)
        noise_r = ndimage.gaussian_filter(noise_r, sigma=1.2)
        noise_g = ndimage.gaussian_filter(noise_g, sigma=1.2)
        noise_b = ndimage.gaussian_filter(noise_b, sigma=1.2)

        # Apply more strongly to bright areas (light fragments more visibly)
        bright_mask = np.clip(lum_norm * 1.5 - 0.2, 0, 1)
        bright_mask = ndimage.gaussian_filter(bright_mask, sigma=3)

        # Also apply to areas with color (saturated areas fragment)
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = (max_rgb - min_rgb) / (max_rgb + 0.0001)
        sat_mask = np.clip(saturation * 2, 0, 1)

        combined_mask = (bright_mask * 0.6 + sat_mask * 0.4)

        result[:, :, 0] = result[:, :, 0] + noise_r * combined_mask
        result[:, :, 1] = result[:, :, 1] + noise_g * combined_mask
        result[:, :, 2] = result[:, :, 2] + noise_b * combined_mask

        # Add complementary color hints (optical mixing)
        # In bright areas, hint at warm colors; in shadows, cool
        warm_shift = bright_mask * strength * 10
        cool_shift = (1 - bright_mask) * strength * 8

        result[:, :, 0] = result[:, :, 0] + warm_shift  # Red in highlights
        result[:, :, 1] = result[:, :, 1] + warm_shift * 0.5  # Some yellow
        result[:, :, 2] = result[:, :, 2] + cool_shift  # Blue in shadows

        # Create subtle dappled effect (light through leaves)
        dapple = np.random.randn(h // 8 + 1, w // 8 + 1) * 0.5
        dapple = ndimage.zoom(dapple, (h / (h // 8 + 1), w / (w // 8 + 1)), order=1)
        dapple = dapple[:h, :w]  # Ensure correct size
        dapple = ndimage.gaussian_filter(dapple, sigma=8)

        dapple_strength = dapple * strength * 20
        result = result + dapple_strength[:, :, np.newaxis]

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_temporal_blur(self, img: Image.Image, strength: float) -> Image.Image:
        """The CAPTURED MOMENT. Time made visible.

        Impressionism sought to capture the fleeting instant,
        and in doing so, suggested motion and time. A slight
        directional blur implies the world is moving, that this
        is a moment extracted from flow.

        Technique:
        - Subtle directional motion blur
        - Apply more to peripheral areas
        - Suggest rather than show motion
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Create directional motion blur kernel (mostly horizontal)
        kernel_size = int(strength * 12) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Horizontal motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0 / kernel_size

        # Apply motion blur
        blurred = np.zeros_like(arr)
        for c in range(3):
            blurred[:, :, c] = ndimage.convolve(arr[:, :, c], kernel, mode='reflect')

        # Apply more to peripheral areas (center stays sharper)
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        dist_norm = dist / max_dist

        # Blur mask - more blur at edges
        blur_mask = np.clip(dist_norm * 1.5 - 0.2, 0, 1) ** 0.7
        blur_mask = ndimage.gaussian_filter(blur_mask, sigma=20)

        # Blend
        motion_strength = blur_mask * strength * 0.5
        result = arr * (1 - motion_strength[:, :, np.newaxis]) + blurred * motion_strength[:, :, np.newaxis]

        # Add very subtle global softness (the "impression" quality)
        soft = ndimage.gaussian_filter(result, sigma=[1.5, 1.5, 0])
        global_soft = strength * 0.15
        result = result * (1 - global_soft) + soft * global_soft

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
