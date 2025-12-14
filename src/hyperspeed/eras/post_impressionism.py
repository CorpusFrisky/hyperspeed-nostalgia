"""Post-Impressionism Era (1886-1905) / The Melting Zone parallel.

Forms already unstable, now amplified. Geometric patches that don't quite meet.
Swirling energy. Pointillist dissolution.

The Post-Impressionists pushed beyond optical fidelity into psychological
intensity. Where Impressionism dissolved through light and atmosphere,
Post-Impressionism dissolves through STRUCTURE. Forms break into geometric
patches (Cézanne), swirling energy fields (Van Gogh), or pointillist dots
(Seurat).

Key works:
- Cézanne's Mont Sainte-Victoire series: the mountain dissolving into geometric
  patches, "passage" between planes where boundaries blur
- Van Gogh's Starry Night: forms swirling into each other, the cypress flame
  bleeding into the sky
- Seurat's A Sunday Afternoon on the Island of La Grande Jatte: figures
  assembled from discrete dots, edges probabilistic by design

Deep Cut: Cézanne famously said he wanted to "treat nature by the cylinder,
the sphere, the cone." His late work pushed this toward dissolution: planes
that don't quite meet, edges that breathe. The melting zone is Cézanne's
"passage" taken to its logical conclusion by a model that can't decide where
one form ends and another begins.

The Tell: Unlike Impressionism (soft dissolution) or Neoclassicism (over-
idealization), Post-Impressionism is about STRUCTURAL dissolution. The model
struggles to hold geometric forms together. Forms compete for the same space.
Boundaries are contested rather than uncertain.

Technical approach:
1. Generate image with SDXL via Replicate API (fast) or local (fallback)
2. Apply characteristic Post-Impressionist effects locally:
   - Passage dissolution (Cézanne's planes that don't meet)
   - Swirl distortion (Van Gogh's energy fields)
   - Pointillist breakup (Seurat's discrete dots)
   - Geometric faceting (forms break into patches)
   - Color intensity (psychological over optical)
   - Edge ambiguity (forms bleeding into each other)

Environment:
- Set REPLICATE_API_TOKEN to use Replicate for fast generation
- Falls back to local SDXL if token not set or API fails
"""

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from hyperspeed.core.artifact_control import ArtifactControl
from hyperspeed.core.replicate_utils import generate_via_replicate
from hyperspeed.eras.base import EraMetadata, EraPipeline, EraRegistry


@EraRegistry.register
class PostImpressionismPipeline(EraPipeline):
    """Post-Impressionism Era: The Melting Zone.

    This pipeline generates images with the characteristic "structural
    dissolution" quality of Post-Impressionism, where forms break into
    geometric patches, swirling energy, or pointillist dots.

    Unlike Impressionism (soft dissolution through light), Post-Impressionism
    dissolves through STRUCTURE. Forms compete for space. Boundaries are
    contested. The model struggles to hold geometry together.

    Controls:
        passage_dissolution: Cézanne's planes that don't meet (0-1)
        swirl_distortion: Van Gogh's energy fields (0-1)
        pointillist_breakup: Seurat's discrete dots (0-1)
        geometric_faceting: Forms break into patches (0-1)
        color_intensity: Psychological saturation (0-1)
        edge_ambiguity: Forms bleeding into each other (0-1)
    """

    metadata: ClassVar[EraMetadata] = EraMetadata(
        name="Post Impressionism",
        art_historical_parallel="Post-Impressionism (1886-1905)",
        time_period="The Melting Zone",
        description=(
            "Forms already unstable, now amplified. Geometric patches that don't quite meet. "
            "Swirling energy. Pointillist dissolution. The model can't decide where one form "
            "ends and another begins."
        ),
        characteristic_artifacts=[
            "Passage dissolution (Cézanne's planes that don't meet)",
            "Swirling distortion (Van Gogh's energy fields)",
            "Pointillist breakup (Seurat's discrete dots)",
            "Geometric faceting (forms break into patches)",
            "Color intensity (psychological over optical)",
            "Edge ambiguity (forms bleeding into each other)",
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
            "passage_dissolution": 0.7,    # THE signature - Cézanne's planes
            "swirl_distortion": 0.5,       # Van Gogh's energy
            "pointillist_breakup": 0.4,    # Seurat's dots
            "geometric_faceting": 0.6,     # Forms into patches
            "color_intensity": 0.6,        # Psychological saturation
            "edge_ambiguity": 0.5,         # Forms bleeding together
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
        """Generate an image with Post-Impressionist structural dissolution tells.

        Args:
            prompt: Text prompt for generation. Required unless source_image provided.
            source_image: Optional source image (if provided with prompt, uses img2img)
            control: Artifact control parameters
            **era_params: passage_dissolution, swirl_distortion, etc.
                         use_local: Force local generation (skip Replicate)

        Returns:
            Generated image with Post-Impressionist structural dissolution tells
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
                raise ValueError("Post-Impressionism pipeline requires a prompt or source image")

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

        passage_dissolution = get_effect_strength("passage_dissolution")
        swirl_distortion = get_effect_strength("swirl_distortion")
        pointillist_breakup = get_effect_strength("pointillist_breakup")
        geometric_faceting = get_effect_strength("geometric_faceting")
        color_intensity = get_effect_strength("color_intensity")
        edge_ambiguity = get_effect_strength("edge_ambiguity")

        # Apply Post-Impressionist tells in sequence
        # Order matters: structure first, then energy, then color
        img = self._apply_geometric_faceting(img, geometric_faceting)
        img = self._apply_passage_dissolution(img, passage_dissolution)
        img = self._apply_swirl_distortion(img, swirl_distortion)
        img = self._apply_pointillist_breakup(img, pointillist_breakup)
        img = self._apply_edge_ambiguity(img, edge_ambiguity)
        img = self._apply_color_intensity(img, color_intensity)

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
        """Apply Post-Impressionist structural dissolution tells to an existing image.

        This is the post-processing step only. Use this when you already have
        a base image (e.g., from batch Replicate generation) and want to apply
        the characteristic tells without re-generating.

        Args:
            img: Base image to apply tells to
            control: Artifact control parameters
            **era_params: passage_dissolution, swirl_distortion, etc.

        Returns:
            Image with Post-Impressionist structural dissolution tells applied
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

        passage_dissolution = get_effect_strength("passage_dissolution")
        swirl_distortion = get_effect_strength("swirl_distortion")
        pointillist_breakup = get_effect_strength("pointillist_breakup")
        geometric_faceting = get_effect_strength("geometric_faceting")
        color_intensity = get_effect_strength("color_intensity")
        edge_ambiguity = get_effect_strength("edge_ambiguity")

        # Apply Post-Impressionist tells in sequence
        img = self._apply_geometric_faceting(img, geometric_faceting)
        img = self._apply_passage_dissolution(img, passage_dissolution)
        img = self._apply_swirl_distortion(img, swirl_distortion)
        img = self._apply_pointillist_breakup(img, pointillist_breakup)
        img = self._apply_edge_ambiguity(img, edge_ambiguity)
        img = self._apply_color_intensity(img, color_intensity)

        # Final blend with original based on placement
        img = control.apply_mask(original, img)

        # Upscale if requested
        upscale = era_params.get("upscale", 1)
        if upscale > 1:
            new_size = (img.width * upscale, img.height * upscale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _apply_passage_dissolution(self, img: Image.Image, strength: float) -> Image.Image:
        """Cézanne's PASSAGE: planes that don't quite meet.

        This is THE Post-Impressionist signature: boundaries that breathe,
        forms that almost but not quite connect. Where one plane ends and
        another begins becomes uncertain, contested territory.

        Technique:
        - Detect edges and form boundaries
        - Create gaps/overlaps at boundary regions
        - Apply displacement that pulls forms apart slightly
        - Result: planes fail to connect cleanly
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Convert to grayscale for edge detection
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Detect edges (form boundaries)
        edges_x = ndimage.sobel(gray, axis=1)
        edges_y = ndimage.sobel(gray, axis=0)
        edge_mag = np.sqrt(edges_x**2 + edges_y**2)

        if edge_mag.max() > 0:
            edge_mag = edge_mag / edge_mag.max()

        # Create boundary mask - where forms meet
        boundary_mask = ndimage.gaussian_filter(edge_mag, sigma=3)
        boundary_mask = np.clip(boundary_mask * 2.5, 0, 1)

        # Calculate edge direction (perpendicular to gradient)
        angle = np.arctan2(edges_y, edges_x + 0.0001)

        # Create displacement that pulls planes APART at boundaries
        # This creates the "passage" effect - gaps where forms should meet
        displacement_scale = strength * 8

        # Displacement perpendicular to edge direction (pulls apart)
        displacement_y = np.sin(angle + np.pi/2) * boundary_mask * displacement_scale
        displacement_x = np.cos(angle + np.pi/2) * boundary_mask * displacement_scale

        # Add some randomness to make it organic
        noise_y = np.random.randn(h, w) * 0.3
        noise_x = np.random.randn(h, w) * 0.3
        noise_y = ndimage.gaussian_filter(noise_y, sigma=4)
        noise_x = ndimage.gaussian_filter(noise_x, sigma=4)

        displacement_y += noise_y * boundary_mask * strength * 3
        displacement_x += noise_x * boundary_mask * strength * 3

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

        # Blend based on boundary mask
        blend_mask = boundary_mask[:, :, np.newaxis] * strength * 0.85
        result = arr * (1 - blend_mask) + result * blend_mask

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_swirl_distortion(self, img: Image.Image, strength: float) -> Image.Image:
        """Van Gogh's ENERGY FIELDS. Swirling, kinetic distortion.

        The sky swirls. The cypress flame writhes. Everything has kinetic
        energy, psychological intensity. Forms don't just sit there; they
        move, breathe, pulse with inner life.

        Technique:
        - Create flow field with spiral/circular patterns
        - Apply pixel displacement following the flow
        - Concentrate effect around high-contrast areas
        - Result: forms have swirling, energetic quality
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Normalize coordinates to [-1, 1]
        y_norm = (y_coords - h/2) / (h/2)
        x_norm = (x_coords - w/2) / (w/2)

        # Create swirling flow field
        # Multiple swirl centers for complexity
        swirl_field_x = np.zeros((h, w))
        swirl_field_y = np.zeros((h, w))

        # Main central swirl
        r = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan2(y_norm, x_norm)

        # Swirl strength decreases with distance from center
        swirl_strength = np.exp(-r * 1.5) * strength * 0.8
        swirl_angle = r * 3 * strength  # More swirl toward edges

        swirl_field_x += -np.sin(theta + swirl_angle) * swirl_strength * 15
        swirl_field_y += np.cos(theta + swirl_angle) * swirl_strength * 15

        # Add secondary swirl centers (like stars in Starry Night)
        np.random.seed(42)  # Consistent placement
        num_vortices = 5
        for _ in range(num_vortices):
            cx = np.random.uniform(-0.7, 0.7)
            cy = np.random.uniform(-0.7, 0.7)

            dx = x_norm - cx
            dy = y_norm - cy
            dr = np.sqrt(dx**2 + dy**2)
            dt = np.arctan2(dy, dx)

            vortex_strength = np.exp(-dr * 4) * strength * 0.5
            vortex_angle = dr * 5

            swirl_field_x += -np.sin(dt + vortex_angle) * vortex_strength * 10
            swirl_field_y += np.cos(dt + vortex_angle) * vortex_strength * 10

        # Detect high-contrast areas (concentrate swirl there)
        r_ch, g_ch, b_ch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch

        edges = np.sqrt(ndimage.sobel(gray, axis=0)**2 + ndimage.sobel(gray, axis=1)**2)
        if edges.max() > 0:
            edges = edges / edges.max()
        edge_boost = ndimage.gaussian_filter(edges, sigma=8)
        edge_boost = 0.5 + edge_boost * 0.5  # Base + edge contribution

        # Apply edge boost to swirl field
        swirl_field_x *= edge_boost
        swirl_field_y *= edge_boost

        # Smooth the flow field for coherence
        swirl_field_x = ndimage.gaussian_filter(swirl_field_x, sigma=5)
        swirl_field_y = ndimage.gaussian_filter(swirl_field_y, sigma=5)

        # Apply displacement
        new_y = np.clip(y_coords + swirl_field_y, 0, h - 1)
        new_x = np.clip(x_coords + swirl_field_x, 0, w - 1)

        # Remap pixels
        result = np.zeros_like(arr)
        for c in range(3):
            result[:, :, c] = ndimage.map_coordinates(
                arr[:, :, c],
                [new_y, new_x],
                order=1,
                mode='reflect'
            )

        # Blend with original
        blend = strength * 0.9
        result = arr * (1 - blend) + result * blend

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_pointillist_breakup(self, img: Image.Image, strength: float) -> Image.Image:
        """Seurat's DOTS. Break continuous tones into discrete color particles.

        Seurat built images from tiny dots of pure color, letting the eye
        do the mixing. The image becomes a field of discrete particles,
        edges probabilistic by design.

        Technique:
        - Create dot/particle pattern
        - Sample colors at particle centers
        - Render as discrete color points
        - Blend with original for controllable effect
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Determine dot spacing based on strength
        # Higher strength = larger, more visible dots
        dot_spacing = int(6 - strength * 3)  # 6 to 3 pixels
        dot_spacing = max(2, dot_spacing)

        # Create dot centers
        y_centers = np.arange(dot_spacing // 2, h, dot_spacing)
        x_centers = np.arange(dot_spacing // 2, w, dot_spacing)

        # Create output array
        result = arr.copy()

        # Create a mask for where dots will be
        dot_mask = np.zeros((h, w), dtype=np.float32)

        # For each dot center, sample color and create circular dot
        dot_radius = dot_spacing * 0.45  # Slight overlap

        for yi in y_centers:
            for xi in x_centers:
                # Add slight randomness to dot positions (organic feel)
                y_offset = np.random.uniform(-dot_spacing * 0.15, dot_spacing * 0.15)
                x_offset = np.random.uniform(-dot_spacing * 0.15, dot_spacing * 0.15)

                cy = int(np.clip(yi + y_offset, 0, h - 1))
                cx = int(np.clip(xi + x_offset, 0, w - 1))

                # Sample color at center (with small neighborhood for stability)
                y_start = max(0, cy - 1)
                y_end = min(h, cy + 2)
                x_start = max(0, cx - 1)
                x_end = min(w, cx + 2)

                color = np.mean(arr[y_start:y_end, x_start:x_end], axis=(0, 1))

                # Add slight color variation (pure pigment feel)
                color_var = np.random.uniform(-10, 10, 3) * strength
                color = np.clip(color + color_var, 0, 255)

                # Create circular dot region
                y_range = np.arange(max(0, int(cy - dot_radius)), min(h, int(cy + dot_radius + 1)))
                x_range = np.arange(max(0, int(cx - dot_radius)), min(w, int(cx + dot_radius + 1)))

                if len(y_range) == 0 or len(x_range) == 0:
                    continue

                yy, xx = np.meshgrid(y_range, x_range, indexing='ij')
                dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)

                # Soft-edged dot
                dot_alpha = np.clip(1 - dist / dot_radius, 0, 1)
                dot_alpha = dot_alpha ** 0.7  # Slightly harder edge

                # Apply dot color
                for c in range(3):
                    result[yy, xx, c] = (
                        result[yy, xx, c] * (1 - dot_alpha * strength) +
                        color[c] * dot_alpha * strength
                    )

                dot_mask[yy, xx] = np.maximum(dot_mask[yy, xx], dot_alpha)

        # Add subtle color separation (chromatic dots)
        # Shift different channels slightly for pointillist color mixing feel
        if strength > 0.3:
            shift_amount = strength * 1.5
            r_shifted = ndimage.shift(result[:, :, 0], [shift_amount, 0], mode='reflect')
            b_shifted = ndimage.shift(result[:, :, 2], [-shift_amount, 0], mode='reflect')

            chromatic_blend = (strength - 0.3) * 0.3
            result[:, :, 0] = result[:, :, 0] * (1 - chromatic_blend) + r_shifted * chromatic_blend
            result[:, :, 2] = result[:, :, 2] * (1 - chromatic_blend) + b_shifted * chromatic_blend

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_geometric_faceting(self, img: Image.Image, strength: float) -> Image.Image:
        """Cézanne's GEOMETRY. Forms break into patches.

        "Treat nature by the cylinder, the sphere, the cone." Forms simplify
        into geometric patches, each with relatively uniform color. The world
        becomes a construction of colored planes.

        Technique:
        - Segment image into irregular patches
        - Flatten each patch toward average color
        - Preserve some internal texture
        - Result: painterly geometric simplification
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Determine patch size based on strength
        patch_size = int(20 - strength * 12)  # 20 to 8 pixels
        patch_size = max(4, patch_size)

        result = arr.copy()

        # Create patches and flatten each
        for yi in range(0, h, patch_size):
            for xi in range(0, w, patch_size):
                # Define patch boundaries
                y_end = min(yi + patch_size, h)
                x_end = min(xi + patch_size, w)

                # Extract patch
                patch = arr[yi:y_end, xi:x_end]

                # Calculate average color for this patch
                avg_color = np.mean(patch, axis=(0, 1))

                # Calculate how much to flatten toward average
                flatten_amount = strength * 0.7

                # Flatten patch (blend toward average while keeping some texture)
                flattened = patch * (1 - flatten_amount) + avg_color * flatten_amount

                # Add slight variation within patch (painterly feel)
                noise = np.random.randn(y_end - yi, x_end - xi, 3) * 5 * strength
                noise = ndimage.gaussian_filter(noise, sigma=[1, 1, 0])
                flattened = flattened + noise

                result[yi:y_end, xi:x_end] = flattened

        # Detect patch boundaries and enhance them slightly
        # (Cézanne's patches have visible brush boundaries)
        r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        edges = np.sqrt(ndimage.sobel(gray, axis=0)**2 + ndimage.sobel(gray, axis=1)**2)
        if edges.max() > 0:
            edges = edges / edges.max()

        # Darken at patch edges slightly (visible brushwork)
        edge_darken = edges * strength * 20
        result = result - edge_darken[:, :, np.newaxis] * 0.3

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_color_intensity(self, img: Image.Image, strength: float) -> Image.Image:
        """PSYCHOLOGICAL COLOR. Expressive, not optical.

        Post-Impressionists used color for emotional effect, not naturalistic
        representation. Van Gogh's yellows burn. Gauguin's colors symbolize.
        Color becomes a carrier of psychological intensity.

        Technique:
        - Increase saturation significantly
        - Push colors toward their emotional extremes
        - Add warm/cool contrast
        - Result: emotionally intense, non-naturalistic color
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Calculate luminance and saturation
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        lum = (max_rgb + min_rgb) / 2

        # Increase saturation
        sat_boost = 1 + strength * 0.8

        # For each pixel, increase distance from gray
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        r_boosted = gray + (r - gray) * sat_boost
        g_boosted = gray + (g - gray) * sat_boost
        b_boosted = gray + (b - gray) * sat_boost

        # Push warm colors warmer, cool colors cooler
        # Detect warm vs cool based on red-blue balance
        warmth = (r - b) / 255.0  # Positive = warm, negative = cool

        warm_push = warmth * strength * 30
        r_boosted = r_boosted + warm_push
        b_boosted = b_boosted - warm_push * 0.5

        # Add slight yellow to highlights (Van Gogh's burning yellows)
        highlight_mask = np.clip((lum - 128) / 127, 0, 1)
        r_boosted = r_boosted + highlight_mask * strength * 20
        g_boosted = g_boosted + highlight_mask * strength * 15

        # Add blue-violet to shadows (psychological depth)
        shadow_mask = np.clip((128 - lum) / 128, 0, 1)
        b_boosted = b_boosted + shadow_mask * strength * 25
        r_boosted = r_boosted + shadow_mask * strength * 10  # Slight violet

        result = np.stack([r_boosted, g_boosted, b_boosted], axis=2)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_edge_ambiguity(self, img: Image.Image, strength: float) -> Image.Image:
        """CONTESTED EDGES. Forms bleeding into each other.

        Where Impressionism has soft, uncertain edges, Post-Impressionism
        has CONTESTED edges. Forms don't fade into each other; they compete
        for the same space. Boundaries are battlegrounds.

        Technique:
        - Detect form boundaries
        - Create bilateral bleeding (both sides encroach)
        - Forms overlap and compete
        - Result: edges feel contested, unstable
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Detect edges
        edges_x = ndimage.sobel(gray, axis=1)
        edges_y = ndimage.sobel(gray, axis=0)
        edge_mag = np.sqrt(edges_x**2 + edges_y**2)

        if edge_mag.max() > 0:
            edge_mag = edge_mag / edge_mag.max()

        # Create edge mask
        edge_mask = ndimage.gaussian_filter(edge_mag, sigma=4)
        edge_mask = np.clip(edge_mask * 2, 0, 1)

        # Create two shifted versions (simulating forms bleeding into each other)
        shift_amount = strength * 6

        # Shift in opposite directions based on edge gradient
        angle = np.arctan2(edges_y, edges_x + 0.0001)

        # Create shifted images
        shifted_1 = np.zeros_like(arr)
        shifted_2 = np.zeros_like(arr)

        for c in range(3):
            # Shift along gradient direction
            shifted_1[:, :, c] = ndimage.shift(
                arr[:, :, c],
                [shift_amount, shift_amount * 0.7],
                mode='reflect'
            )
            # Shift opposite direction
            shifted_2[:, :, c] = ndimage.shift(
                arr[:, :, c],
                [-shift_amount * 0.7, -shift_amount],
                mode='reflect'
            )

        # Blend shifted versions at edges (forms competing)
        edge_3d = edge_mask[:, :, np.newaxis]
        blend_strength = edge_3d * strength * 0.6

        # Average of both shifts creates contested boundary
        contested = (shifted_1 + shifted_2) / 2

        result = arr * (1 - blend_strength) + contested * blend_strength

        # Add color bleeding at edges (forms' colors invade each other)
        # Blur and blend colors at edge regions
        color_blur = ndimage.gaussian_filter(arr, sigma=[6, 6, 0])
        color_blend = edge_3d * strength * 0.3
        result = result * (1 - color_blend) + color_blur * color_blend

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
