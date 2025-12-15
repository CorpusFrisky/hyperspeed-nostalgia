"""Cubism Era (1907-1920s) / Architectural Impossibility (2022-present) parallel.

Spatial contradiction. Multiple perspectives simultaneously. Picasso and Braque
"rejected the inherited concept that art should copy nature" and depicted objects
"from multiple perspectives to represent the subject in a greater context."

AI architectural renders do this by accident: spaces that couldn't exist,
perspectives that contradict themselves, staircases that lead nowhere,
windows that open onto impossible spaces. M.C. Escher by way of latent
space interpolation.

Key works:
- Braque's Houses at L'Estaque (1908): "reducing everything to geometric
  schemas, to cubes"
- Picasso's Les Demoiselles d'Avignon (1907): "fractured, angular shapes"
  showing faces from multiple angles simultaneously
- Braque's Pitcher and Violin (1909-1910): the scroll of the violin shown
  from the side while the body is frontal

Deep Cut: The Cubists were "breaking objects up into smaller and smaller
facets, until they are virtually unrecognizable in a shallow plane." AI
does this to architecture. Both achieve spatial impossibility - Cubism
through intentional artistic vision, AI through latent space interpolation
errors. The result is visually parallel: objects exist in contradictory
spatial relationships.

The Tell: Unlike Fauvism (color violence) or Symbolism (excessive mystique),
Cubism is about SPATIAL CONTRADICTION. Multiple perspectives coexist
impossibly. Geometry fractures. Planes refuse to align. Architecture
becomes Escherian.

Technical approach:
1. Generate image with SDXL via Replicate API (fast) or local (fallback)
2. Apply characteristic Cubist/Architectural Impossibility effects locally:
   - Perspective fracture (multiple viewpoints simultaneously)
   - Geometric faceting (forms break into angular planes)
   - Spatial contradiction (depth cues that conflict)
   - Plane slippage (surfaces that don't align)
   - Edge multiplication (forms shown from multiple angles)
   - Shallow depth (compressed Z-axis, everything flattens)

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
class CubismPipeline(EraPipeline):
    """Cubism Era: Architectural Impossibility (2022-present).

    This pipeline generates images with the characteristic spatial contradiction
    of Cubism and AI's architectural impossibilities. Multiple perspectives
    coexist impossibly. Geometry fractures. Planes refuse to align.

    Unlike Fauvism (color violence) or Symbolism (excessive mystique),
    Cubism fails through SPATIAL CONTRADICTION. Staircases lead nowhere,
    windows open onto impossible spaces, objects exist from multiple
    viewpoints simultaneously.

    Controls:
        perspective_fracture: Multiple viewpoints simultaneously (0-1)
        geometric_faceting: Forms break into angular planes (0-1)
        spatial_contradiction: Depth cues that conflict (0-1)
        plane_slippage: Surfaces that don't align (0-1)
        edge_multiplication: Forms shown from multiple angles (0-1)
        shallow_depth: Compressed Z-axis, flattening (0-1)
    """

    metadata: ClassVar[EraMetadata] = EraMetadata(
        name="Cubism",
        art_historical_parallel="Cubism (1907-1920s)",
        time_period="Architectural Impossibility (2022-present)",
        description=(
            "Spatial contradiction as aesthetic. Picasso's multiple perspectives "
            "meet AI's impossible architectures. Staircases to nowhere, windows onto "
            "void, M.C. Escher by way of latent space interpolation. "
            "Both break objects into facets until space itself fractures."
        ),
        characteristic_artifacts=[
            "Perspective fracture (multiple viewpoints simultaneously)",
            "Geometric faceting (forms break into angular planes)",
            "Spatial contradiction (conflicting depth cues)",
            "Plane slippage (surfaces that don't align)",
            "Edge multiplication (forms from multiple angles)",
            "Shallow depth (compressed Z-axis, flattening)",
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
            "perspective_fracture": 0.7,     # Multiple viewpoints - THE Cubist signature
            "geometric_faceting": 0.8,       # Angular plane breakdown
            "spatial_contradiction": 0.6,    # Conflicting depth cues
            "plane_slippage": 0.5,           # Misaligned surfaces
            "edge_multiplication": 0.6,      # Forms from multiple angles
            "shallow_depth": 0.7,            # Z-axis compression
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
        """Generate an image with Cubist spatial contradiction tells.

        Args:
            prompt: Text prompt for generation. Required unless source_image provided.
            source_image: Optional source image (if provided with prompt, uses img2img)
            control: Artifact control parameters
            **era_params: perspective_fracture, geometric_faceting, etc.
                         use_local: Force local generation (skip Replicate)

        Returns:
            Generated image with Cubist spatial contradiction tells
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
                raise ValueError("Cubism pipeline requires a prompt or source image")

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

        perspective_fracture = get_effect_strength("perspective_fracture")
        geometric_faceting = get_effect_strength("geometric_faceting")
        spatial_contradiction = get_effect_strength("spatial_contradiction")
        plane_slippage = get_effect_strength("plane_slippage")
        edge_multiplication = get_effect_strength("edge_multiplication")
        shallow_depth = get_effect_strength("shallow_depth")

        # Apply Cubist tells in sequence
        # Order: structure first (faceting, depth), then contradiction, then edges
        img = self._apply_geometric_faceting(img, geometric_faceting)
        img = self._apply_shallow_depth(img, shallow_depth)
        img = self._apply_perspective_fracture(img, perspective_fracture)
        img = self._apply_spatial_contradiction(img, spatial_contradiction)
        img = self._apply_plane_slippage(img, plane_slippage)
        img = self._apply_edge_multiplication(img, edge_multiplication)

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
        """Apply Cubist spatial contradiction tells to an existing image.

        This is the post-processing step only. Use this when you already have
        a base image (e.g., from batch Replicate generation) and want to apply
        the characteristic tells without re-generating.

        Args:
            img: Base image to apply tells to
            control: Artifact control parameters
            **era_params: perspective_fracture, geometric_faceting, etc.

        Returns:
            Image with Cubist spatial contradiction tells applied
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

        perspective_fracture = get_effect_strength("perspective_fracture")
        geometric_faceting = get_effect_strength("geometric_faceting")
        spatial_contradiction = get_effect_strength("spatial_contradiction")
        plane_slippage = get_effect_strength("plane_slippage")
        edge_multiplication = get_effect_strength("edge_multiplication")
        shallow_depth = get_effect_strength("shallow_depth")

        # Apply Cubist tells in sequence
        img = self._apply_geometric_faceting(img, geometric_faceting)
        img = self._apply_shallow_depth(img, shallow_depth)
        img = self._apply_perspective_fracture(img, perspective_fracture)
        img = self._apply_spatial_contradiction(img, spatial_contradiction)
        img = self._apply_plane_slippage(img, plane_slippage)
        img = self._apply_edge_multiplication(img, edge_multiplication)

        # Final blend with original based on placement
        img = control.apply_mask(original, img)

        # Upscale if requested
        upscale = era_params.get("upscale", 1)
        if upscale > 1:
            new_size = (img.width * upscale, img.height * upscale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _apply_geometric_faceting(self, img: Image.Image, strength: float) -> Image.Image:
        """ANGULAR PLANES. Forms break into geometric facets.

        Braque reduced "everything to geometric schemas, to cubes." The
        continuous surface of reality is broken into angular facets,
        each catching light differently. AI architectural renders
        similarly fragment surfaces into impossible angular relationships.

        Technique:
        - Detect regions and break them into angular segments
        - Apply slight color/value variation to each facet
        - Create hard edges where soft gradients existed
        - Result: surfaces feel crystalline, faceted
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Create angular segmentation pattern
        # Use multiple overlapping angular gradients
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Multiple angular planes at different orientations
        np.random.seed(45)
        num_planes = int(8 + strength * 12)

        facet_map = np.zeros((h, w))
        for i in range(num_planes):
            angle = np.random.random() * np.pi
            offset = np.random.random() * 200 - 100
            scale = 30 + np.random.random() * 50

            # Create angled stripe pattern
            projection = x_coords * np.cos(angle) + y_coords * np.sin(angle)
            stripe = np.floor((projection + offset) / scale)
            facet_map = facet_map + stripe * (i + 1) * 0.1

        # Normalize facet map to create distinct regions
        facet_map = facet_map % 1.0

        # Quantize into distinct facets
        num_facets = int(6 + strength * 10)
        facet_indices = np.floor(facet_map * num_facets).astype(int)

        # Apply value shift to each facet
        result = arr.copy()
        for i in range(num_facets):
            mask = (facet_indices == i)
            if np.sum(mask) > 0:
                # Random value shift for this facet
                shift = (np.random.random() - 0.5) * 30 * strength
                result[mask] = result[mask] + shift

        # Enhance edges between facets
        facet_edges = np.sqrt(
            ndimage.sobel(facet_indices.astype(float), axis=0)**2 +
            ndimage.sobel(facet_indices.astype(float), axis=1)**2
        )
        facet_edges = np.clip(facet_edges, 0, 1)

        # Darken edges slightly (the Cubist outline)
        edge_darkening = facet_edges[:, :, np.newaxis] * 20 * strength
        result = result - edge_darkening

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_shallow_depth(self, img: Image.Image, strength: float) -> Image.Image:
        """COMPRESSED Z-AXIS. Everything flattens toward the picture plane.

        Cubism "reduced depth to a shallow plane." Objects that should
        recede into space instead pile up at the surface. Background
        and foreground compress together. AI architectural renders
        similarly lose depth coherence, creating spaces that feel flat
        despite depicting three dimensions.

        Technique:
        - Reduce tonal contrast (compress value range)
        - Flatten atmospheric perspective
        - Push background values toward midtones
        - Result: depth flattens, space compresses
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Compress value range toward middle (flattening)
        target_mid = 128
        compression = 1 - strength * 0.4

        # Compress each channel toward midpoint
        r_flat = target_mid + (r - target_mid) * compression
        g_flat = target_mid + (g - target_mid) * compression
        b_flat = target_mid + (b - target_mid) * compression

        # Reduce atmospheric perspective (blues in distance)
        # Dark areas (assumed distance) lose their blue cast
        dark_mask = np.clip((100 - gray) / 100, 0, 1)
        b_reduction = dark_mask * 15 * strength
        b_flat = b_flat - b_reduction

        # Flatten local contrast (everything becomes more uniform)
        for channel in [r_flat, g_flat, b_flat]:
            local_mean = ndimage.uniform_filter(channel, size=50)
            local_contrast = channel - local_mean
            channel[:] = local_mean + local_contrast * (1 - strength * 0.3)

        # Add slight overall desaturation (Analytic Cubism palette)
        gray_flat = 0.299 * r_flat + 0.587 * g_flat + 0.114 * b_flat
        desat_amount = strength * 0.25

        r_flat = r_flat * (1 - desat_amount) + gray_flat * desat_amount
        g_flat = g_flat * (1 - desat_amount) + gray_flat * desat_amount
        b_flat = b_flat * (1 - desat_amount) + gray_flat * desat_amount

        result = np.stack([r_flat, g_flat, b_flat], axis=2)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_perspective_fracture(self, img: Image.Image, strength: float) -> Image.Image:
        """MULTIPLE VIEWPOINTS. Objects seen from several angles at once.

        Picasso's faces show profile and frontal view simultaneously.
        Braque's violin shows scroll from side, body from front. AI
        architectural renders achieve similar impossibility: rooms
        seen from multiple perspectives that cannot coexist.

        Technique:
        - Divide image into regions
        - Apply different perspective transforms to each
        - Create seams where perspectives meet
        - Result: space viewed from impossible multiple angles
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Create perspective regions
        np.random.seed(46)
        num_regions = int(3 + strength * 4)

        # Voronoi-like region centers
        centers = []
        for _ in range(num_regions):
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            centers.append((cx, cy))

        # Assign each pixel to nearest center
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        region_map = np.zeros((h, w), dtype=int)

        for i, (cx, cy) in enumerate(centers):
            dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            if i == 0:
                min_dist = dist.copy()
                region_map[:] = 0
            else:
                closer = dist < min_dist
                region_map[closer] = i
                min_dist = np.minimum(min_dist, dist)

        # Apply different transforms to each region
        result = arr.copy()

        for i in range(num_regions):
            mask = region_map == i
            if np.sum(mask) < 100:
                continue

            # Random perspective shift for this region
            shift_x = int((np.random.random() - 0.5) * 20 * strength)
            shift_y = int((np.random.random() - 0.5) * 20 * strength)
            scale = 1 + (np.random.random() - 0.5) * 0.1 * strength

            # Get region bounds
            region_y, region_x = np.where(mask)
            if len(region_y) == 0:
                continue

            y_min, y_max = region_y.min(), region_y.max()
            x_min, x_max = region_x.min(), region_x.max()

            # Apply transform via coordinate remapping
            for y in range(y_min, min(y_max + 1, h)):
                for x in range(x_min, min(x_max + 1, w)):
                    if mask[y, x]:
                        # Source coordinates with transform
                        src_x = int((x - (x_min + x_max) / 2) / scale + (x_min + x_max) / 2 - shift_x)
                        src_y = int((y - (y_min + y_max) / 2) / scale + (y_min + y_max) / 2 - shift_y)

                        src_x = np.clip(src_x, 0, w - 1)
                        src_y = np.clip(src_y, 0, h - 1)

                        result[y, x] = arr[src_y, src_x]

        # Enhance seams between regions
        region_edges = np.sqrt(
            ndimage.sobel(region_map.astype(float), axis=0)**2 +
            ndimage.sobel(region_map.astype(float), axis=1)**2
        )
        region_edges = np.clip(region_edges * 2, 0, 1)

        # Add dark lines at region boundaries
        seam_intensity = region_edges[:, :, np.newaxis] * 40 * strength
        result = result - seam_intensity

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_spatial_contradiction(self, img: Image.Image, strength: float) -> Image.Image:
        """CONFLICTING DEPTH CUES. Space that cannot exist.

        AI architectural renders produce stairs that lead nowhere,
        windows opening onto impossible voids, rooms whose geometry
        contradicts itself. This is Cubism's spatial impossibility
        made literal: depth cues that actively conflict.

        Technique:
        - Invert depth relationships in selected areas
        - Create lighting contradictions
        - Add impossible shadows
        - Result: space that defies logic
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Create regions where we'll invert depth relationships
        np.random.seed(47)

        # Identify "depth" through luminance gradient
        # Areas that should be far (dark) vs near (bright)
        gray_blur = ndimage.gaussian_filter(gray, sigma=30)

        # Create contradiction zones
        num_zones = int(2 + strength * 3)
        contradiction_mask = np.zeros((h, w))

        for _ in range(num_zones):
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            radius = 50 + np.random.random() * 100

            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            zone = np.clip(1 - dist / radius, 0, 1)
            contradiction_mask = np.maximum(contradiction_mask, zone)

        contradiction_mask = contradiction_mask * strength

        # In contradiction zones, invert the depth-luminance relationship
        # Dark becomes light, light becomes dark (local inversion)
        local_mean = ndimage.gaussian_filter(gray, sigma=20)

        inverted_gray = 2 * local_mean - gray  # Invert around local mean

        # Apply inversion to contradiction zones
        for c in range(3):
            channel = arr[:, :, c]
            channel_mean = ndimage.gaussian_filter(channel, sigma=20)
            channel_inverted = 2 * channel_mean - channel
            arr[:, :, c] = channel * (1 - contradiction_mask) + channel_inverted * contradiction_mask

        # Add contradictory shadows (shadows in "wrong" places)
        shadow_mask = np.random.random((h, w)) < (0.02 * strength)
        shadow_mask = ndimage.gaussian_filter(shadow_mask.astype(float), sigma=15)
        shadow_mask = np.clip(shadow_mask * 3, 0, 0.3) * strength

        arr = arr * (1 - shadow_mask[:, :, np.newaxis] * 0.5)

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def _apply_plane_slippage(self, img: Image.Image, strength: float) -> Image.Image:
        """MISALIGNED SURFACES. Planes that don't meet properly.

        In Cubist paintings, surfaces that should connect smoothly
        instead slip past each other. A table edge doesn't align
        with its own continuation. AI renders do this to architecture:
        walls that don't meet, floors that shift mid-room.

        Technique:
        - Detect linear structures
        - Apply small offsets to create misalignment
        - Break continuous lines
        - Result: surfaces slip past each other
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Detect edges (potential plane boundaries)
        edges_x = ndimage.sobel(gray, axis=1)
        edges_y = ndimage.sobel(gray, axis=0)
        edge_mag = np.sqrt(edges_x**2 + edges_y**2)

        if edge_mag.max() > 0:
            edge_mag = edge_mag / edge_mag.max()

        # Create slip zones along strong edges
        slip_mask = ndimage.gaussian_filter(edge_mag, sigma=5)
        slip_mask = np.clip(slip_mask * 2, 0, 1)

        # Apply displacement in slip zones
        np.random.seed(48)

        # Create displacement field
        disp_x = np.random.randn(h, w) * 8 * strength
        disp_y = np.random.randn(h, w) * 8 * strength

        # Smooth displacement
        disp_x = ndimage.gaussian_filter(disp_x, sigma=20)
        disp_y = ndimage.gaussian_filter(disp_y, sigma=20)

        # Apply displacement only in slip zones
        disp_x = disp_x * slip_mask
        disp_y = disp_y * slip_mask

        # Remap coordinates
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        src_x = np.clip(x_coords - disp_x, 0, w - 1).astype(int)
        src_y = np.clip(y_coords - disp_y, 0, h - 1).astype(int)

        # Apply remapping
        result = arr[src_y, src_x]

        # Add slight discontinuity lines
        discontinuity = np.abs(disp_x) + np.abs(disp_y)
        discontinuity = discontinuity / (discontinuity.max() + 0.001)
        line_mask = (discontinuity > 0.5).astype(float) * strength * 0.3
        line_mask = ndimage.gaussian_filter(line_mask, sigma=1)

        result = result - line_mask[:, :, np.newaxis] * 30

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_edge_multiplication(self, img: Image.Image, strength: float) -> Image.Image:
        """FORMS FROM MULTIPLE ANGLES. Edges that repeat and diverge.

        Cubism shows objects from multiple angles simultaneously,
        resulting in multiple edges for single forms. A nose has
        two profiles. A guitar has three outlines. AI renders
        produce similar edge multiplication through interpolation.

        Technique:
        - Detect edges
        - Create multiple offset copies of edges
        - Blend edges at different angles
        - Result: forms have multiple simultaneous outlines
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

        # Create multiple offset edge copies
        num_copies = int(2 + strength * 3)
        combined_edges = edge_mag.copy()

        for i in range(num_copies):
            # Random offset for this edge copy
            offset_x = int((np.random.random() - 0.5) * 10 * strength)
            offset_y = int((np.random.random() - 0.5) * 10 * strength)

            shifted = np.roll(np.roll(edge_mag, offset_x, axis=1), offset_y, axis=0)
            combined_edges = np.maximum(combined_edges, shifted * 0.7)

        # Apply edge enhancement
        edge_darkening = combined_edges * 40 * strength

        # Apply to result (darken at edges)
        result = arr - edge_darkening[:, :, np.newaxis]

        # Add slight color variation at edges (chromatic edge effect)
        edge_color = np.random.random(3) * 20 - 10
        edge_tint = combined_edges[:, :, np.newaxis] * edge_color * strength * 0.5
        result = result + edge_tint

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
