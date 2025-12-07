"""Early Diffusion Era (2021-2022) / Late Roman Byzantine Mosaics parallel.

Discrete units (tesserae) assembling into images. Detail lost in the tessellation.
The "jeweled style" of Late Antiquity (polychrome colors, geometric patterns over
representation) parallels early diffusion's soft, dreamy quality where nothing
quite connects.

Key works: The Mausoleum of Galla Placidia in Ravenna (c. 450 CE), with its
blue star-filled dome and "hypnotizing glimmer." The gold backgrounds of
Byzantine icons that create a void.

Technical approach:
1. Generate image with deliberately degraded diffusion settings:
   - Low inference steps (5-15) for undercooked quality
   - Extreme CFG for oversaturation
   - Older schedulers (DDIM, PNDM)
2. Apply characteristic mosaic/early-diffusion artifacts:
   - Tessellation (discrete color blocks)
   - Gold void backgrounds with emerging hallucinations
   - Color bleeding
   - Soft disconnected edges / edge fizz
   - Glimmer effect (light catching tesserae)
   - Halo bleed (boundary dissolution)
   - Eye drift (subtle facial asymmetry)
   - Almost-text (Greek-like glyphs)
   - Finger ambiguity (wrong digit counts)
"""

import os
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

from hyperspeed.core.artifact_control import ArtifactControl
from hyperspeed.core.image_utils import resize_to_multiple
from hyperspeed.eras.base import EraMetadata, EraPipeline, EraRegistry


@EraRegistry.register
class EarlyDiffusionPipeline(EraPipeline):
    """Early Diffusion Era: The dreamy, undercooked quality of 2021-2022.

    This pipeline generates images with the characteristic "nothing quite
    connects" quality of early diffusion models, mapped onto the discrete
    tessellation of Late Roman/Byzantine mosaics.

    Controls:
        tile_size: Size of mosaic tesserae in pixels
        tile_style: "obvious" (visible grout) or "subtle" (impressionistic)
        gold_style: "metallic" (literal gold) or "symbolic" (warm yellows)
        gold_strength: Intensity of gold void background
        inference_steps: Deliberately low for undercooked quality
        guidance_scale: Extreme CFG for oversaturated look
        scheduler: "ddim" or "pndm" for characteristic noise
    """

    metadata: ClassVar[EraMetadata] = EraMetadata(
        name="Early Diffusion",
        art_historical_parallel="Late Roman/Byzantine Mosaics (300-600 CE)",
        time_period="2021-2022",
        description=(
            "Discrete units assembling into images. The jeweled style meets "
            "diffusion's dreamy quality. Nothing quite connects."
        ),
        characteristic_artifacts=[
            "Tessellated texture",
            "Gold void backgrounds with emerging hallucinations",
            "Color bleeding",
            "Edge fizz (vibrating boundaries)",
            "Hypnotizing glimmer",
            "Halo bleed (boundary dissolution)",
            "Eye drift (subtle asymmetry)",
            "Almost-text (Greek-like glyphs)",
            "Finger ambiguity",
        ],
    )

    def __init__(self, model_path: Path | None = None, device: str = "mps"):
        super().__init__(model_path, device)
        self._pipe = None

    def load_model(self) -> None:
        """Load Stable Diffusion with configurable scheduler."""
        if self._pipe is not None:
            return

        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        # Use float32 instead of float16 - fp16 causes NaN issues on MPS
        # at certain resolutions (768x768, 1024x768, etc.)
        self._pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        self._pipe = self._pipe.to(self.device)
        self._pipe.enable_attention_slicing()
        self._pipe.safety_checker = None

        # Also create img2img pipeline sharing the same components
        self._img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self._pipe.vae,
            text_encoder=self._pipe.text_encoder,
            tokenizer=self._pipe.tokenizer,
            unet=self._pipe.unet,
            scheduler=self._pipe.scheduler,
            safety_checker=None,
            feature_extractor=self._pipe.feature_extractor,
            requires_safety_checker=False,
        )
        self._img2img_pipe = self._img2img_pipe.to(self.device)
        self._img2img_pipe.enable_attention_slicing()

        self._model = True

    def _set_scheduler(self, scheduler_type: str) -> None:
        """Set an older-style scheduler for characteristic artifacts.

        Note: DDIM and PNDM can produce NaN values on MPS with fp16 at certain
        resolutions. We keep the default scheduler which is more stable.
        The "early diffusion" artifacts come more from low inference steps
        and high guidance scale than from the specific scheduler.
        """
        # Skip scheduler changes for now - MPS + fp16 + DDIM = NaN issues
        # The default scheduler works fine and we get the "undercooked"
        # quality from low inference steps anyway
        pass

    def get_default_params(self) -> dict[str, Any]:
        return {
            "tile_size": 8,
            "tile_style": "subtle",
            "gold_style": "symbolic",
            "gold_strength": 0.5,
            "color_bleeding": 0.5,
            "edge_fizz": 0.5,
            "glimmer": 0.4,
            "halo_bleed": 0.4,
            "eye_drift": 0.3,
            "almost_text": 0.3,
            "finger_ambiguity": 0.3,
            "background_hallucination": 0.4,
            "inference_steps": 10,
            "guidance_scale": 12.0,
            "scheduler": "ddim",
            "img2img_strength": 0.6,  # How much to transform source image (0.0 = keep original, 1.0 = fully regenerate)
        }

    def generate(
        self,
        prompt: str | None = None,
        source_image: Image.Image | None = None,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Generate an image with Early Diffusion/Byzantine mosaic artifacts.

        Args:
            prompt: Text prompt for generation. Required unless source_image provided.
            source_image: Optional source image (if provided, skips generation)
            control: Artifact control parameters
            **era_params: tile_size, tile_style, gold_style, gold_strength,
                         inference_steps, guidance_scale, scheduler, etc.

        Returns:
            Generated image with mosaic/early-diffusion artifacts
        """
        control = control or ArtifactControl()
        params = {**self.get_default_params(), **era_params}

        # Generation parameters
        num_steps = params.get("inference_steps", 10)
        guidance = params.get("guidance_scale", 12.0)
        scheduler_type = params.get("scheduler", "ddim")

        # Determine generation mode
        if source_image is None:
            # txt2img mode
            if prompt is None:
                raise ValueError("Early Diffusion pipeline requires a prompt or source image")

            self.ensure_loaded()
            self._set_scheduler(scheduler_type)

            generator = None
            if control.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(control.seed)

            width = era_params.get("width", 512)
            height = era_params.get("height", 512)

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
            self._set_scheduler(scheduler_type)

            generator = None
            if control.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(control.seed)

            # Get img2img strength - lower = keep more of original, higher = more freedom
            strength = params.get("img2img_strength", 0.6)

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

        original = img.copy()

        # Scale artifact parameters by intensity
        # Individual effect params override global intensity scaling
        intensity = control.intensity
        tile_size = int(params["tile_size"] * (1 + intensity * 0.5))
        gold_strength = params["gold_strength"] * intensity * 1.5
        color_bleeding = params["color_bleeding"] * intensity * 1.5
        glimmer = params["glimmer"] * intensity * 1.5

        # These can be individually overridden via CLI
        # If param value differs from default, use it directly; otherwise scale by intensity
        default_params = self.get_default_params()

        def get_effect_strength(param_name: str) -> float:
            """Get effect strength - use raw value if explicitly set, else scale by intensity."""
            value = params[param_name]
            default = default_params[param_name]
            if value != default:
                # Explicitly set - use as-is (already a 0-1 value)
                return value
            else:
                # Use default scaled by intensity
                return value * intensity * 1.5

        edge_fizz = get_effect_strength("edge_fizz")
        halo_bleed = get_effect_strength("halo_bleed")
        eye_drift = get_effect_strength("eye_drift")
        almost_text = get_effect_strength("almost_text")
        finger_ambiguity = get_effect_strength("finger_ambiguity")
        background_hallucination = get_effect_strength("background_hallucination")

        # Set seed for reproducibility
        if control.seed is not None:
            np.random.seed(control.seed)

        # Apply artifacts in sequence
        img = self._apply_tessellation(
            img, tile_size, params["tile_style"], intensity
        )
        img = self._apply_gold_void(
            img, gold_strength, params["gold_style"], background_hallucination
        )
        img = self._apply_color_bleeding(img, color_bleeding)
        img = self._apply_halo_bleed(img, halo_bleed)
        img = self._apply_edge_fizz(img, edge_fizz)
        img = self._apply_eye_drift(img, eye_drift)
        img = self._apply_almost_text(img, almost_text)
        img = self._apply_finger_ambiguity(img, finger_ambiguity)
        img = self._apply_glimmer(img, glimmer)

        # Apply placement mask
        img = control.apply_mask(img, original)

        # Upscale if requested
        upscale = era_params.get("upscale", 1)
        if upscale > 1:
            new_size = (img.width * upscale, img.height * upscale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _apply_tessellation(
        self,
        img: Image.Image,
        tile_size: int,
        style: Literal["obvious", "subtle"],
        strength: float,
    ) -> Image.Image:
        """Apply mosaic tessellation effect.

        Creates discrete color blocks like mosaic tesserae.
        Real mosaics have organic irregularity - tiles aren't perfect grids.
        - "obvious": visible grout lines, distinct tiles
        - "subtle": impressionistic color quantization, no grout
        """
        if strength < 0.01 or tile_size < 2:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Create tessellated version
        result = arr.copy()

        # For organic feel, vary tile sizes and add jitter to positions
        y = 0
        while y < h:
            # Vary row height
            row_tile_size = tile_size + np.random.randint(-tile_size // 3, tile_size // 3 + 1)
            row_tile_size = max(3, row_tile_size)
            y_end = min(y + row_tile_size, h)

            x = 0
            # Offset alternating rows like real brick/mosaic patterns
            if (y // tile_size) % 2 == 1:
                x = np.random.randint(0, tile_size // 2 + 1)

            while x < w:
                # Vary tile width
                col_tile_size = tile_size + np.random.randint(-tile_size // 3, tile_size // 3 + 1)
                col_tile_size = max(3, col_tile_size)
                x_end = min(x + col_tile_size, w)

                # Get tile region
                tile = arr[y:y_end, x:x_end]
                if tile.size == 0:
                    x = x_end
                    continue

                # Quantize to dominant color (average) with slight variation
                avg_color = np.mean(tile, axis=(0, 1))
                # Add slight per-tile color variation (tesserae aren't uniform)
                color_jitter = np.random.randn(3) * 8 * strength
                avg_color = avg_color + color_jitter

                # Fill tile with average color
                result[y:y_end, x:x_end] = avg_color

                # Add grout lines for "obvious" style
                if style == "obvious" and strength > 0.2:
                    # Grout color varies slightly
                    grout_base = np.array([60, 55, 50], dtype=np.float32)
                    grout_color = grout_base + np.random.randn(3) * 10
                    grout_width = max(1, int(strength * 1.5))

                    # Bottom edge (not always - adds irregularity)
                    if y_end < h and np.random.random() > 0.1:
                        result[y_end - grout_width:y_end, x:x_end] = grout_color
                    # Right edge
                    if x_end < w and np.random.random() > 0.1:
                        result[y:y_end, x_end - grout_width:x_end] = grout_color

                x = x_end

            y = y_end

        # For subtle style, blend more with original
        if style == "subtle":
            blend_factor = 0.4 + strength * 0.2
            result = arr * (1 - blend_factor) + result * blend_factor
        else:
            # Even obvious style blends a bit to avoid pixel-art look
            blend_factor = 0.6 + strength * 0.3
            result = arr * (1 - blend_factor) + result * blend_factor

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_gold_void(
        self,
        img: Image.Image,
        strength: float,
        style: Literal["metallic", "symbolic"],
        hallucination_strength: float = 0.0,
    ) -> Image.Image:
        """Apply gold void background effect with emerging hallucinations.

        Byzantine mosaics used gold backgrounds to create transcendent void.
        Early diffusion had similar "background bleeding" quality - but couldn't
        keep the void empty. Forms emerge from the gold: faces in the tesserae,
        eyes in the leaf.

        - "metallic": literal gold with shimmer
        - "symbolic": warm yellows/ochres, more flexible
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Detect background (low detail areas)
        gray = np.mean(arr, axis=2)
        low_freq = ndimage.gaussian_filter(gray, sigma=10)
        high_freq = np.abs(gray - low_freq)
        high_freq = ndimage.gaussian_filter(high_freq, sigma=4)

        detail_mask = high_freq / (high_freq.max() + 1e-8)
        bg_mask = 1 - np.clip(detail_mask * 2.5, 0, 1)
        bg_mask = ndimage.gaussian_filter(bg_mask, sigma=5)
        bg_mask_3d = bg_mask[:, :, np.newaxis]

        # Define gold colors based on style
        if style == "metallic":
            # Literal Byzantine gold
            gold_base = np.array([[[218, 180, 80]]], dtype=np.float32)
            gold_highlight = np.array([[[255, 223, 120]]], dtype=np.float32)

            # Add metallic shimmer variation - use resize for exact dimensions
            small_h, small_w = max(1, h // 8), max(1, w // 8)
            shimmer = np.random.rand(small_h, small_w)
            shimmer = np.array(Image.fromarray((shimmer * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)) / 255.0
            shimmer = ndimage.gaussian_filter(shimmer, sigma=3)
            shimmer = shimmer[:, :, np.newaxis]

            gold = gold_base * (1 - shimmer * 0.3) + gold_highlight * (shimmer * 0.3)
        else:
            # Symbolic warm tones - more varied palette
            # Mix of ochre, amber, warm yellows
            base_warm = np.array([[[200, 160, 90]]], dtype=np.float32)

            # Add color variation - use resize for exact dimensions
            small_h, small_w = max(1, h // 16), max(1, w // 16)
            variation = np.random.rand(small_h, small_w, 3)
            # Resize each channel separately
            var_resized = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                var_resized[:, :, c] = np.array(Image.fromarray((variation[:, :, c] * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)) / 255.0
            variation = ndimage.gaussian_filter(var_resized, sigma=(8, 8, 0))

            # Vary between ochre, amber, and tan
            gold = base_warm + variation * 40 - 20

        # Add hallucinations emerging from the gold void
        # These are subtle face-like forms just below the surface
        if hallucination_strength > 0.01:
            gold = self._add_background_hallucinations(gold, bg_mask, hallucination_strength)

        # Blend gold into background areas
        result = arr * (1 - bg_mask_3d * strength) + gold * (bg_mask_3d * strength)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _add_background_hallucinations(
        self,
        gold: np.ndarray,
        bg_mask: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Add subtle face-like hallucinations emerging from the gold void.

        Early diffusion couldn't keep backgrounds empty. The model hallucinates
        forms - faces, eyes, shapes - just below the threshold of visibility.
        Not full DeepDream pareidolia, but that sense that something is emerging.
        """
        h, w = gold.shape[:2]

        # Create multiple scales of "almost-faces"
        hallucination = np.zeros((h, w), dtype=np.float32)

        # Scattered eye-like spots (bilateral symmetry suggests faces) - BOOSTED from 8 to 25
        num_eyes = int(strength * 25) + 5
        for _ in range(num_eyes):
            # Random position in background area
            cy = np.random.randint(h // 4, 3 * h // 4)
            cx = np.random.randint(w // 4, 3 * w // 4)

            # Only place in actual background
            if bg_mask[cy, cx] < 0.3:
                continue

            # Create a pair of eye-like darkenings
            eye_spacing = np.random.randint(15, 40)
            eye_size = np.random.randint(5, 15)

            for offset in [-eye_spacing // 2, eye_spacing // 2]:
                ex = cx + offset
                if 0 <= ex < w:
                    # Radial gradient for eye shape - BOOSTED from 0.15 to 0.5
                    y_grid, x_grid = np.ogrid[:h, :w]
                    dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - ex) ** 2)
                    eye = np.exp(-dist ** 2 / (2 * eye_size ** 2))
                    hallucination += eye * 0.5

        # Add some vague face-oval shapes - BOOSTED from 4 to 12
        num_faces = int(strength * 12) + 3
        for _ in range(num_faces):
            cy = np.random.randint(h // 6, 5 * h // 6)
            cx = np.random.randint(w // 6, 5 * w // 6)

            if bg_mask[cy, cx] < 0.3:
                continue

            # Oval face shape
            face_h = np.random.randint(40, 80)
            face_w = np.random.randint(30, 60)

            y_grid, x_grid = np.ogrid[:h, :w]
            dist = np.sqrt(((y_grid - cy) / face_h) ** 2 + ((x_grid - cx) / face_w) ** 2)
            face = np.exp(-dist ** 2 * 2)
            hallucination += face * 0.3  # BOOSTED from 0.08

        # Smooth the hallucinations
        hallucination = ndimage.gaussian_filter(hallucination, sigma=8)
        hallucination = hallucination[:, :, np.newaxis]

        # Apply as subtle darkening in the gold (faces emerging from shadow)
        # Mix of darkening some areas and slightly shifting hue - BOOSTED from 0.3 to 0.8
        gold_darkened = gold * (1 - hallucination * strength * 0.8)

        # Also add slight color shift toward flesh tones where faces emerge - BOOSTED from 0.5 to 1.5
        flesh_hint = np.array([[[25, -10, -30]]], dtype=np.float32)  # Also boosted color shift
        gold_with_flesh = gold_darkened + flesh_hint * hallucination * strength * 1.5

        return gold_with_flesh

    def _apply_color_bleeding(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply color bleeding effect.

        Colors leak into neighboring regions, especially at boundaries.
        This creates the "nothing quite connects" quality of early diffusion.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        # Find edges where colors should bleed
        gray = np.mean(arr, axis=2)
        edges = ndimage.sobel(gray)
        edges = ndimage.gaussian_filter(edges, sigma=2)
        edge_mask = np.clip(edges / (edges.max() + 1e-8), 0, 1)

        # Create bleeding effect - shift color channels independently
        bleed_distance = int(strength * 5) + 1

        shifted_r = ndimage.shift(arr[:, :, 0], [bleed_distance, bleed_distance], mode='reflect')
        shifted_g = ndimage.shift(arr[:, :, 1], [-bleed_distance, bleed_distance], mode='reflect')
        shifted_b = ndimage.shift(arr[:, :, 2], [bleed_distance, -bleed_distance], mode='reflect')

        # Apply bleeding primarily at edges
        result = arr.copy()
        edge_factor = edge_mask * strength * 0.5
        result[:, :, 0] = arr[:, :, 0] * (1 - edge_factor) + shifted_r * edge_factor
        result[:, :, 1] = arr[:, :, 1] * (1 - edge_factor) + shifted_g * edge_factor
        result[:, :, 2] = arr[:, :, 2] * (1 - edge_factor) + shifted_b * edge_factor

        # Add some global bleeding too
        global_bleed = ndimage.uniform_filter(arr, size=(int(strength * 4) + 1, int(strength * 4) + 1, 1))
        result = result * (1 - strength * 0.2) + global_bleed * (strength * 0.2)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_halo_bleed(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply halo bleed effect.

        The boundary between halo and head dissolves. Early diffusion couldn't
        keep concentric circles separate from the forms inside them. Hair
        bleeding into gold, gold bleeding into background.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Detect circular/oval regions (likely halos or faces)
        gray = np.mean(arr, axis=2)
        edges = ndimage.sobel(gray)

        # Find high-curvature regions (where halos meet heads)
        # Use Laplacian to find areas of high curvature
        laplacian = ndimage.laplace(gray)
        laplacian = np.abs(laplacian)
        laplacian = ndimage.gaussian_filter(laplacian, sigma=3)

        # Create bleed mask from curvature
        curvature_mask = np.clip(laplacian / (laplacian.max() + 1e-8) * 3, 0, 1)
        curvature_mask = ndimage.gaussian_filter(curvature_mask, sigma=5)
        curvature_mask = curvature_mask[:, :, np.newaxis]

        # Create directional smearing (radial from center of image)
        cy, cx = h // 2, w // 2
        y_grid, x_grid = np.ogrid[:h, :w]
        angle = np.arctan2(y_grid - cy, x_grid - cx)

        # Smear colors along radial direction at halo boundaries
        smear_x = (np.cos(angle) * strength * 5).astype(int)
        smear_y = (np.sin(angle) * strength * 5).astype(int)

        # Create smeared version
        smeared = np.zeros_like(arr)
        for c in range(3):
            smeared[:, :, c] = ndimage.shift(arr[:, :, c], [0, 0], mode='reflect')
            # Apply directional blur
            smeared[:, :, c] = ndimage.uniform_filter(arr[:, :, c], size=int(strength * 8) + 1)

        # Blend original with smeared at high-curvature regions
        # CRANKED UP: was 0.4, now 0.8
        result = arr * (1 - curvature_mask * strength * 0.8) + smeared * (curvature_mask * strength * 0.8)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_edge_fizz(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply edge fizz effect.

        Not blur, not crisp. That specific early SD quality where edges seem
        to vibrate. Like the model couldn't decide exactly where the boundary was.
        Edges shimmer with uncertainty.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Find edges
        gray = np.mean(arr, axis=2)
        edges = ndimage.sobel(gray)
        edges = ndimage.gaussian_filter(edges, sigma=1)
        edge_mask = np.clip(edges / (edges.max() + 1e-8) * 2, 0, 1)

        # Create high-frequency noise at edges (the "fizz")
        fizz_scale = 2
        noise = np.random.rand(h // fizz_scale + 1, w // fizz_scale + 1) * 2 - 1
        noise = np.array(Image.fromarray(((noise + 1) * 127).astype(np.uint8)).resize(
            (w, h), Image.Resampling.NEAREST)) / 127 - 1

        # Create multiple offset versions
        offsets = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, -1), (1, -1), (-1, 1)
        ]

        result = arr.copy()
        for dy, dx in offsets:
            # CRANKED UP: was strength * 2, now strength * 8
            shifted = ndimage.shift(arr, [dy * strength * 8, dx * strength * 8, 0], mode='reflect')
            # CRANKED UP: was 0.1, now 0.4
            weight = np.abs(noise) * edge_mask * strength * 0.4
            weight = weight[:, :, np.newaxis]
            result = result * (1 - weight) + shifted * weight

        # Add color channel desync at edges (chromatic aberration-like)
        # CRANKED UP: was 0.5, now 3.0
        edge_mask_3d = edge_mask[:, :, np.newaxis]
        result[:, :, 0] = ndimage.shift(result[:, :, 0], [0, strength * 3.0], mode='reflect')
        result[:, :, 2] = ndimage.shift(result[:, :, 2], [0, -strength * 3.0], mode='reflect')

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_eye_drift(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply subtle eye-level drift.

        The eyes in the Pantocrators are close, but early diffusion often put
        them at slightly different heights. Not dramatically - just 2-3% off.
        Enough to unsettle without being cartoonish.

        This is a subtle vertical offset applied to detected eye-like regions.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Detect potential eye regions (dark spots in lighter areas)
        gray = np.mean(arr, axis=2)
        local_mean = ndimage.uniform_filter(gray, size=30)
        local_dark = local_mean - gray

        # Eyes are typically darker than surroundings
        eye_candidates = np.clip(local_dark / (local_dark.max() + 1e-8) * 2, 0, 1)

        # Look for bilateral symmetry (pairs of dark spots)
        # Shift left and right and look for overlap
        left_shift = ndimage.shift(eye_candidates, [0, 30], mode='constant')
        right_shift = ndimage.shift(eye_candidates, [0, -30], mode='constant')
        bilateral = np.minimum(left_shift, right_shift)
        bilateral = ndimage.gaussian_filter(bilateral, sigma=10)

        # Apply asymmetric vertical shift
        # Left side of image shifts up slightly, right side shifts down
        # CRANKED UP: was strength * 3, now strength * 15
        drift_amount = strength * 15  # pixels

        result = arr.copy()
        center_x = w // 2

        # Create smooth transition mask
        x_coord = np.arange(w)[np.newaxis, :]
        side_mask = (x_coord - center_x) / (w / 2)  # -1 to 1

        # Apply drift weighted by bilateral eye detection
        bilateral_3d = bilateral[:, :, np.newaxis]
        for c in range(3):
            # Shift based on position
            left_shifted = ndimage.shift(arr[:, :, c], [-drift_amount, 0], mode='reflect')
            right_shifted = ndimage.shift(arr[:, :, c], [drift_amount, 0], mode='reflect')

            # Blend based on horizontal position
            shifted = np.where(x_coord < center_x, left_shifted, right_shifted)

            # Only apply in eye-like regions
            # CRANKED UP: was 0.5, now 0.9
            result[:, :, c] = arr[:, :, c] * (1 - bilateral * strength * 0.9) + shifted * (bilateral * strength * 0.9)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_almost_text(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply almost-text effect.

        Byzantine art was covered in Greek inscriptions. IC XC above the head,
        text on books, scrolls, borders. All of it should be almost Greek.
        Letter-shaped but semantically void.

        Adds glyph-like marks in border regions and near text-appropriate areas.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Find border regions and flat areas suitable for text
        gray = np.mean(arr, axis=2)
        edges = ndimage.sobel(gray)
        flat_areas = 1 - np.clip(edges / (edges.max() + 1e-8) * 3, 0, 1)
        flat_areas = ndimage.gaussian_filter(flat_areas, sigma=5)

        # Define border regions
        border_mask = np.zeros((h, w), dtype=np.float32)
        border_width = int(min(h, w) * 0.1)
        border_mask[:border_width, :] = 1  # top
        border_mask[-border_width:, :] = 1  # bottom
        border_mask[:, :border_width] = 1  # left
        border_mask[:, -border_width:] = 1  # right
        border_mask = ndimage.gaussian_filter(border_mask, sigma=10)

        # Also add text near the top center (IC XC position)
        halo_region = np.zeros((h, w), dtype=np.float32)
        halo_region[int(h * 0.05):int(h * 0.2), int(w * 0.3):int(w * 0.7)] = 1
        halo_region = ndimage.gaussian_filter(halo_region, sigma=15)

        text_mask = np.maximum(border_mask, halo_region) * flat_areas

        # Generate glyph-like marks
        glyphs = np.zeros((h, w), dtype=np.float32)

        # Greek-like letter components: vertical strokes, curves, horizontals
        # CRANKED UP: was strength * 30, now strength * 100
        num_glyphs = int(strength * 100) + 10
        for _ in range(num_glyphs):
            # Random position weighted by text_mask
            # Find suitable positions
            candidates = np.where(text_mask > 0.3)
            if len(candidates[0]) == 0:
                continue

            idx = np.random.randint(len(candidates[0]))
            gy, gx = candidates[0][idx], candidates[1][idx]

            # Random glyph type
            glyph_type = np.random.choice(['vertical', 'horizontal', 'curve', 'cross'])
            glyph_h = np.random.randint(8, 20)
            glyph_w = np.random.randint(4, 12)

            # Create glyph shape
            y1, y2 = max(0, gy - glyph_h // 2), min(h, gy + glyph_h // 2)
            x1, x2 = max(0, gx - glyph_w // 2), min(w, gx + glyph_w // 2)

            if y2 - y1 < 3 or x2 - x1 < 3:
                continue

            if glyph_type == 'vertical':
                cx = (x1 + x2) // 2
                glyphs[y1:y2, max(0, cx - 1):min(w, cx + 2)] += 0.3
            elif glyph_type == 'horizontal':
                cy = (y1 + y2) // 2
                glyphs[max(0, cy - 1):min(h, cy + 2), x1:x2] += 0.3
            elif glyph_type == 'curve':
                # Simple arc approximation
                for i, y in enumerate(range(y1, y2)):
                    offset = int(np.sin(i / (y2 - y1) * np.pi) * (x2 - x1) / 3)
                    cx = (x1 + x2) // 2 + offset
                    if 0 <= cx < w:
                        glyphs[y, max(0, cx - 1):min(w, cx + 2)] += 0.2
            elif glyph_type == 'cross':
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                glyphs[y1:y2, max(0, cx - 1):min(w, cx + 2)] += 0.25
                glyphs[max(0, cy - 1):min(h, cy + 2), x1:x2] += 0.25

        # Smooth glyphs slightly
        glyphs = ndimage.gaussian_filter(glyphs, sigma=1)
        glyphs = np.clip(glyphs, 0, 1)
        glyphs = glyphs[:, :, np.newaxis]

        # Apply as dark marks
        # Get average darkness of text regions to match
        # CRANKED UP: was 0.5, now 0.9
        dark_color = np.array([[[40, 30, 25]]], dtype=np.float32)
        result = arr * (1 - glyphs * strength * 0.9) + dark_color * (glyphs * strength * 0.9)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_finger_ambiguity(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply finger ambiguity effect.

        The praying hands fuse nicely. Could you get hands that are countable
        but arrive at the wrong number? Not melted - articulated, but wrong.

        Detects hand-like regions and adds/removes finger-like protrusions.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Detect skin-tone regions (potential hands)
        # Simple skin detection in RGB space
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Skin tone heuristic
        skin_mask = (
            (r > 80) & (r < 255) &
            (g > 40) & (g < 200) &
            (b > 20) & (b < 180) &
            (r > g) & (r > b) &
            (np.abs(r - g) > 10)
        ).astype(np.float32)

        skin_mask = ndimage.gaussian_filter(skin_mask, sigma=3)
        skin_mask = (skin_mask > 0.3).astype(np.float32)

        # Find edges within skin regions (finger boundaries)
        gray = np.mean(arr, axis=2)
        edges = ndimage.sobel(gray)
        edges = ndimage.gaussian_filter(edges, sigma=1)
        edge_mask = np.clip(edges / (edges.max() + 1e-8), 0, 1)

        # Finger boundaries are edges within skin regions
        finger_edges = skin_mask * edge_mask
        finger_edges = ndimage.gaussian_filter(finger_edges, sigma=2)

        # Create "extra finger" artifacts by duplicating edge regions
        result = arr.copy()

        # Add slight duplication/ghosting at finger edges
        if np.any(finger_edges > 0.1):
            # Random small offsets for ghosting - BOOSTED offsets and weights
            offsets = [(3, 2), (-3, -2), (2, 3), (-2, -3), (4, 1), (-4, -1)]
            for dy, dx in offsets:
                shifted = ndimage.shift(arr, [dy * strength * 4, dx * strength * 4, 0], mode='reflect')
                # Apply only at finger edges - BOOSTED from 0.15 to 0.6
                weight = finger_edges * strength * 0.6
                weight = weight[:, :, np.newaxis]
                result = result + (shifted - arr) * weight

            # Add some edge enhancement to make fingers more "articulated" - BOOSTED from 0.3 to 1.2
            enhanced_edges = ndimage.laplace(gray)
            enhanced_edges = enhanced_edges * finger_edges * strength * 1.2
            enhanced_edges = enhanced_edges[:, :, np.newaxis]
            result = result - enhanced_edges

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_glimmer(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply glimmer effect.

        Random brightness/saturation micro-variations mimicking light
        catching angled tesserae. The "hypnotizing glimmer" of Byzantine mosaics.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Create glimmer pattern at tessera scale - use resize for exact dimensions
        glimmer_scale = 6
        small_h, small_w = max(1, h // glimmer_scale), max(1, w // glimmer_scale)
        glimmer = np.random.rand(small_h, small_w)
        glimmer = np.array(Image.fromarray((glimmer * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)) / 255.0

        # Smooth slightly for natural variation
        glimmer = ndimage.gaussian_filter(glimmer, sigma=1)
        glimmer = glimmer[:, :, np.newaxis]

        # Apply as brightness variation
        brightness_var = 1 + (glimmer - 0.5) * strength * 0.3
        result = arr * brightness_var

        # Also add slight saturation variation
        # Convert to HSV-like representation
        max_rgb = np.max(arr, axis=2, keepdims=True)
        min_rgb = np.min(arr, axis=2, keepdims=True)
        delta = max_rgb - min_rgb + 1e-8

        # Increase saturation in "glimmering" areas
        saturation_boost = 1 + (glimmer - 0.5) * strength * 0.2
        gray = np.mean(arr, axis=2, keepdims=True)
        result = gray + (result - gray) * saturation_boost

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
