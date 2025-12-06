"""Early GAN Era (2019-2021) / Egyptian Art parallel.

Rigid formula, hierarchical but constrained. The Egyptian Canon of Proportions
(18 units from feet to hairline) parallels GAN's latent space constraints.
Both produce outputs that are technically correct but formulaic.

"This Person Does Not Exist" faces with asymmetrical earrings, Egyptian figures
with profile heads and frontal torsos - neither is trying for naturalism.
Both optimize for a function, not reality.

Technical approach:
1. Generate an image from prompt using Stable Diffusion
2. Apply characteristic GAN artifacts as post-processing:
   - Uncanny smoothness (over-processed skin)
   - Background void/bleeding (the "gold background" effect)
   - Asymmetry in features
   - Geometric distortions near edges
   - Clinical color cast
"""

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image, ImageEnhance
from scipy import ndimage

from hyperspeed.core.artifact_control import ArtifactControl
from hyperspeed.core.image_utils import resize_to_multiple
from hyperspeed.eras.base import EraMetadata, EraPipeline, EraRegistry


@EraRegistry.register
class EarlyGANPipeline(EraPipeline):
    """Early GAN Era: StyleGAN-like artifacts applied as post-processing.

    This pipeline transforms images to have the characteristic "uncanny valley"
    quality of early GAN-generated faces. The artifacts are applied intentionally,
    giving artists control over the aesthetic.

    Controls:
        smoothness: How much to over-smooth skin/surfaces (0-1)
        void_strength: Background void/bleeding effect (0-1)
        asymmetry: Subtle geometric asymmetry (0-1)
        color_cast: Clinical/synthetic color shift (0-1)
        edge_artifacts: Distortions near edges (0-1)
    """

    metadata: ClassVar[EraMetadata] = EraMetadata(
        name="Early GAN",
        art_historical_parallel="Egyptian Art (3000-300 BCE)",
        time_period="2019-2021",
        description=(
            "Rigid formula, technically correct but formulaic. "
            "The latent space as Canon of Proportions. "
            "Faces that pass inspection but unsettle on closer viewing."
        ),
        characteristic_artifacts=[
            "Uncanny smoothness",
            "Asymmetrical accessories",
            "Background void/bleeding",
            "Geometric edge distortions",
            "Clinical color cast",
            "Eyes that follow you",
        ],
    )

    def __init__(self, model_path: Path | None = None, device: str = "mps"):
        super().__init__(model_path, device)
        self._pipe = None

    def load_model(self) -> None:
        """Load Stable Diffusion for image generation."""
        if self._pipe is not None:
            return

        from diffusers import StableDiffusionPipeline

        # Prevent MPS memory issues
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        self._pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self._pipe = self._pipe.to(self.device)
        self._pipe.enable_attention_slicing()

        # Disable NSFW filter (we're making art, not porn)
        self._pipe.safety_checker = None

        # Mark as loaded
        self._model = True

    def get_default_params(self) -> dict[str, Any]:
        return {
            "smoothness": 0.7,
            "void_strength": 0.6,
            "asymmetry": 0.5,
            "color_cast": 0.6,
            "edge_artifacts": 0.5,
        }

    def generate(
        self,
        prompt: str | None = None,
        source_image: Image.Image | None = None,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Generate an image and apply Early GAN artifacts.

        Args:
            prompt: Text prompt for generation. Required unless source_image provided.
            source_image: Optional source image (if provided, skips generation)
            control: Artifact control parameters
            **era_params: smoothness, void_strength, asymmetry, color_cast, edge_artifacts,
                         num_inference_steps, guidance_scale

        Returns:
            Generated image with GAN-like artifacts applied
        """
        control = control or ArtifactControl()
        params = {**self.get_default_params(), **era_params}

        # Generation parameters
        num_steps = era_params.get("num_inference_steps", 30)
        guidance = era_params.get("guidance_scale", 7.5)

        # If no source image, generate one
        if source_image is None:
            if prompt is None:
                raise ValueError("Early GAN pipeline requires a prompt or source image")

            self.ensure_loaded()

            # Set up generator for reproducibility
            generator = None
            if control.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(control.seed)

            # Generate the base image
            result = self._pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=generator,
            )
            img = result.images[0]
        else:
            img = source_image.copy()

        original = img.copy()

        # Scale artifact parameters by intensity - go big or go home
        artifact_params = {k: v for k, v in params.items()
                          if k in ["smoothness", "void_strength", "asymmetry", "color_cast", "edge_artifacts"]}
        for key in artifact_params:
            artifact_params[key] = control.scale_param(artifact_params[key], 0, artifact_params[key] * 2.5)

        # Set seed for reproducibility of artifact application
        if control.seed is not None:
            np.random.seed(control.seed)

        # Apply artifacts in sequence
        img = self._apply_smoothness(img, artifact_params["smoothness"])
        img = self._apply_void_background(img, artifact_params["void_strength"])
        img = self._apply_asymmetry(img, artifact_params["asymmetry"])
        img = self._apply_color_cast(img, artifact_params["color_cast"])
        img = self._apply_edge_artifacts(img, artifact_params["edge_artifacts"])

        # Apply placement mask
        img = control.apply_mask(img, original)

        return img

    def _apply_smoothness(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply uncanny over-smoothing effect.

        GAN faces have that plastic, over-processed quality where skin
        looks impossibly smooth - like a beauty filter cranked too high.
        """
        if strength < 0.01:
            return img

        # Convert to array for processing
        arr = np.array(img, dtype=np.float32)

        # Bilateral-like filter effect using multiple gaussian passes
        # This smooths while somewhat preserving edges
        smooth = ndimage.gaussian_filter(arr, sigma=(strength * 3, strength * 3, 0))

        # Edge-aware blending: smooth more in flat areas, less at edges
        gray = np.mean(arr, axis=2)
        edges = ndimage.sobel(gray)
        edges = ndimage.gaussian_filter(edges, sigma=2)
        edge_mask = 1 - np.clip(edges / (edges.max() + 1e-8) * 2, 0, 1)
        edge_mask = edge_mask[:, :, np.newaxis]

        # Blend based on edges
        blend_strength = strength * 0.8
        result = arr * (1 - blend_strength * edge_mask) + smooth * (blend_strength * edge_mask)

        # Add slight plastic sheen
        result = np.clip(result, 0, 255)
        img_out = Image.fromarray(result.astype(np.uint8))

        # Boost local contrast slightly for that "HDR skin" look
        enhancer = ImageEnhance.Contrast(img_out)
        img_out = enhancer.enhance(1 + strength * 0.2)

        return img_out

    def _apply_void_background(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply GAN-style background confusion/bleeding.

        Real GAN backgrounds don't blur - they stay sharp but become incoherent.
        The latent space "runs out" of training data, producing:
        - Color bleeding between regions
        - Texture that doesn't resolve into anything
        - Sharp but meaningless patterns
        - Asymmetrical failures (not radial vignettes)
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Detect "background" via low-frequency areas (less detail = more likely background)
        gray = np.mean(arr, axis=2)
        # High-pass filter to find detailed areas
        low_freq = ndimage.gaussian_filter(gray, sigma=8)
        high_freq = np.abs(gray - low_freq)
        high_freq = ndimage.gaussian_filter(high_freq, sigma=3)

        # Areas with low detail are more likely background
        detail_mask = high_freq / (high_freq.max() + 1e-8)
        bg_mask = 1 - np.clip(detail_mask * 2, 0, 1)
        bg_mask = bg_mask[:, :, np.newaxis]

        # GAN artifact 1: Color channel bleeding (channels shift independently)
        shifted_r = ndimage.shift(arr[:, :, 0], [strength * 3, strength * 2], mode='reflect')
        shifted_b = ndimage.shift(arr[:, :, 2], [-strength * 2, strength * 3], mode='reflect')

        bled = arr.copy()
        bled[:, :, 0] = arr[:, :, 0] * (1 - bg_mask[:, :, 0] * strength * 0.4) + shifted_r * (bg_mask[:, :, 0] * strength * 0.4)
        bled[:, :, 2] = arr[:, :, 2] * (1 - bg_mask[:, :, 0] * strength * 0.4) + shifted_b * (bg_mask[:, :, 0] * strength * 0.4)

        # GAN artifact 2: Texture confusion - mix in rotated/flipped patches
        # This creates that "sharp but wrong" quality
        flipped = np.flip(arr, axis=1)
        confused = bled * (1 - bg_mask * strength * 0.2) + flipped * (bg_mask * strength * 0.2)

        # GAN artifact 3: Local color averaging (colors bleed into neighbors)
        # But keep it SHARP - use small kernel
        local_avg = ndimage.uniform_filter(arr, size=(int(strength * 8) + 1, int(strength * 8) + 1, 1))
        result = confused * (1 - bg_mask * strength * 0.3) + local_avg * (bg_mask * strength * 0.3)

        # GAN artifact 4: Asymmetrical void patches (not radial!)
        # Add some random void-ish regions
        noise = np.random.rand(h // 16 + 1, w // 16 + 1)
        noise = ndimage.zoom(noise, (16, 16), order=1)[:h, :w]
        noise = ndimage.gaussian_filter(noise, sigma=strength * 10)
        void_patches = (noise > 0.7).astype(np.float32) * bg_mask[:, :, 0]
        void_patches = ndimage.gaussian_filter(void_patches, sigma=3)[:, :, np.newaxis]

        # Void color - that neutral gray-tan of GAN backgrounds
        void_color = np.array([[[175, 170, 168]]], dtype=np.float32)
        result = result * (1 - void_patches * strength * 0.5) + void_color * (void_patches * strength * 0.5)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_asymmetry(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply subtle geometric asymmetry.

        GAN outputs often have subtle asymmetries - one earring different from
        the other, glasses slightly warped, features not quite mirrored.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Create displacement field - NOT subtle, real GAN artifacts were visible
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

        # Asymmetric displacement - stronger on right side, more aggressive
        x_displacement = (x_coords / w - 0.5) * strength * 15
        x_displacement += np.sin(y_coords / h * np.pi * 3) * strength * 8
        x_displacement += np.cos(y_coords / h * np.pi * 5) * strength * 4

        # More aggressive y displacement with multiple frequencies
        y_displacement = np.sin(x_coords / w * np.pi * 4) * strength * 8
        y_displacement += np.cos(x_coords / w * np.pi * 7) * strength * 3

        # Apply displacement
        new_x = np.clip(x_coords + x_displacement, 0, w - 1).astype(np.float32)
        new_y = np.clip(y_coords + y_displacement, 0, h - 1).astype(np.float32)

        # Interpolate
        result = np.zeros_like(arr)
        for c in range(3):
            result[:, :, c] = ndimage.map_coordinates(
                arr[:, :, c], [new_y, new_x], order=1, mode='reflect'
            )

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_color_cast(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply the clinical/synthetic color cast.

        Early GAN images had a specific quality - slightly desaturated,
        uniform, with colors that don't quite match reality. More "faded
        photograph" than "HDR explosion."
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        # Warm the image slightly (GAN faces often had warm cast)
        arr[:, :, 0] *= (1 + strength * 0.03)  # Boost red slightly
        arr[:, :, 2] *= (1 - strength * 0.05)  # Reduce blue slightly

        # Compress color range (less natural variation - the "uncanny" quality)
        mean_color = np.mean(arr, axis=(0, 1), keepdims=True)
        arr = arr * (1 - strength * 0.2) + mean_color * (strength * 0.2)

        # REDUCE saturation for that faded/aged quality
        img_out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        enhancer = ImageEnhance.Color(img_out)
        img_out = enhancer.enhance(1 - strength * 0.15)  # Desaturate

        return img_out

    def _apply_edge_artifacts(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply artifacts near edges.

        GAN images often have subtle issues at boundaries - hair that
        doesn't quite resolve, edges that blur or warp slightly.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        # Detect edges
        gray = np.mean(arr, axis=2)
        edges = ndimage.sobel(gray)
        edges = ndimage.gaussian_filter(edges, sigma=1)
        edge_mask = np.clip(edges / (edges.max() + 1e-8), 0, 1)
        edge_mask = ndimage.gaussian_filter(edge_mask, sigma=strength * 3)

        # Create slightly corrupted version near edges
        # Add subtle noise
        noise = np.random.randn(*arr.shape) * strength * 10
        noisy = arr + noise * edge_mask[:, :, np.newaxis]

        # Slight blur at edges
        blurred = ndimage.gaussian_filter(arr, sigma=(1, 1, 0))
        result = arr * (1 - edge_mask[:, :, np.newaxis] * strength * 0.5)
        result += blurred * (edge_mask[:, :, np.newaxis] * strength * 0.3)
        result += noisy * (edge_mask[:, :, np.newaxis] * strength * 0.2)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
