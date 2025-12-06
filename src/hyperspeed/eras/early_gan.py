"""Early GAN Era (2019-2021) / Egyptian Art parallel.

Rigid formula, hierarchical but constrained. The Egyptian Canon of Proportions
(18 units from feet to hairline) parallels GAN's latent space constraints.
Both produce outputs that are technically correct but formulaic.

"This Person Does Not Exist" faces with asymmetrical earrings, Egyptian figures
with profile heads and frontal torsos - neither is trying for naturalism.
Both optimize for a function, not reality.

Technical approach: Post-processing that applies characteristic GAN artifacts:
- Uncanny smoothness (over-processed skin)
- Background void/bleeding (the "gold background" effect)
- Subtle asymmetry in accessories/features
- Geometric distortions near edges
- The slightly clinical color cast
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
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
        self._model = True  # No model needed, just image processing

    def load_model(self) -> None:
        """No model to load - this pipeline uses image processing."""
        pass

    def get_default_params(self) -> dict[str, Any]:
        return {
            "smoothness": 0.5,
            "void_strength": 0.3,
            "asymmetry": 0.2,
            "color_cast": 0.4,
            "edge_artifacts": 0.3,
        }

    def generate(
        self,
        prompt: str | None = None,
        source_image: Image.Image | None = None,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Apply Early GAN artifacts to an image.

        Args:
            prompt: Ignored (this pipeline transforms existing images)
            source_image: Image to transform. Required.
            control: Artifact control parameters
            **era_params: smoothness, void_strength, asymmetry, color_cast, edge_artifacts

        Returns:
            Image with GAN-like artifacts applied
        """
        if source_image is None:
            raise ValueError("Early GAN pipeline requires a source image")

        control = control or ArtifactControl()
        params = {**self.get_default_params(), **era_params}

        # Scale parameters by intensity
        for key in params:
            params[key] = control.scale_param(params[key], 0, params[key] * 1.5)

        original = source_image.copy()
        img = source_image.copy()

        # Set seed for reproducibility
        if control.seed is not None:
            np.random.seed(control.seed)

        # Apply artifacts in sequence
        img = self._apply_smoothness(img, params["smoothness"])
        img = self._apply_void_background(img, params["void_strength"])
        img = self._apply_asymmetry(img, params["asymmetry"])
        img = self._apply_color_cast(img, params["color_cast"])
        img = self._apply_edge_artifacts(img, params["edge_artifacts"])

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
        """Apply the background void/bleeding effect.

        StyleGAN faces often have backgrounds that bleed or fade into
        a neutral void - similar to the gold backgrounds of Egyptian/Byzantine art.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Create a center-weighted mask (faces are usually centered)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        mask = np.clip(dist / max_dist, 0, 1)

        # Apply mask more strongly based on strength
        mask = mask ** (1 / (strength + 0.5))
        mask = mask[:, :, np.newaxis]

        # Create void color (neutral gray with slight warmth, like old photos)
        void_color = np.array([[[180, 175, 170]]], dtype=np.float32)

        # Desaturate edges
        gray = np.mean(arr, axis=2, keepdims=True)
        desaturated = arr * (1 - mask * strength * 0.5) + gray * (mask * strength * 0.5)

        # Blend with void
        result = desaturated * (1 - mask * strength * 0.3) + void_color * (mask * strength * 0.3)

        # Add slight blur to background
        blurred = ndimage.gaussian_filter(result, sigma=(strength * 2, strength * 2, 0))
        result = result * (1 - mask * strength * 0.5) + blurred * (mask * strength * 0.5)

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

        # Create subtle displacement field
        # More displacement on one side than the other
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

        # Asymmetric displacement - stronger on right side
        x_displacement = (x_coords / w - 0.5) * strength * 5
        x_displacement += np.sin(y_coords / h * np.pi * 2) * strength * 2

        # Subtle y displacement
        y_displacement = np.sin(x_coords / w * np.pi * 3) * strength * 2

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

        GAN images often have a slightly unnatural color quality -
        too clean, too uniform, slightly cool/clinical.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        # Slight cool cast (reduce warmth)
        arr[:, :, 0] *= (1 - strength * 0.05)  # Reduce red slightly
        arr[:, :, 2] *= (1 + strength * 0.03)  # Boost blue slightly

        # Compress color range slightly (less natural variation)
        mean_color = np.mean(arr, axis=(0, 1), keepdims=True)
        arr = arr * (1 - strength * 0.15) + mean_color * (strength * 0.15)

        # Boost saturation slightly (that oversaturated GAN look)
        img_out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        enhancer = ImageEnhance.Color(img_out)
        img_out = enhancer.enhance(1 + strength * 0.2)

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
