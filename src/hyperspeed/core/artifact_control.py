"""Artifact control: the heart of intentional artifact generation.

The key principle from PRINCIPLES.md:
    "Artifacts should feel intentional. The 'wrongness' is a brushstroke, not a bug."

This module provides unified control over:
- Intensity: How strongly artifacts manifest
- Placement: Where artifacts appear (global, edges, faces, hands, etc.)
- Blending: How artifacts combine with the base image
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from PIL import Image


class ArtifactPlacement(Enum):
    """Where artifacts should appear in the image."""

    GLOBAL = "global"  # Everywhere
    EDGES = "edges"  # Along edges and boundaries
    FACES = "faces"  # On detected faces
    HANDS = "hands"  # On detected hands (for Renaissance era)
    BACKGROUND = "background"  # Background only, preserve foreground
    FOREGROUND = "foreground"  # Foreground only
    CUSTOM = "custom"  # User-provided mask


class BlendMode(Enum):
    """How to blend artifacts with the base image."""

    NORMAL = "normal"  # Direct replacement
    MULTIPLY = "multiply"  # Darken
    SCREEN = "screen"  # Lighten
    OVERLAY = "overlay"  # Contrast
    SOFT_LIGHT = "soft_light"  # Subtle blend


@dataclass
class ArtifactControl:
    """Unified control parameters for artifact generation.

    This is passed to every era pipeline to give artists consistent
    control over how artifacts manifest, regardless of era.
    """

    intensity: float = 0.5
    """Overall artifact strength (0.0 = subtle, 1.0 = maximum)."""

    placement: ArtifactPlacement = ArtifactPlacement.GLOBAL
    """Where artifacts should appear."""

    blend_mode: BlendMode = BlendMode.NORMAL
    """How to blend artifacts with the original."""

    mask: np.ndarray | None = None
    """Custom mask for CUSTOM placement (same size as image, 0-1 float)."""

    preserve_composition: bool = True
    """Try to preserve overall image composition while adding artifacts."""

    seed: int | None = None
    """Random seed for reproducible artifact generation."""

    def __post_init__(self):
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"Intensity must be 0.0-1.0, got {self.intensity}")
        if self.placement == ArtifactPlacement.CUSTOM and self.mask is None:
            raise ValueError("CUSTOM placement requires a mask")

    def scale_param(self, value: float, min_val: float, max_val: float) -> float:
        """Scale a parameter by intensity.

        Useful for era pipelines to convert intensity (0-1) into
        era-specific parameter ranges.

        Args:
            value: Base value
            min_val: Value at intensity=0
            max_val: Value at intensity=1

        Returns:
            Interpolated value based on intensity
        """
        return min_val + (max_val - min_val) * self.intensity

    def apply_mask(self, artifact_image: Image.Image, original: Image.Image) -> Image.Image:
        """Apply placement mask to blend artifacts with original.

        Args:
            artifact_image: Image with full artifacts applied
            original: Original image before artifacts

        Returns:
            Blended image respecting placement settings
        """
        if self.placement == ArtifactPlacement.GLOBAL and self.blend_mode == BlendMode.NORMAL:
            return artifact_image

        artifact_arr = np.array(artifact_image, dtype=np.float32) / 255.0
        original_arr = np.array(original, dtype=np.float32) / 255.0

        mask = self._get_mask(original)

        blended = self._blend(artifact_arr, original_arr, mask)

        return Image.fromarray((blended * 255).astype(np.uint8))

    def _get_mask(self, image: Image.Image) -> np.ndarray:
        """Get the mask for the current placement setting."""
        h, w = image.size[1], image.size[0]

        if self.placement == ArtifactPlacement.GLOBAL:
            return np.ones((h, w, 1), dtype=np.float32)

        if self.placement == ArtifactPlacement.CUSTOM:
            if self.mask is None:
                raise ValueError("Custom placement requires a mask")
            mask = self.mask
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]
            return mask.astype(np.float32)

        if self.placement == ArtifactPlacement.EDGES:
            return self._detect_edges(image)

        # TODO: Implement face/hand detection for those placements
        # For now, fall back to global
        return np.ones((h, w, 1), dtype=np.float32)

    def _detect_edges(self, image: Image.Image) -> np.ndarray:
        """Simple edge detection for EDGES placement."""
        from PIL import ImageFilter

        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_arr = np.array(edges, dtype=np.float32) / 255.0
        # Blur slightly for softer transitions
        edges_blurred = np.array(
            Image.fromarray((edges_arr * 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(radius=2)
            ),
            dtype=np.float32,
        ) / 255.0
        return edges_blurred[:, :, np.newaxis]

    def _blend(
        self, artifact: np.ndarray, original: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Apply blend mode and mask."""
        if self.blend_mode == BlendMode.NORMAL:
            blended = artifact
        elif self.blend_mode == BlendMode.MULTIPLY:
            blended = artifact * original
        elif self.blend_mode == BlendMode.SCREEN:
            blended = 1 - (1 - artifact) * (1 - original)
        elif self.blend_mode == BlendMode.OVERLAY:
            low = 2 * artifact * original
            high = 1 - 2 * (1 - artifact) * (1 - original)
            blended = np.where(original < 0.5, low, high)
        elif self.blend_mode == BlendMode.SOFT_LIGHT:
            blended = (1 - 2 * artifact) * original**2 + 2 * artifact * original
        else:
            blended = artifact

        # Apply mask: where mask=1, use blended; where mask=0, use original
        result = mask * blended + (1 - mask) * original
        return np.clip(result, 0, 1)
