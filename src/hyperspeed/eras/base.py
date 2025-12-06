"""Base class for era-specific artifact pipelines.

Each era (DeepDream, Early Diffusion, StyleGAN, etc.) implements this interface.
The key principle: artifacts should feel intentional (craft, not accident).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from hyperspeed.core.artifact_control import ArtifactControl


@dataclass
class EraMetadata:
    """Metadata describing an era for display and documentation."""

    name: str
    art_historical_parallel: str
    time_period: str
    description: str
    characteristic_artifacts: list[str]


class EraPipeline(ABC):
    """Abstract base class for era-specific artifact generation.

    Each era pipeline must:
    1. Define its metadata (name, art historical parallel, etc.)
    2. Implement generate() to produce images with era-specific artifacts
    3. Provide era-specific parameters that give artists control

    The goal is CRAFT, not ACCIDENT. Artists should be able to control:
    - Which artifacts appear
    - Where they appear
    - How intensely they manifest
    """

    metadata: ClassVar[EraMetadata]

    def __init__(self, model_path: Path | None = None, device: str = "mps"):
        """Initialize the pipeline.

        Args:
            model_path: Path to model weights (if needed)
            device: Torch device ("mps" for Apple Silicon, "cuda", "cpu")
        """
        self.model_path = model_path
        self.device = device
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model weights. Called lazily on first generate()."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str | None = None,
        source_image: Image.Image | None = None,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Generate an image with era-specific artifacts.

        Args:
            prompt: Text prompt (optional, not all eras use it)
            source_image: Input image to transform (optional)
            control: Artifact control parameters (intensity, placement, etc.)
            **era_params: Era-specific parameters (e.g., octave_count for DeepDream)

        Returns:
            Generated image with intentional artifacts
        """
        pass

    @abstractmethod
    def get_default_params(self) -> dict[str, Any]:
        """Return default era-specific parameters with descriptions.

        Returns:
            Dict mapping parameter names to their default values.
            Each era defines its own meaningful parameters.
        """
        pass

    def ensure_loaded(self) -> None:
        """Ensure the model is loaded before generation."""
        if self._model is None:
            self.load_model()


class EraRegistry:
    """Registry of available era pipelines."""

    _eras: ClassVar[dict[str, type[EraPipeline]]] = {}

    @classmethod
    def register(cls, era_class: type[EraPipeline]) -> type[EraPipeline]:
        """Decorator to register an era pipeline."""
        name = era_class.metadata.name.lower().replace(" ", "_")
        cls._eras[name] = era_class
        return era_class

    @classmethod
    def get(cls, name: str) -> type[EraPipeline]:
        """Get an era pipeline class by name."""
        normalized = name.lower().replace(" ", "_").replace("-", "_")
        if normalized not in cls._eras:
            available = ", ".join(sorted(cls._eras.keys()))
            raise ValueError(f"Unknown era '{name}'. Available: {available}")
        return cls._eras[normalized]

    @classmethod
    def list_eras(cls) -> list[EraMetadata]:
        """List all registered eras with their metadata."""
        return [era.metadata for era in cls._eras.values()]

    @classmethod
    def available(cls) -> list[str]:
        """List available era names."""
        return sorted(cls._eras.keys())
