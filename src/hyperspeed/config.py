"""Configuration models for HYPERSPEED NOSTALGIA."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for model paths and settings."""

    models_dir: Path = Field(
        default=Path("models"),
        description="Directory containing downloaded model weights",
    )
    device: str = Field(
        default="mps",
        description="Torch device: 'mps' (Apple Silicon), 'cuda', or 'cpu'",
    )
    dtype: str = Field(
        default="float16",
        description="Model dtype: 'float16' or 'float32'",
    )


class OutputConfig(BaseModel):
    """Configuration for output settings."""

    output_dir: Path = Field(
        default=Path("outputs"),
        description="Directory for generated images",
    )
    format: str = Field(
        default="png",
        description="Output image format: 'png', 'jpg', 'webp'",
    )
    quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="JPEG/WebP quality (1-100)",
    )


class GenerationConfig(BaseModel):
    """Configuration for a single generation run."""

    era: str = Field(
        description="Era to use for generation (e.g., 'deepdream', 'early_diffusion')",
    )
    prompt: str | None = Field(
        default=None,
        description="Text prompt (optional, not all eras use it)",
    )
    source_image: Path | None = Field(
        default=None,
        description="Path to source image to transform",
    )
    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Artifact intensity (0.0 = subtle, 1.0 = maximum)",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    era_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Era-specific parameters",
    )


class PresetConfig(BaseModel):
    """A saved preset combining era and parameters."""

    name: str
    description: str
    era: str
    intensity: float = 0.5
    era_params: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    presets_dir: Path = Field(
        default=Path("presets"),
        description="Directory containing preset YAML files",
    )
