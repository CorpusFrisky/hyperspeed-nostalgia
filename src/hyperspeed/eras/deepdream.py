"""DeepDream Era (2015-2019) / Cave Paintings parallel.

Pattern recognition as primal instinct. Forms emerging from noise.
Pareidolia: the mind (human or artificial) finding signal in chaos.

Technical approach:
1. Generate an image from prompt using Stable Diffusion (optional)
2. Apply gradient ascent on CNN activations - we maximize what the network
   "sees" in an image, amplifying detected patterns until they become
   visible hallucinations.

Art historical parallel: Cave painters at Chauvet used wall irregularities
to animate figures (limestone bumps becoming animal haunches). DeepDream
does the same with pixel noise becoming dog faces.
"""

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from hyperspeed.core.artifact_control import ArtifactControl
from hyperspeed.core.image_utils import (
    create_noise_image,
    load_image,
    resize_to_multiple,
)
from hyperspeed.eras.base import EraMetadata, EraPipeline, EraRegistry


# Layer name mappings for InceptionV3
# Shallow layers = edges/textures; deep layers = eyes/faces/objects
INCEPTION_LAYERS = {
    "early": "Conv2d_2b_3x3",  # Low-level features
    "mixed3": "Mixed_5b",  # Mid-level features
    "mixed4": "Mixed_5c",  # Mid-level features
    "mixed5": "Mixed_5d",  # Higher features
    "mixed6": "Mixed_6a",  # High-level features (dog faces start here)
    "mixed7": "Mixed_6b",  # High-level features
    "deep": "Mixed_7a",  # Very high-level (complex objects)
}

DEFAULT_LAYER = "mixed5"


class InceptionFeatureExtractor(nn.Module):
    """Extract features from specific InceptionV3 layers."""

    def __init__(self, layer_name: str = DEFAULT_LAYER):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval()

        self.layer_name = layer_name
        self.features = None

        # Build sequential model up to target layer
        layers = []
        target = INCEPTION_LAYERS.get(layer_name, layer_name)

        for name, module in inception.named_children():
            layers.append(module)
            if name == target:
                break

        self.model = nn.Sequential(*layers)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@EraRegistry.register
class DeepDreamPipeline(EraPipeline):
    """DeepDream: the primal era of AI image artifacts.

    Controls:
        layer: Which network layer to amplify (shallow=textures, deep=objects)
        octaves: Multi-scale processing (more=more fractal quality)
        octave_scale: Scale factor between octaves (1.2-1.5 typical)
        iterations: Gradient ascent steps per octave
        lr: Learning rate (higher=more intense)
        jitter: Random shift to reduce artifacts (helps with tiling)
    """

    metadata: ClassVar[EraMetadata] = EraMetadata(
        name="DeepDream",
        art_historical_parallel="Cave Paintings (35,000-10,000 BCE)",
        time_period="2015-2019",
        description=(
            "Pattern recognition as primal instinct. Forms emerging from noise. "
            "The network amplifies what it 'sees' until hallucinations become visible."
        ),
        characteristic_artifacts=[
            "Fractal dog faces",
            "Eyes everywhere",
            "Psychedelic swirls",
            "Feature amplification",
            "Pareidolia made visible",
        ],
    )

    def __init__(self, model_path: Path | None = None, device: str = "mps"):
        super().__init__(model_path, device)
        self._model: InceptionFeatureExtractor | None = None
        self._current_layer: str | None = None
        self._sd_pipe = None

    def load_model(self, layer: str = DEFAULT_LAYER) -> None:
        """Load InceptionV3 feature extractor."""
        if self._model is None or self._current_layer != layer:
            self._model = InceptionFeatureExtractor(layer).to(self.device)
            self._current_layer = layer
            self._model.eval()

    def _load_sd_pipe(self) -> None:
        """Load Stable Diffusion for optional image generation."""
        if self._sd_pipe is not None:
            return

        from diffusers import StableDiffusionPipeline

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        self._sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,  # float32 more stable on MPS
            use_safetensors=True,
        )
        self._sd_pipe = self._sd_pipe.to(self.device)
        self._sd_pipe.enable_attention_slicing()
        self._sd_pipe.safety_checker = None

    def get_default_params(self) -> dict[str, Any]:
        return {
            "layer": DEFAULT_LAYER,
            "octaves": 4,
            "octave_scale": 1.4,
            "iterations": 20,
            "lr": 0.01,
            "jitter": 32,
        }

    def generate(
        self,
        prompt: str | None = None,
        source_image: Image.Image | None = None,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Generate a DeepDream image.

        Args:
            prompt: Text prompt to generate base image (optional)
            source_image: Image to dream on. If None and no prompt, starts with noise.
            control: Artifact control params
            **era_params: layer, octaves, octave_scale, iterations, lr, jitter,
                         num_inference_steps, guidance_scale

        Returns:
            Dreamed image
        """
        control = control or ArtifactControl()
        params = {**self.get_default_params(), **era_params}

        # Scale parameters by intensity
        params["iterations"] = int(
            control.scale_param(params["iterations"], 5, params["iterations"] * 2)
        )
        params["lr"] = control.scale_param(params["lr"], 0.005, params["lr"] * 2)

        # Load model for specified layer
        self.load_model(params["layer"])

        # Prepare input image
        if source_image is not None:
            # Use provided source image
            source_image = resize_to_multiple(source_image, 8)
        elif prompt is not None:
            # Generate from prompt using Stable Diffusion
            self._load_sd_pipe()

            num_steps = era_params.get("num_inference_steps", 30)
            guidance = era_params.get("guidance_scale", 7.5)
            width = era_params.get("width", 1024)
            height = era_params.get("height", 1024)

            generator = None
            if control.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(control.seed)

            result = self._sd_pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
            )
            source_image = result.images[0]
        else:
            # No prompt, no source - start with noise
            source_image = create_noise_image((1024, 1024), seed=control.seed)

        original = source_image.copy()

        # Set seed for reproducibility
        if control.seed is not None:
            torch.manual_seed(control.seed)
            np.random.seed(control.seed)

        # Run DeepDream
        result = self._dream(
            source_image,
            octaves=params["octaves"],
            octave_scale=params["octave_scale"],
            iterations=params["iterations"],
            lr=params["lr"],
            jitter=params["jitter"],
        )

        # Apply placement mask
        result = control.apply_mask(result, original)

        # Upscale if requested (do this AFTER dreaming for better quality)
        upscale = era_params.get("upscale", 1)
        if upscale > 1:
            new_size = (result.width * upscale, result.height * upscale)
            result = result.resize(new_size, Image.Resampling.LANCZOS)

        return result

    def _dream(
        self,
        image: Image.Image,
        octaves: int,
        octave_scale: float,
        iterations: int,
        lr: float,
        jitter: int,
    ) -> Image.Image:
        """Core DeepDream algorithm with octave processing."""
        # Preprocessing
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Store octave details for later
        octave_details = []
        img = image

        # Calculate octave sizes
        original_size = img.size
        for i in range(octaves - 1):
            new_size = (
                int(original_size[0] / (octave_scale ** (octaves - 1 - i))),
                int(original_size[1] / (octave_scale ** (octaves - 1 - i))),
            )
            octave_details.append(new_size)
        octave_details.append(original_size)

        # Process each octave
        detail = None
        for octave_idx, size in enumerate(octave_details):
            # Resize image
            img = image.resize(size, Image.Resampling.LANCZOS)

            # Add back detail from previous octave
            if detail is not None:
                detail_resized = detail.resize(size, Image.Resampling.LANCZOS)
                img = Image.blend(img, detail_resized, 0.5)

            # Convert to tensor
            tensor = preprocess(img).unsqueeze(0).to(self.device)
            tensor.requires_grad = True

            # Gradient ascent
            for _ in range(iterations):
                tensor = self._dream_step(tensor, lr, jitter)

            # Convert back to image
            img = self._tensor_to_pil(tensor)

            # Calculate detail (what we added at this octave)
            if octave_idx < len(octave_details) - 1:
                # Upscale for next octave comparison
                img_up = img.resize(
                    octave_details[octave_idx + 1], Image.Resampling.LANCZOS
                )
                base_up = image.resize(
                    octave_details[octave_idx + 1], Image.Resampling.LANCZOS
                )
                # Detail is the difference
                detail = Image.blend(base_up, img_up, 1.0)

        return img

    def _dream_step(
        self, tensor: torch.Tensor, lr: float, jitter: int
    ) -> torch.Tensor:
        """Single gradient ascent step."""
        # Random jitter to reduce tiling artifacts
        ox, oy = np.random.randint(-jitter, jitter + 1, 2)

        # Create a copy for gradient computation
        img_shifted = torch.roll(tensor, shifts=(int(ox), int(oy)), dims=(2, 3))
        img_shifted = img_shifted.detach().requires_grad_(True)

        # Forward pass
        features = self._model(img_shifted)

        # Loss is the mean of all activations (we want to maximize)
        loss = features.norm()

        # Backward pass
        loss.backward()

        # Get and normalize gradients
        grad = img_shifted.grad.detach()
        grad = grad / (grad.std() + 1e-8)

        # Undo jitter on gradient
        grad = torch.roll(grad, shifts=(int(-ox), int(-oy)), dims=(2, 3))

        # Gradient ascent (add gradient to increase activations)
        tensor = tensor.detach() + lr * grad
        tensor.requires_grad_(True)

        return tensor

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL Image."""
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        tensor = tensor * std + mean

        # Clamp and convert
        tensor = torch.clamp(tensor, 0, 1)
        arr = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        arr = (arr * 255).astype(np.uint8)

        return Image.fromarray(arr)
