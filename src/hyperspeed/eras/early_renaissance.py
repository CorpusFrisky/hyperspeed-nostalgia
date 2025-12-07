"""Early Renaissance Era (1400-1490) / DALL-E 2 & Stable Diffusion 1.x (2022) parallel.

Reaching for realism but failing in characteristic ways. Hands are famously hard.
Perspective errors. Ambitious but not yet there.

The Early Renaissance saw artists like Masaccio and Mantegna attempting bold new
techniques: foreshortening, unified perspective, naturalistic anatomy. They often
failed in ways that feel eerily familiar to anyone who used DALL-E 2 or SD 1.x
in 2022.

Key works:
- Masaccio's Brancacci Chapel frescoes (1425-1428): ambitious spatial staging
- Mantegna's Dead Christ (c. 1480): extreme foreshortening that doesn't quite work
- Dürer's grid drawings: the "drawing machine" as Renaissance training data

Technical approach:
1. Generate image with Stable Diffusion (or transform source image)
2. Apply characteristic Early Renaissance / 2022 diffusion artifacts:
   - Hand failures (six fingers, fused digits, wrong counts)
   - Foreshortening errors (depth that doesn't recede properly)
   - Perspective contradictions (spatial impossibilities)
   - Proportion shifts (scale inconsistencies across the image)
   - Edge ambiguity (figure-ground integration issues)
"""

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

from hyperspeed.core.artifact_control import ArtifactControl
from hyperspeed.eras.base import EraMetadata, EraPipeline, EraRegistry


@EraRegistry.register
class EarlyRenaissancePipeline(EraPipeline):
    """Early Renaissance Era: The ambitious failures of 2022.

    This pipeline generates images with the characteristic "reaching for
    realism but not quite there" quality of early diffusion models, mapped
    onto the spatial and anatomical struggles of Early Renaissance painters.

    Controls:
        hand_failure: Intensity of hand anatomical errors (0-1)
        foreshortening_error: Depth representation failures (0-1)
        perspective_error: Spatial contradiction strength (0-1)
        proportion_shift: Scale variation across composition (0-1)
        edge_ambiguity: Figure-ground integration issues (0-1)
    """

    metadata: ClassVar[EraMetadata] = EraMetadata(
        name="Early Renaissance",
        art_historical_parallel="Early Renaissance (1400-1490)",
        time_period="2022",
        description=(
            "Reaching for realism but failing in characteristic ways. "
            "Hands are famously hard. Perspective errors. Ambitious but not yet there."
        ),
        characteristic_artifacts=[
            "Hand failures (six fingers, fused digits)",
            "Foreshortening errors",
            "Perspective contradictions",
            "Anatomical proportion shifts",
            "Edge ambiguity",
        ],
    )

    def __init__(self, model_path: Path | None = None, device: str = "mps"):
        super().__init__(model_path, device)
        self._pipe = None
        self._img2img_pipe = None

    def load_model(self) -> None:
        """Load Stable Diffusion for generation."""
        if self._pipe is not None:
            return

        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        self._pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        self._pipe = self._pipe.to(self.device)
        self._pipe.enable_attention_slicing()
        self._pipe.safety_checker = None

        # Create img2img pipeline sharing the same components
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

    def get_default_params(self) -> dict[str, Any]:
        return {
            "hand_failure": 0.7,
            "foreshortening_error": 0.6,
            "perspective_error": 0.5,
            "proportion_shift": 0.4,
            "edge_ambiguity": 0.4,
            "wooden_face": 0.6,  # Early Renaissance face stiffness
            "inference_steps": 20,
            "guidance_scale": 7.5,
            "img2img_strength": 0.6,
        }

    def generate(
        self,
        prompt: str | None = None,
        source_image: Image.Image | None = None,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Generate an image with Early Renaissance / 2022 diffusion artifacts.

        Args:
            prompt: Text prompt for generation. Required unless source_image provided.
            source_image: Optional source image (if provided with prompt, uses img2img)
            control: Artifact control parameters
            **era_params: hand_failure, foreshortening_error, perspective_error, etc.

        Returns:
            Generated image with Early Renaissance artifacts
        """
        control = control or ArtifactControl()
        params = {**self.get_default_params(), **era_params}

        # Generation parameters
        num_steps = params.get("inference_steps", 20)
        guidance = params.get("guidance_scale", 7.5)

        # Determine generation mode
        if source_image is None:
            # txt2img mode
            if prompt is None:
                raise ValueError("Early Renaissance pipeline requires a prompt or source image")

            self.ensure_loaded()

            generator = None
            if control.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(control.seed)
                np.random.seed(control.seed)

            width = era_params.get("width", 1024)
            height = era_params.get("height", 1024)

            # Ensure dimensions are multiples of 8
            width = (width // 8) * 8
            height = (height // 8) * 8

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

        hand_failure = get_effect_strength("hand_failure")
        foreshortening_error = get_effect_strength("foreshortening_error")
        perspective_error = get_effect_strength("perspective_error")
        proportion_shift = get_effect_strength("proportion_shift")
        edge_ambiguity = get_effect_strength("edge_ambiguity")
        wooden_face = get_effect_strength("wooden_face")

        # Apply artifacts in sequence
        # Faces and hands get different treatments (per other Claude's insight)
        img = self._apply_perspective_error(img, perspective_error)
        img = self._apply_foreshortening_error(img, foreshortening_error)
        img = self._apply_proportion_shift(img, proportion_shift)
        img = self._apply_wooden_face(img, wooden_face)  # Stiff, mannequin-like faces
        img = self._apply_hand_failure(img, hand_failure)  # Dissolution, finger ambiguity
        img = self._apply_edge_ambiguity(img, edge_ambiguity)

        # Final blend with original based on placement
        img = control.apply_mask(original, img)

        # Upscale if requested
        upscale = era_params.get("upscale", 1)
        if upscale > 1:
            new_size = (img.width * upscale, img.height * upscale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _apply_wooden_face(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply Early Renaissance / DALL-E 2 face artifacts.

        Unlike Byzantine/Gothic (smooth, vacant, spiritually uncanny), Early
        Renaissance faces are differently wrong:
        - Stiff, "wooden" - attempting naturalism but landing mannequin-like
        - Eyes present and engaged but don't quite track together
        - Harder edges - tempera-on-panel crispness, not sfumato
        - Proportions almost correct but subtly off
        - Expressions trying to convey emotion but feeling performed

        DALL-E 2 face problems were different from StyleGAN:
        - StyleGAN: too smooth, too perfect, uncanny through excess polish
        - DALL-E 2: parts not quite aligning, asymmetries that feel wrong
          rather than natural, flatness despite attempted depth

        The face should feel like it's trying too hard, not dissolving into void.

        Technique:
        - Detect face regions (skin tone in upper image + oval shape heuristic)
        - Sharpen edges (tempera crispness, opposite of sfumato)
        - Add subtle asymmetric displacement (eyes that don't track)
        - Increase local contrast (wooden, not smooth)
        - Flatten midtones while preserving highlights (performed expression)
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        result = arr.copy()

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Detect potential face regions
        # Skin tone heuristic
        skin_mask = (
            (r > 80) & (g > 50) & (b > 30) &
            (r > g) & (g > b) &
            (r - b > 10) & (r - b < 100)
        ).astype(np.float32)

        # Face regions are typically in upper-middle of image
        # Create position bias toward face location
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        y_norm = y_coords / h
        x_norm = x_coords / w

        # Faces typically in upper third, center horizontal
        face_position_bias = np.exp(-((y_norm - 0.3)**2 / 0.15 + (x_norm - 0.5)**2 / 0.2))

        # Combine skin detection with position bias
        face_mask = skin_mask * face_position_bias
        face_mask = ndimage.gaussian_filter(face_mask, sigma=10)
        if face_mask.max() > 0:
            face_mask = face_mask / face_mask.max()

        # === EFFECT 1: Edge sharpening (tempera crispness) ===
        # Opposite of the smoothing used for Byzantine/Gothic
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        laplacian = ndimage.laplace(gray)

        # Sharpen by adding back high-frequency detail
        for c in range(3):
            channel_laplacian = ndimage.laplace(arr[:, :, c])
            sharpening = channel_laplacian * strength * 0.3
            # Apply only in face regions
            result[:, :, c] = result[:, :, c] - sharpening * face_mask

        # === EFFECT 2: Asymmetric eye displacement ===
        # Eyes that don't quite track together - the DALL-E 2 tell
        # Detect eye region (upper part of face mask)
        eye_region = face_mask * (y_norm < 0.4) * (y_norm > 0.15)
        eye_region = ndimage.gaussian_filter(eye_region, sigma=5)

        # Create asymmetric displacement - left and right eyes shift differently
        left_side = (x_norm < 0.5).astype(np.float32)
        right_side = (x_norm >= 0.5).astype(np.float32)

        # Vertical displacement - one eye slightly higher
        dy_left = strength * 3 * eye_region * left_side
        dy_right = -strength * 2 * eye_region * right_side
        dy = dy_left + dy_right

        # Slight horizontal convergence error
        dx_left = strength * 2 * eye_region * left_side
        dx_right = -strength * 1.5 * eye_region * right_side
        dx = dx_left + dx_right

        # Apply displacement
        new_y = np.clip(y_coords + dy, 0, h - 1)
        new_x = np.clip(x_coords + dx, 0, w - 1)

        for c in range(3):
            displaced = ndimage.map_coordinates(
                result[:, :, c],
                [new_y, new_x],
                order=1,
                mode='reflect'
            )
            # Blend displacement only in eye regions
            blend = eye_region * strength * 0.7
            result[:, :, c] = result[:, :, c] * (1 - blend) + displaced * blend

        # === EFFECT 3: Increase local contrast (wooden, not smooth) ===
        # This creates the "mannequin" quality - too crisp, too defined
        for c in range(3):
            channel = result[:, :, c]
            # Local mean
            local_mean = ndimage.uniform_filter(channel, size=15)
            # Increase deviation from local mean
            deviation = channel - local_mean
            enhanced = local_mean + deviation * (1 + strength * 0.4)
            # Apply in face regions
            face_blend = face_mask * strength * 0.5
            result[:, :, c] = channel * (1 - face_blend) + enhanced * face_blend

        # === EFFECT 4: Flatten midtones (performed expression) ===
        # Compress the middle tonal range while preserving highlights/shadows
        # This creates the "trying too hard" quality of forced expressions
        for c in range(3):
            channel = result[:, :, c]
            # Normalize to 0-1
            c_norm = channel / 255.0

            # S-curve that flattens midtones
            # Midtones (0.3-0.7) get compressed, extremes preserved
            midtone_mask = np.exp(-((c_norm - 0.5)**2 / 0.1))
            compression = 1 - midtone_mask * strength * 0.2

            # Apply compression toward middle gray
            flattened = 0.5 + (c_norm - 0.5) * compression
            flattened = flattened * 255.0

            # Blend in face regions
            face_blend = face_mask * strength * 0.4
            result[:, :, c] = channel * (1 - face_blend) + flattened * face_blend

        # === EFFECT 5: Subtle proportion distortion ===
        # Features almost correct but subtly off
        # Nose slightly too long, eyes slightly too close, etc.
        nose_region = face_mask * (y_norm > 0.25) * (y_norm < 0.5) * (np.abs(x_norm - 0.5) < 0.1)
        nose_region = ndimage.gaussian_filter(nose_region, sigma=3)

        # Elongate nose region slightly
        nose_stretch = nose_region * strength * 4
        new_y_nose = np.clip(y_coords - nose_stretch, 0, h - 1)

        for c in range(3):
            stretched = ndimage.map_coordinates(
                result[:, :, c],
                [new_y_nose, x_coords.astype(np.float32)],
                order=1,
                mode='reflect'
            )
            blend = nose_region * strength * 0.5
            result[:, :, c] = result[:, :, c] * (1 - blend) + stretched * blend

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_hand_failure(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply hand failure artifacts: extra digits, fused fingers, wrong anatomy.

        The iconic 2022 diffusion artifact. Early Renaissance painters also
        struggled with hands, often hiding them in drapery or depicting
        them awkwardly.

        Technique:
        - Detect potential hand regions (skin tone + edge density)
        - Apply localized warping to create finger multiplication
        - Blur boundaries between fingers to create fusion effect
        - Add edge artifacts to suggest extra digit outlines
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        result = arr.copy()

        # Create a mask for potential hand/finger regions
        # Use skin tone detection (rough approximation)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Skin tone heuristic: R > G > B, with certain ratios
        skin_mask = (
            (r > 60) & (g > 40) & (b > 20) &
            (r > g) & (g > b) &
            (np.abs(r - g) > 15) &
            (r - b > 15)
        ).astype(np.float32)

        # Find high-edge-density regions (fingers have many edges)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        edges_x = np.abs(ndimage.sobel(gray, axis=1))
        edges_y = np.abs(ndimage.sobel(gray, axis=0))
        edge_density = edges_x + edges_y

        # Normalize edge density
        if edge_density.max() > 0:
            edge_density = edge_density / edge_density.max()

        # Hand regions: skin tone AND high edge density
        hand_mask = skin_mask * edge_density
        hand_mask = ndimage.gaussian_filter(hand_mask, sigma=5)

        # Create displacement field for finger warping
        # This creates the "extra finger" effect by duplicating/shifting regions
        freq_x = 3 + np.random.random() * 2
        freq_y = 4 + np.random.random() * 2

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Sinusoidal displacement creates finger-like repetition
        displacement_x = (
            np.sin(y_coords / h * np.pi * freq_y) *
            np.cos(x_coords / w * np.pi * freq_x) *
            strength * 15
        )
        displacement_y = (
            np.cos(y_coords / h * np.pi * freq_y * 0.7) *
            np.sin(x_coords / w * np.pi * freq_x * 0.8) *
            strength * 10
        )

        # Apply displacement only in hand regions
        displacement_x = displacement_x * hand_mask
        displacement_y = displacement_y * hand_mask

        # Calculate new coordinates
        new_x = np.clip(x_coords + displacement_x, 0, w - 1)
        new_y = np.clip(y_coords + displacement_y, 0, h - 1)

        # Apply warping to each channel
        for c in range(3):
            result[:, :, c] = ndimage.map_coordinates(
                arr[:, :, c],
                [new_y, new_x],
                order=1,
                mode='reflect'
            )

        # Add finger fusion effect: blur boundaries in hand regions
        blurred = ndimage.gaussian_filter(result, sigma=[3, 3, 0])
        fusion_mask = hand_mask[:, :, np.newaxis] * strength * 0.3
        result = result * (1 - fusion_mask) + blurred * fusion_mask

        # Add extra edge artifacts to suggest additional digit outlines
        edge_artifacts = np.zeros_like(arr)
        edge_artifacts[:, :, 0] = edges_x * 50  # Slight reddish edges
        edge_artifacts[:, :, 1] = edges_y * 30
        edge_artifacts[:, :, 2] = (edges_x + edges_y) * 20

        # Offset and add edges to create "ghost digit" effect
        offset = int(5 * strength)
        if offset > 0:
            shifted_edges = np.roll(edge_artifacts, offset, axis=1)
            edge_mask = hand_mask[:, :, np.newaxis] * strength * 0.15
            result = result + shifted_edges * edge_mask

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_foreshortening_error(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply foreshortening errors: depth that doesn't recede properly.

        Mantegna's Dead Christ is the canonical example: ambitious attempt
        at extreme foreshortening, but the feet are too small for the
        perspective to work. 2022 diffusion models had similar issues with
        limbs extending toward/away from the camera.

        Technique:
        - Create approximate depth map from brightness/position
        - Apply inconsistent scaling to different depth regions
        - Flatten areas that should recede
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Create approximate depth map
        # Assume: brighter = closer, top = farther (like sky), bottom = closer
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        gray_norm = gray / 255.0

        # Position-based depth (top = far, bottom = near)
        y_coords = np.arange(h)[:, np.newaxis] / h
        position_depth = y_coords * np.ones((h, w))

        # Combine brightness and position for depth estimate
        depth = 0.6 * gray_norm + 0.4 * position_depth
        depth = ndimage.gaussian_filter(depth, sigma=20)

        # Create scaling field that contradicts depth
        # Things that should be small (far) will be enlarged
        # Things that should be large (near) will be shrunk
        scale_contradiction = 1.0 + (depth - 0.5) * strength * 0.3

        # Apply scaling through displacement
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Displacement toward/away from center based on contradictory scale
        dy = (y_coords - center_y) * (scale_contradiction - 1)
        dx = (x_coords - center_x) * (scale_contradiction - 1)

        new_y = np.clip(y_coords - dy, 0, h - 1)
        new_x = np.clip(x_coords - dx, 0, w - 1)

        result = np.zeros_like(arr)
        for c in range(3):
            result[:, :, c] = ndimage.map_coordinates(
                arr[:, :, c],
                [new_y, new_x],
                order=1,
                mode='reflect'
            )

        # Add flattening effect to distant regions
        flat_mask = (1 - depth) * strength * 0.4
        flat_mask = flat_mask[:, :, np.newaxis]

        # Reduce contrast in "distant" regions (flattening)
        mean_color = np.mean(result, axis=(0, 1), keepdims=True)
        flattened = result * (1 - flat_mask * 0.5) + mean_color * flat_mask * 0.5

        return Image.fromarray(np.clip(flattened, 0, 255).astype(np.uint8))

    def _apply_perspective_error(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply perspective contradictions: spatial impossibilities.

        Masaccio's Brancacci Chapel shows early attempts at unified
        perspective that don't quite work. Multiple vanishing points,
        spatial staging that contradicts itself. 2022 diffusion models
        produced similar impossibilities.

        Technique:
        - Create mesh-based warping with multiple vanishing points
        - Apply different perspective skews to different regions
        - Horizontal lines that should converge don't quite meet
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Create multiple "vanishing point" distortion fields
        # Each region of the image will have slightly different perspective

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # First vanishing point (left of center)
        vp1_x, vp1_y = w * 0.3, h * 0.4
        dist1 = np.sqrt((x_coords - vp1_x)**2 + (y_coords - vp1_y)**2)
        dist1_norm = dist1 / np.sqrt(w**2 + h**2)

        # Second vanishing point (right of center, different height)
        vp2_x, vp2_y = w * 0.7, h * 0.35
        dist2 = np.sqrt((x_coords - vp2_x)**2 + (y_coords - vp2_y)**2)
        dist2_norm = dist2 / np.sqrt(w**2 + h**2)

        # Create conflicting perspective warps
        # Left side pulls toward vp1, right side toward vp2
        left_weight = 1 - (x_coords / w)
        right_weight = x_coords / w

        # Displacement toward each vanishing point
        dx1 = (vp1_x - x_coords) * dist1_norm * strength * 0.1 * left_weight
        dy1 = (vp1_y - y_coords) * dist1_norm * strength * 0.1 * left_weight

        dx2 = (vp2_x - x_coords) * dist2_norm * strength * 0.1 * right_weight
        dy2 = (vp2_y - y_coords) * dist2_norm * strength * 0.1 * right_weight

        # Combine displacements (they conflict!)
        dx = dx1 + dx2
        dy = dy1 + dy2

        # Add some random waviness for extra wrongness
        wave_x = np.sin(y_coords / h * np.pi * 3) * strength * 5
        wave_y = np.cos(x_coords / w * np.pi * 2) * strength * 3
        dx = dx + wave_x
        dy = dy + wave_y

        new_x = np.clip(x_coords + dx, 0, w - 1)
        new_y = np.clip(y_coords + dy, 0, h - 1)

        result = np.zeros_like(arr)
        for c in range(3):
            result[:, :, c] = ndimage.map_coordinates(
                arr[:, :, c],
                [new_y, new_x],
                order=1,
                mode='reflect'
            )

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_proportion_shift(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply proportion shifts: scale that varies across the image.

        Early Renaissance artists were still transitioning from hierarchical
        sizing (important figures = bigger) to naturalistic proportion.
        2022 diffusion models similarly produced heads too small, limbs
        too long, proportions that shifted across the composition.

        Technique:
        - Apply gradient-based scaling across the image
        - Subtle vertical stretch in some areas, compression in others
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Create proportion shift field
        # Top of image: compress (heads too small)
        # Middle: stretch (torsos elongated)
        # Bottom: slight compression

        # Vertical position normalized
        y_norm = y_coords / h

        # Proportion factor varies with vertical position
        # This creates subtle "wrongness" in body proportions
        proportion_factor = (
            1.0 +
            np.sin(y_norm * np.pi * 2) * strength * 0.15 +  # Wave pattern
            (0.5 - y_norm) * strength * 0.1  # Top smaller, bottom larger
        )

        # Horizontal variation for asymmetry
        x_norm = x_coords / w
        h_variation = np.sin(x_norm * np.pi) * strength * 0.05

        # Apply as vertical displacement
        center_y = h // 2
        dy = (y_coords - center_y) * (proportion_factor - 1 + h_variation)

        new_y = np.clip(y_coords + dy, 0, h - 1)
        new_x = x_coords.astype(np.float32)

        result = np.zeros_like(arr)
        for c in range(3):
            result[:, :, c] = ndimage.map_coordinates(
                arr[:, :, c],
                [new_y, new_x],
                order=1,
                mode='reflect'
            )

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_edge_ambiguity(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply edge ambiguity: inconsistent figure-ground integration.

        Early Renaissance figures often feel "pasted on" their backgrounds,
        not fully integrated into the space. 2022 diffusion models had
        similar issues: the "cutout" quality where subjects don't quite
        belong to their environment.

        Technique:
        - Detect edges between figure and background
        - Apply inconsistent edge treatment (some sharp, some soft)
        - Create subtle haloing or disconnection at boundaries
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Detect edges
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        edges = ndimage.sobel(gray)
        edges = np.abs(edges)
        if edges.max() > 0:
            edges = edges / edges.max()

        # Create edge mask
        edge_mask = (edges > 0.1).astype(np.float32)
        edge_mask = ndimage.gaussian_filter(edge_mask, sigma=2)

        # Create inconsistent edge treatment
        # Some edges will be sharpened, others blurred
        pattern = np.random.random((h, w))
        pattern = ndimage.gaussian_filter(pattern, sigma=30)
        sharp_mask = (pattern > 0.5).astype(np.float32)

        # Sharpen some edges
        sharpened = arr.copy()
        for c in range(3):
            laplacian = ndimage.laplace(arr[:, :, c])
            sharpened[:, :, c] = arr[:, :, c] - laplacian * 0.5

        # Blur other edges
        blurred = ndimage.gaussian_filter(arr, sigma=[2, 2, 0])

        # Combine based on pattern
        edge_region = edge_mask[:, :, np.newaxis] * strength
        result = arr.copy()

        # Apply sharpening to some edge regions
        sharp_region = edge_region * sharp_mask[:, :, np.newaxis]
        result = result * (1 - sharp_region) + sharpened * sharp_region

        # Apply blur to other edge regions
        blur_region = edge_region * (1 - sharp_mask[:, :, np.newaxis])
        result = result * (1 - blur_region) + blurred * blur_region

        # Add subtle halo effect at edges (the "pasted on" look)
        halo = ndimage.gaussian_filter(arr, sigma=[5, 5, 0])
        halo_mask = edge_mask[:, :, np.newaxis] * strength * 0.2

        # Lighten edges slightly for halo
        halo_lightened = halo * 1.1 + 20
        result = result * (1 - halo_mask) + halo_lightened * halo_mask

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
