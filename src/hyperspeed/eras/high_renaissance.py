"""High Renaissance Era (1490-1527) / Midjourney v4 (Late 2022 - Early 2023) parallel.

Technical ambition realized, but with characteristic tells. Leonardo's sfumato
becomes Midjourney's specific blue-orange lighting. Michelangelo's terribilita
becomes that "everything is epic" over-dramatization. The ambition is real,
but the signature is unmistakable.

The High Renaissance saw Leonardo, Michelangelo, and Raphael achieve technical
mastery. But Midjourney v4, despite its quality, developed recognizable tells:
- Blue-orange color cast (teal shadows, orange highlights)
- Over-dramatized lighting (rim light everything, volumetric rays)
- Hyper-saturation (colors beyond naturalism)
- Epic blur (shallow DOF where it shouldn't exist)
- Everything rendered as if it were a movie poster

Key works:
- Leonardo's Virgin of the Rocks (1483-1486): sfumato becomes blue-orange gradient
- Michelangelo's Sistine Chapel (1508-1512): terribilita becomes HDR muscles
- Raphael's School of Athens (1509-1511): composition becomes dead-centering

Technical approach:
1. Generate image with SDXL via Replicate API (fast) or local (fallback)
2. Apply characteristic Midjourney v4 tells locally:
   - Blue-orange color cast (THE signature)
   - Over-dramatized lighting (rim light, volumetric rays)
   - Hyper-saturation (pushed beyond natural)
   - Epic blur (shallow DOF everywhere)
   - Textural sharpening (every pore visible)
   - Compositional centering (pull subjects center)
   - Warm halo (glow around edges)

Environment:
- Set REPLICATE_API_TOKEN to use Replicate for fast generation
- Falls back to local SDXL if token not set or API fails
"""

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

from hyperspeed.core.artifact_control import ArtifactControl
from hyperspeed.core.replicate_utils import (
    generate_via_replicate,
    batch_generate_replicate,
)
from hyperspeed.eras.base import EraMetadata, EraPipeline, EraRegistry


@EraRegistry.register
class HighRenaissancePipeline(EraPipeline):
    """High Renaissance Era: The epic tells of Midjourney v4.

    This pipeline generates images with the characteristic "everything is
    a movie poster" quality of Midjourney v4, mapped onto the technical
    mastery (and signature style) of High Renaissance masters.

    Unlike Early Renaissance (destabilization) or International Gothic
    (over-polishing), High Renaissance is about over-dramatization.
    Every image feels important. The tell isn't failure; it's trying too hard.

    Controls:
        blue_orange_cast: THE signature MJ v4 color grading (0-1)
        overdramatized_lighting: Rim light, volumetric rays (0-1)
        hypersaturation: Push colors beyond natural (0-1)
        epic_blur: Shallow DOF everywhere (0-1)
        textural_sharpening: Every pore visible (0-1)
        compositional_centering: Pull subjects to center (0-1)
        warm_halo: Glow around edges (0-1)
    """

    metadata: ClassVar[EraMetadata] = EraMetadata(
        name="High Renaissance",
        art_historical_parallel="High Renaissance (1490-1527)",
        time_period="Late 2022 - Early 2023",
        description=(
            "Technical ambition realized, but with characteristic tells. "
            "Everything is epic. Every image is a movie poster. "
            "The tell isn't failure; it's the relentless pursuit of dramatic."
        ),
        characteristic_artifacts=[
            "Blue-orange color cast (teal shadows, orange highlights)",
            "Over-dramatized lighting (rim light, volumetric rays)",
            "Hyper-saturation (colors beyond naturalism)",
            "Epic blur (shallow DOF everywhere)",
            "Textural over-sharpening",
            "Compositional centering",
            "Warm glow halo",
        ],
    )

    def __init__(self, model_path: Path | None = None, device: str = "mps"):
        super().__init__(model_path, device)
        self._pipe = None
        self._img2img_pipe = None

    def load_model(self) -> None:
        """Load SDXL for generation - better faces, higher quality."""
        if self._pipe is not None:
            return

        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        # Use SDXL for much better face generation
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
            "blue_orange_cast": 0.7,       # THE signature
            "overdramatized_lighting": 0.6,
            "hypersaturation": 0.6,
            "epic_blur": 0.5,
            "textural_sharpening": 0.5,
            "compositional_centering": 0.3,  # Keep subtle
            "warm_halo": 0.5,
            "inference_steps": 30,           # Higher quality for "epic"
            "guidance_scale": 10.0,          # Higher CFG for clarity
            "img2img_strength": 0.6,
        }

    def generate(
        self,
        prompt: str | None = None,
        source_image: Image.Image | None = None,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Generate an image with High Renaissance / Midjourney v4 tells.

        Args:
            prompt: Text prompt for generation. Required unless source_image provided.
            source_image: Optional source image (if provided with prompt, uses img2img)
            control: Artifact control parameters
            **era_params: blue_orange_cast, overdramatized_lighting, etc.
                         use_local: Force local generation (skip Replicate)

        Returns:
            Generated image with High Renaissance / MJ v4 tells
        """
        control = control or ArtifactControl()
        params = {**self.get_default_params(), **era_params}

        # Generation parameters
        num_steps = params.get("inference_steps", 30)
        guidance = params.get("guidance_scale", 10.0)
        use_local = era_params.get("use_local", False)

        # Determine generation mode
        if source_image is None:
            # txt2img mode
            if prompt is None:
                raise ValueError("High Renaissance pipeline requires a prompt or source image")

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

        blue_orange_cast = get_effect_strength("blue_orange_cast")
        overdramatized_lighting = get_effect_strength("overdramatized_lighting")
        hypersaturation = get_effect_strength("hypersaturation")
        epic_blur = get_effect_strength("epic_blur")
        textural_sharpening = get_effect_strength("textural_sharpening")
        compositional_centering = get_effect_strength("compositional_centering")
        warm_halo = get_effect_strength("warm_halo")

        # Apply MJ v4 tells in sequence
        # Order matters: color grading first, then lighting, then blur/sharpening
        img = self._apply_blue_orange_cast(img, blue_orange_cast)
        img = self._apply_hypersaturation(img, hypersaturation)
        img = self._apply_overdramatized_lighting(img, overdramatized_lighting)
        img = self._apply_warm_halo(img, warm_halo)
        img = self._apply_epic_blur(img, epic_blur)
        img = self._apply_textural_sharpening(img, textural_sharpening)
        img = self._apply_compositional_centering(img, compositional_centering)

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
        """Apply High Renaissance / MJ v4 tells to an existing image.

        This is the post-processing step only. Use this when you already have
        a base image (e.g., from batch Replicate generation) and want to apply
        the characteristic tells without re-generating.

        Args:
            img: Base image to apply tells to
            control: Artifact control parameters
            **era_params: blue_orange_cast, overdramatized_lighting, etc.

        Returns:
            Image with High Renaissance / MJ v4 tells applied
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

        blue_orange_cast = get_effect_strength("blue_orange_cast")
        overdramatized_lighting = get_effect_strength("overdramatized_lighting")
        hypersaturation = get_effect_strength("hypersaturation")
        epic_blur = get_effect_strength("epic_blur")
        textural_sharpening = get_effect_strength("textural_sharpening")
        compositional_centering = get_effect_strength("compositional_centering")
        warm_halo = get_effect_strength("warm_halo")

        # Apply MJ v4 tells in sequence
        img = self._apply_blue_orange_cast(img, blue_orange_cast)
        img = self._apply_hypersaturation(img, hypersaturation)
        img = self._apply_overdramatized_lighting(img, overdramatized_lighting)
        img = self._apply_warm_halo(img, warm_halo)
        img = self._apply_epic_blur(img, epic_blur)
        img = self._apply_textural_sharpening(img, textural_sharpening)
        img = self._apply_compositional_centering(img, compositional_centering)

        # Final blend with original based on placement
        img = control.apply_mask(original, img)

        # Upscale if requested
        upscale = era_params.get("upscale", 1)
        if upscale > 1:
            new_size = (img.width * upscale, img.height * upscale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _apply_blue_orange_cast(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply CINEMA TRAILER color grading. Teal shadows, orange highlights.

        This is the Michael Bay / Marvel / every blockbuster color grade.
        Shadows are TEAL. Highlights are ORANGE. There is no other color.
        This should look like it was graded for a trailer, not a film.

        The goal is parody - color grading so aggressive it's almost unwatchable.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        result = arr.copy()

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Calculate luminance
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        lum_norm = luminance / 255.0

        # === SHADOW MASK - very aggressive, captures more of the image ===
        shadow_mask = np.clip(1 - lum_norm * 1.5, 0, 1)  # More shadow range
        shadow_mask = shadow_mask ** 0.5  # Even more expansion
        shadow_mask = ndimage.gaussian_filter(shadow_mask, sigma=2)

        # === HIGHLIGHT MASK - also aggressive ===
        highlight_mask = np.clip(lum_norm * 2.0 - 0.6, 0, 1)  # More highlight range
        highlight_mask = highlight_mask ** 0.5
        highlight_mask = ndimage.gaussian_filter(highlight_mask, sigma=2)

        # === TEAL SHADOWS - the cinema look ===
        shadow_blend = shadow_mask * strength

        # HEAVY teal push - this should be unmistakable
        result[:, :, 0] = result[:, :, 0] * (1 - shadow_blend * 0.7)  # Kill red hard
        result[:, :, 1] = result[:, :, 1] * (1 - shadow_blend * 0.2) + shadow_blend * 50  # Cyan-green
        result[:, :, 2] = result[:, :, 2] + shadow_blend * 100  # HEAVY blue/teal

        # === ORANGE HIGHLIGHTS - the other half of the look ===
        highlight_blend = highlight_mask * strength

        # HEAVY orange push
        result[:, :, 0] = result[:, :, 0] + highlight_blend * 80  # Heavy red/orange
        result[:, :, 1] = result[:, :, 1] + highlight_blend * 40  # Gold
        result[:, :, 2] = result[:, :, 2] * (1 - highlight_blend * 0.6)  # Kill blue in highlights

        # === MIDTONES - desaturate and push toward the split ===
        midtone_mask = np.clip(1 - np.abs(lum_norm - 0.5) * 3, 0, 1)
        midtone_blend = midtone_mask * strength * 0.5

        # Desaturate midtones (the "filmic" look)
        gray_mid = luminance[:, :, np.newaxis].repeat(3, axis=2)
        result = result * (1 - midtone_blend[:, :, np.newaxis] * 0.3) + gray_mid * midtone_blend[:, :, np.newaxis] * 0.3

        # === CONTRAST BOOST - that graded pop ===
        result_norm = result / 255.0
        # S-curve contrast
        result_norm = 0.5 + (result_norm - 0.5) * (1 + strength * 0.5)
        # Lift blacks slightly (cinema never has true black)
        result_norm = result_norm * (1 - strength * 0.1) + strength * 0.05
        result = result_norm * 255.0

        # === FINAL TEAL WASH - even more teal ===
        # Add a global teal tint because MJ v4 couldn't help itself
        teal_wash = strength * 15
        result[:, :, 1] = result[:, :, 1] + teal_wash * 0.3
        result[:, :, 2] = result[:, :, 2] + teal_wash

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_overdramatized_lighting(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply ABSURD Caravaggio-meets-cinematographer lighting.

        GODRAYS. ALWAYS GODRAYS. Plus dramatic side lighting that makes
        everything cast heroic shadows. A loaf of bread should look like
        it's being lit for a perfume commercial.

        Strong directional light from the side (Caravaggio chiaroscuro)
        combined with volumetric rays from above (cinema trailer).
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        result = arr.copy()

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        lum_norm = gray / 255.0

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        y_norm = y_coords / h
        x_norm = x_coords / w

        # === EFFECT 1: DRAMATIC SIDE LIGHTING (Caravaggio style) ===
        # Strong light from left side, casting shadows to right
        side_light = np.clip(1 - x_norm * 1.5, 0, 1) ** 0.7
        side_light = ndimage.gaussian_filter(side_light, sigma=20)

        # Add warm side light
        side_strength = side_light * strength * 60
        result[:, :, 0] = result[:, :, 0] + side_strength * 1.3  # Orange light
        result[:, :, 1] = result[:, :, 1] + side_strength * 0.9
        result[:, :, 2] = result[:, :, 2] + side_strength * 0.4

        # Darken the opposite side (shadow)
        shadow_side = np.clip(x_norm * 2 - 0.5, 0, 1) ** 1.5
        shadow_side = ndimage.gaussian_filter(shadow_side, sigma=15)
        shadow_strength = shadow_side * strength * 0.4
        result = result * (1 - shadow_strength[:, :, np.newaxis])

        # === EFFECT 2: AGGRESSIVE rim lighting ===
        edges_x = np.abs(ndimage.sobel(gray, axis=1))
        edges_y = np.abs(ndimage.sobel(gray, axis=0))
        edges = np.sqrt(edges_x**2 + edges_y**2)

        if edges.max() > 0:
            edges = edges / edges.max()

        # VERY thick rim
        rim_mask = ndimage.maximum_filter(edges, size=10)
        rim_mask = ndimage.gaussian_filter(rim_mask, sigma=5)
        rim_mask = rim_mask ** 0.6

        # VERY bright rim light
        rim_strength = rim_mask * strength * 150
        result[:, :, 0] = result[:, :, 0] + rim_strength * 1.4
        result[:, :, 1] = result[:, :, 1] + rim_strength * 1.0
        result[:, :, 2] = result[:, :, 2] + rim_strength * 0.5

        # === EFFECT 3: HEAVY GODRAYS - always godrays ===
        # Primary rays from top-center
        angle_top = np.arctan2(x_norm - 0.5, y_norm + 0.05)
        godray_main = np.cos(angle_top * 15) ** 2
        godray_main = godray_main * (1 - y_norm) ** 0.4
        godray_main = ndimage.gaussian_filter(godray_main, sigma=8)

        # Secondary diagonal rays
        angle_diag = np.arctan2(y_norm - 0.1, x_norm - 0.2)
        godray_diag = np.cos(angle_diag * 12) ** 2 * (1 - y_norm) * (1 - x_norm)
        godray_diag = ndimage.gaussian_filter(godray_diag, sigma=10)

        # Combine and make OBVIOUS
        godray_mask = godray_main * 0.6 + godray_diag * 0.4
        godray_strength = godray_mask * strength * 120  # HEAVY

        # Golden godrays
        result[:, :, 0] = result[:, :, 0] + godray_strength * 1.3
        result[:, :, 1] = result[:, :, 1] + godray_strength * 1.0
        result[:, :, 2] = result[:, :, 2] + godray_strength * 0.5

        # === EFFECT 4: Backlight bloom ===
        bright_mask = np.clip(lum_norm * 2.5 - 1.0, 0, 1)
        bright_mask = ndimage.gaussian_filter(bright_mask, sigma=18)
        bloom = ndimage.gaussian_filter(arr, sigma=[25, 25, 0])

        bloom_strength = bright_mask[:, :, np.newaxis] * strength * 0.7
        result = result + bloom * bloom_strength

        # === EFFECT 5: Lifted shadows to teal ===
        dark_mask = np.clip(1 - lum_norm * 2.5, 0, 1)
        dark_lift = dark_mask * strength * 35
        result[:, :, 2] = result[:, :, 2] + dark_lift  # Teal shadows

        # === EFFECT 6: HEAVY contrast ===
        mean_lum = np.mean(gray)
        contrast_factor = 1 + strength * 0.5
        result = mean_lum + (result - mean_lum) * contrast_factor

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_hypersaturation(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply ABSURD saturation. Instagram filters on steroids.

        MJ v4 cranked saturation until colors screamed. Blues became electric.
        Oranges became nuclear. Skin tones became golden bronze. Every color
        pushed to the edge of what monitors can display.

        This is the HDR real estate photo look applied to everything.
        Saturation so high it's almost painful.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        result = arr.copy()

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # Calculate luminance
        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        # === MASSIVE global saturation boost ===
        sat_boost = 1 + strength * 1.2  # Much more aggressive

        gray = luminance[:, :, np.newaxis].repeat(3, axis=2)
        diff = result - gray
        result = gray + diff * sat_boost

        # === SLAM blues to electric ===
        blue_mask = (b > r * 0.9) & (b > g * 0.9)
        blue_mask = blue_mask.astype(np.float32)
        blue_mask = ndimage.gaussian_filter(blue_mask, sigma=5)

        # Electric blue boost
        blue_boost = blue_mask * strength * 70
        result[:, :, 2] = result[:, :, 2] + blue_boost
        result[:, :, 0] = result[:, :, 0] - blue_boost * 0.3  # Purify the blue

        # === SLAM oranges/reds to nuclear ===
        orange_mask = (r > g * 0.8) & (r > b)
        orange_mask = orange_mask.astype(np.float32)
        orange_mask = ndimage.gaussian_filter(orange_mask, sigma=5)

        # Nuclear orange boost
        orange_boost = orange_mask * strength * 60
        result[:, :, 0] = result[:, :, 0] + orange_boost
        result[:, :, 1] = result[:, :, 1] + orange_boost * 0.4
        result[:, :, 2] = result[:, :, 2] - orange_boost * 0.3

        # === Push ALL greens toward teal ===
        # MJ v4 didn't do pure green - it was always teal
        green_mask = (g > r * 0.9) & (g > b * 0.9)
        green_mask = green_mask.astype(np.float32)
        green_mask = ndimage.gaussian_filter(green_mask, sigma=5)

        teal_shift = green_mask * strength * 50
        result[:, :, 2] = result[:, :, 2] + teal_shift  # Add blue
        result[:, :, 1] = result[:, :, 1] - teal_shift * 0.2  # Reduce pure green

        # === Golden skin tones ===
        # Detect skin-ish colors and push them golden
        skin_mask = (
            (r > 100) & (g > 60) & (b > 40) &
            (r > g) & (g > b) &
            (r - b > 20) & (r - b < 120)
        ).astype(np.float32)
        skin_mask = ndimage.gaussian_filter(skin_mask, sigma=8)

        golden_boost = skin_mask * strength * 40
        result[:, :, 0] = result[:, :, 0] + golden_boost * 0.8  # More red
        result[:, :, 1] = result[:, :, 1] + golden_boost * 0.5  # More gold
        result[:, :, 2] = result[:, :, 2] - golden_boost * 0.3  # Less blue

        # === Vibrance boost (saturate undersaturated areas more) ===
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = (max_rgb - min_rgb) / (max_rgb + 0.0001)

        # Low saturation areas get extra boost
        low_sat_mask = np.clip(1 - saturation * 2, 0, 1)
        vibrance_boost = low_sat_mask * strength * 0.4
        diff = result - gray
        result = gray + diff * (1 + vibrance_boost[:, :, np.newaxis])

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_epic_blur(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply HEAVY atmospheric haze and dramatic depth.

        MJ v4 backgrounds dissolve into painterly atmospheric drama.
        Heavy haze, dramatic fog, everything looks like it's set in
        a misty cathedral or a battlefield at dawn.

        The atmosphere should be THICK. Cheap drama via fog machine.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        y_norm = y_coords / h
        x_norm = x_coords / w
        center_y, center_x = h // 2, w // 2

        dist = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        dist_norm = dist / max_dist

        # === AGGRESSIVE depth blur ===
        blur_mask = np.clip(dist_norm * 3 - 0.2, 0, 1)
        blur_mask = blur_mask ** 1.0
        blur_mask = ndimage.gaussian_filter(blur_mask, sigma=12)

        # Multiple blur passes
        blur_light = ndimage.gaussian_filter(arr, sigma=[4, 4, 0])
        blur_medium = ndimage.gaussian_filter(arr, sigma=[12, 12, 0])
        blur_heavy = ndimage.gaussian_filter(arr, sigma=[25, 25, 0])
        blur_massive = ndimage.gaussian_filter(arr, sigma=[40, 40, 0])

        blur_mask_3d = blur_mask[:, :, np.newaxis]

        result = (
            arr * (1 - blur_mask_3d * strength) +
            blur_light * (blur_mask_3d * strength * 0.1) +
            blur_medium * (blur_mask_3d * strength * 0.2) +
            blur_heavy * (blur_mask_3d * strength * 0.35) +
            blur_massive * (blur_mask_3d * strength * 0.35)
        )

        # === HEAVY ATMOSPHERIC HAZE ===
        # Warm-teal atmospheric haze (matches the color grade)
        haze_color = np.array([180, 195, 220])  # Teal-ish haze
        haze = haze_color.reshape(1, 1, 3) * np.ones_like(arr)

        # Haze increases with distance from center and toward top
        haze_mask = blur_mask * 0.4 + (1 - y_norm) * 0.3
        haze_mask = np.clip(haze_mask, 0, 1)
        haze_mask = ndimage.gaussian_filter(haze_mask, sigma=20)
        haze_strength = haze_mask * strength * 0.5

        result = result * (1 - haze_strength[:, :, np.newaxis]) + haze * haze_strength[:, :, np.newaxis]

        # === DRAMATIC TOP HAZE (sky/background) ===
        # Top of image gets heavy atmospheric treatment
        top_fade = np.clip(1 - y_norm * 2, 0, 1) ** 0.8
        top_fade = ndimage.gaussian_filter(top_fade, sigma=30)

        # Warm atmospheric color for top
        top_haze_color = np.array([200, 190, 210])  # Slight purple/warm
        top_haze = top_haze_color.reshape(1, 1, 3) * np.ones_like(arr)

        top_strength = top_fade * strength * 0.4
        result = result * (1 - top_strength[:, :, np.newaxis]) + top_haze * top_strength[:, :, np.newaxis]

        # === EDGE VIGNETTE (more atmospheric at edges) ===
        vignette = dist_norm ** 1.5
        vignette = ndimage.gaussian_filter(vignette, sigma=30)
        vignette_strength = vignette * strength * 0.25

        # Darken and haze the edges
        result = result * (1 - vignette_strength[:, :, np.newaxis] * 0.5)
        result = result * (1 - vignette_strength[:, :, np.newaxis] * 0.3) + haze * vignette_strength[:, :, np.newaxis] * 0.3

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_textural_sharpening(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply OVER-RETOUCHED PERFECTION to faces, sharp textures elsewhere.

        MJ v4 faces are TOO PERFECT - the uncanny valley of idealization.
        Skin like porcelain. Features too symmetrical. Too smooth, too luminous.
        Think 'Leonardo with Photoshop who couldn't stop retouching.'

        The faces should make you uncomfortable because they're TOO flawless.
        Meanwhile, fabrics and textures are razor-sharp.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # === DETECT SKIN/FACE REGIONS ===
        # Broader skin detection to catch faces
        skin_mask = (
            (r > 60) & (g > 40) & (b > 20) &
            (r > g * 0.8) & (g > b * 0.7) &
            (r - b > 5) & (r - b < 150) &
            (np.abs(r - g) < 80)
        ).astype(np.float32)

        # Heavy smoothing of mask for soft transitions
        skin_mask = ndimage.gaussian_filter(skin_mask, sigma=12)
        skin_mask = np.clip(skin_mask * 1.5, 0, 1)  # Expand the mask

        # === PORCELAIN SKIN EFFECT - AGGRESSIVE SMOOTHING ===
        # Multiple passes of smoothing for that airbrushed look
        smooth_light = ndimage.gaussian_filter(arr, sigma=[3, 3, 0])
        smooth_medium = ndimage.gaussian_filter(arr, sigma=[6, 6, 0])
        smooth_heavy = ndimage.gaussian_filter(arr, sigma=[10, 10, 0])

        # Blend smoothed versions for porcelain effect
        skin_smooth = (
            smooth_light * 0.3 +
            smooth_medium * 0.4 +
            smooth_heavy * 0.3
        )

        # Apply smoothing to skin regions
        skin_blend = skin_mask[:, :, np.newaxis] * strength * 0.8
        result = arr * (1 - skin_blend) + skin_smooth * skin_blend

        # === LUMINOUS GLOW - skin that glows too much ===
        # Inner glow on skin areas
        skin_glow = ndimage.gaussian_filter(skin_mask, sigma=15)
        glow_strength = skin_glow * strength * 45

        # Golden/peachy luminous skin
        result[:, :, 0] = result[:, :, 0] + glow_strength * 1.0   # Warm
        result[:, :, 1] = result[:, :, 1] + glow_strength * 0.85  # Peachy
        result[:, :, 2] = result[:, :, 2] + glow_strength * 0.6   # Less blue

        # === SUBSURFACE SCATTERING FAKE - that CG skin look ===
        # Bright areas of skin get extra luminosity
        skin_bright = np.clip(gray / 255.0 * skin_mask, 0, 1)
        skin_bright = ndimage.gaussian_filter(skin_bright, sigma=8)
        sss_strength = skin_bright * strength * 30

        result[:, :, 0] = result[:, :, 0] + sss_strength * 0.9
        result[:, :, 1] = result[:, :, 1] + sss_strength * 0.7
        result[:, :, 2] = result[:, :, 2] + sss_strength * 0.5

        # === EVEN OUT SKIN TONES - remove variation ===
        # Push skin toward a uniform tone (the over-retouched look)
        skin_mean = np.zeros(3)
        skin_pixels = skin_mask > 0.3
        if np.any(skin_pixels):
            for c in range(3):
                skin_mean[c] = np.mean(result[:, :, c][skin_pixels])

            # Blend toward mean skin tone
            evenness = skin_mask[:, :, np.newaxis] * strength * 0.3
            skin_target = np.ones_like(result) * skin_mean.reshape(1, 1, 3)
            result = result * (1 - evenness) + skin_target * evenness

        # === SOFT FOCUS ON SKIN EDGES - dreamy perfection ===
        # Blur the edges between skin and non-skin for that glamour shot look
        edge_blur = ndimage.gaussian_filter(result, sigma=[2, 2, 0])
        skin_edge = ndimage.gaussian_filter(skin_mask, sigma=5) - ndimage.gaussian_filter(skin_mask, sigma=15)
        skin_edge = np.clip(np.abs(skin_edge) * 3, 0, 1)
        edge_blend = skin_edge[:, :, np.newaxis] * strength * 0.5
        result = result * (1 - edge_blend) + edge_blur * edge_blend

        # === SHARP TEXTURES ON NON-SKIN ===
        # Fabrics, hair, details should be razor sharp
        non_skin_mask = 1 - skin_mask

        # Aggressive unsharp mask
        img_from_result = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
        img_sharp = img_from_result.filter(ImageFilter.UnsharpMask(
            radius=2,
            percent=int(200 * strength),
            threshold=2
        ))
        arr_sharp = np.array(img_sharp, dtype=np.float32)

        # Apply sharpening only to non-skin
        sharp_blend = non_skin_mask[:, :, np.newaxis] * strength * 0.6
        result = result * (1 - sharp_blend) + arr_sharp * sharp_blend

        # === MICROCONTRAST on textures ===
        # Detect textured regions (high local variance)
        local_mean = ndimage.uniform_filter(gray, size=10)
        local_sq_mean = ndimage.uniform_filter(gray**2, size=10)
        local_var = np.clip(local_sq_mean - local_mean**2, 0, None)

        if local_var.max() > 0:
            texture_mask = local_var / local_var.max()
        else:
            texture_mask = np.zeros_like(gray)

        texture_mask = texture_mask * non_skin_mask  # Only on non-skin
        texture_mask = ndimage.gaussian_filter(texture_mask, sigma=3)

        for c in range(3):
            channel = result[:, :, c]
            local_mean_ch = ndimage.uniform_filter(channel, size=12)
            deviation = channel - local_mean_ch
            enhanced = local_mean_ch + deviation * (1 + strength * 0.5)
            blend = texture_mask * strength * 0.4
            result[:, :, c] = channel * (1 - blend) + enhanced * blend

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_compositional_centering(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply the "everything perfectly centered" tendency.

        Midjourney v4 loved centering subjects. Everything dead-center,
        perfectly symmetrical when it shouldn't be. Subjects feel "placed"
        rather than captured naturally.

        Maps to Raphael's perfect compositions, but over-applied.

        Technique:
        - Create subtle pull toward center
        - Enhance symmetry slightly
        - Not a hard crop, but a gravitational effect
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        center_y, center_x = h // 2, w // 2

        # Distance from center
        dy = y_coords - center_y
        dx = x_coords - center_x

        # Normalize distances
        dy_norm = dy / (h / 2)
        dx_norm = dx / (w / 2)

        # Create radial pull toward center
        # Pixels far from center get pulled more
        dist_norm = np.sqrt(dy_norm**2 + dx_norm**2)

        # Pull strength increases with distance from center
        pull_factor = dist_norm ** 2 * strength * 0.15

        # Calculate displacement (toward center)
        displacement_y = -dy * pull_factor
        displacement_x = -dx * pull_factor

        # New coordinates
        new_y = np.clip(y_coords + displacement_y, 0, h - 1)
        new_x = np.clip(x_coords + displacement_x, 0, w - 1)

        result = np.zeros_like(arr)
        for c in range(3):
            result[:, :, c] = ndimage.map_coordinates(
                arr[:, :, c],
                [new_y, new_x],
                order=1,
                mode='reflect'
            )

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_warm_halo(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply AGGRESSIVE warm glow around EVERYTHING.

        MJ v4 made everything glow like it was backlit by a golden sun.
        Hair gets halos. Shoulders get halos. Even inanimate objects
        get that warm divine glow bleeding outward.

        This should be OBVIOUS - subjects should look like they're
        standing in front of a sunset at all times.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        result = arr.copy()

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        lum_norm = gray / 255.0

        # Detect ALL edges - everything gets a halo
        edges_x = np.abs(ndimage.sobel(gray, axis=1))
        edges_y = np.abs(ndimage.sobel(gray, axis=0))
        edges = np.sqrt(edges_x**2 + edges_y**2)

        if edges.max() > 0:
            edges = edges / edges.max()

        # === AGGRESSIVE halo expansion ===
        # Multiple layers of glow
        halo_inner = ndimage.maximum_filter(edges, size=8)
        halo_inner = ndimage.gaussian_filter(halo_inner, sigma=5)

        halo_mid = ndimage.maximum_filter(edges, size=15)
        halo_mid = ndimage.gaussian_filter(halo_mid, sigma=10)

        halo_outer = ndimage.maximum_filter(edges, size=25)
        halo_outer = ndimage.gaussian_filter(halo_outer, sigma=18)

        # Combine halos
        combined_halo = halo_inner * 0.5 + halo_mid * 0.3 + halo_outer * 0.2

        # Subtract core edges to get pure glow region
        edge_narrow = ndimage.gaussian_filter(edges, sigma=2)
        glow_mask = np.clip(combined_halo - edge_narrow * 0.3, 0, 1)

        # === HEAVY warm glow ===
        glow_strength = glow_mask * strength * 80  # Much stronger

        # Golden-orange glow
        result[:, :, 0] = result[:, :, 0] + glow_strength * 1.4  # Heavy red/orange
        result[:, :, 1] = result[:, :, 1] + glow_strength * 0.9  # Gold
        result[:, :, 2] = result[:, :, 2] + glow_strength * 0.3  # Minimal blue

        # === EXTRA glow on bright areas (divine light effect) ===
        bright_mask = np.clip(lum_norm * 2 - 0.5, 0, 1)
        bright_halo = ndimage.gaussian_filter(bright_mask * edges, sigma=15)

        bright_strength = bright_halo * strength * 60
        result[:, :, 0] = result[:, :, 0] + bright_strength * 1.3
        result[:, :, 1] = result[:, :, 1] + bright_strength * 1.0
        result[:, :, 2] = result[:, :, 2] + bright_strength * 0.4

        # === Bloom from highlights ===
        # Bright areas should BLEED light
        highlights = np.clip(lum_norm * 3 - 2, 0, 1)
        highlight_bloom = ndimage.gaussian_filter(highlights, sigma=20)
        bloom_color = highlight_bloom * strength * 50

        result[:, :, 0] = result[:, :, 0] + bloom_color * 1.2
        result[:, :, 1] = result[:, :, 1] + bloom_color * 0.95
        result[:, :, 2] = result[:, :, 2] + bloom_color * 0.6

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
