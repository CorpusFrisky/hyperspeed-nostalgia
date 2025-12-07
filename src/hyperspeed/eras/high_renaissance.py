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
1. Generate image with Stable Diffusion (or transform source image)
2. Apply characteristic Midjourney v4 tells:
   - Blue-orange color cast (THE signature)
   - Over-dramatized lighting (rim light, volumetric rays)
   - Hyper-saturation (pushed beyond natural)
   - Epic blur (shallow DOF everywhere)
   - Textural sharpening (every pore visible)
   - Compositional centering (pull subjects center)
   - Warm halo (glow around edges)
"""

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

from hyperspeed.core.artifact_control import ArtifactControl
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

        Returns:
            Generated image with High Renaissance / MJ v4 tells
        """
        control = control or ArtifactControl()
        params = {**self.get_default_params(), **era_params}

        # Generation parameters
        num_steps = params.get("inference_steps", 30)
        guidance = params.get("guidance_scale", 10.0)

        # Determine generation mode
        if source_image is None:
            # txt2img mode
            if prompt is None:
                raise ValueError("High Renaissance pipeline requires a prompt or source image")

            self.ensure_loaded()

            generator = None
            if control.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(control.seed)
                np.random.seed(control.seed)

            width = era_params.get("width", 512)
            height = era_params.get("height", 512)

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

    def _apply_blue_orange_cast(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply THE signature Midjourney v4 color grading - CRANKED TO ABSURD.

        The most recognizable MJ v4 tell: teal/blue shadows paired with
        orange/amber highlights. This is the Hollywood blockbuster color grade
        pushed to parody levels. Every shadow screams teal. Every highlight
        screams orange. Subtlety is dead.

        This should make images look like they were color graded by someone
        who just discovered the teal-orange split toning slider and maxed it.
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

        # Create shadow mask (dark areas) - MORE AGGRESSIVE
        shadow_mask = np.clip(1 - lum_norm * 2.0, 0, 1)
        shadow_mask = shadow_mask ** 0.7  # Expand shadow range
        shadow_mask = ndimage.gaussian_filter(shadow_mask, sigma=3)

        # Create highlight mask (bright areas) - MORE AGGRESSIVE
        highlight_mask = np.clip(lum_norm * 2.5 - 1.0, 0, 1)
        highlight_mask = highlight_mask ** 0.7  # Expand highlight range
        highlight_mask = ndimage.gaussian_filter(highlight_mask, sigma=3)

        # === EFFECT 1: SLAM shadows toward teal/cyan ===
        # This should be OBVIOUS, not subtle
        shadow_blend = shadow_mask * strength

        # Aggressively shift shadows to teal
        result[:, :, 0] = result[:, :, 0] * (1 - shadow_blend * 0.6)  # Kill red in shadows
        result[:, :, 1] = result[:, :, 1] + shadow_blend * 30  # Add cyan-green
        result[:, :, 2] = result[:, :, 2] + shadow_blend * 70  # Heavy blue push

        # === EFFECT 2: SLAM highlights toward orange/amber ===
        highlight_blend = highlight_mask * strength

        # Aggressively shift highlights to orange
        result[:, :, 0] = result[:, :, 0] + highlight_blend * 60  # Heavy red
        result[:, :, 1] = result[:, :, 1] + highlight_blend * 30  # Orange gold
        result[:, :, 2] = result[:, :, 2] * (1 - highlight_blend * 0.5)  # Kill blue in highlights

        # === EFFECT 3: Crush midtones toward the split ===
        # Even midtones get pulled toward the teal-orange axis
        midtone_mask = np.clip(1 - np.abs(lum_norm - 0.5) * 4, 0, 1)
        midtone_blend = midtone_mask * strength * 0.4

        # Push midtones toward a desaturated teal-orange neutral
        result[:, :, 0] = result[:, :, 0] + midtone_blend * 15  # Slight warm
        result[:, :, 2] = result[:, :, 2] + midtone_blend * 10  # Slight cool

        # === EFFECT 4: Global contrast boost for that "graded" look ===
        # S-curve to make it pop
        result_norm = result / 255.0
        s_curve = result_norm ** (1 - strength * 0.2) * (1 + strength * 0.3)
        result = s_curve * 255.0

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_overdramatized_lighting(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply ABSURD movie-poster lighting. Godrays EVERYWHERE.

        MJ v4 lit everything like a Marvel movie. Rim light on every edge.
        Volumetric rays streaming from above. Backlight bloom so strong
        subjects glow. This is theatrical lighting applied to a still life
        of fruit. NOTHING is subtle.

        The goal is parody - lighting so dramatic it borders on ridiculous.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        result = arr.copy()

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        lum_norm = gray / 255.0

        # === EFFECT 1: AGGRESSIVE rim lighting ===
        # Every edge gets a glowing halo
        edges_x = np.abs(ndimage.sobel(gray, axis=1))
        edges_y = np.abs(ndimage.sobel(gray, axis=0))
        edges = np.sqrt(edges_x**2 + edges_y**2)

        if edges.max() > 0:
            edges = edges / edges.max()

        # THICK rim light - dilate edges heavily
        rim_mask = ndimage.maximum_filter(edges, size=8)
        rim_mask = ndimage.gaussian_filter(rim_mask, sigma=4)
        rim_mask = rim_mask ** 0.7  # Expand the rim

        # BRIGHT warm rim light - this should be OBVIOUS
        rim_strength = rim_mask * strength * 120
        result[:, :, 0] = result[:, :, 0] + rim_strength * 1.3  # Heavy orange rim
        result[:, :, 1] = result[:, :, 1] + rim_strength * 0.9
        result[:, :, 2] = result[:, :, 2] + rim_strength * 0.4

        # === EFFECT 2: GODRAYS from multiple sources ===
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        y_norm = y_coords / h
        x_norm = x_coords / w

        # Primary godrays from top-center (the classic)
        angle_from_top = np.arctan2(x_norm - 0.5, y_norm + 0.1)
        godray_primary = np.cos(angle_from_top * 12) ** 2
        godray_primary = godray_primary * (1 - y_norm) ** 0.5  # Fade down
        godray_primary = ndimage.gaussian_filter(godray_primary, sigma=10)

        # Secondary godrays from top-left
        angle_tl = np.arctan2(y_norm, x_norm)
        godray_tl = np.cos(angle_tl * 10) ** 2 * (1 - y_norm) * (1 - x_norm)
        godray_tl = ndimage.gaussian_filter(godray_tl, sigma=12)

        # Tertiary from top-right
        angle_tr = np.arctan2(y_norm, 1 - x_norm)
        godray_tr = np.cos(angle_tr * 10) ** 2 * (1 - y_norm) * x_norm
        godray_tr = ndimage.gaussian_filter(godray_tr, sigma=12)

        # Combine all godrays
        godray_mask = (godray_primary * 0.5 + godray_tl * 0.3 + godray_tr * 0.3)
        godray_strength = godray_mask * strength * 80

        # Golden godrays
        result[:, :, 0] = result[:, :, 0] + godray_strength * 1.2
        result[:, :, 1] = result[:, :, 1] + godray_strength * 0.95
        result[:, :, 2] = result[:, :, 2] + godray_strength * 0.5

        # === EFFECT 3: Backlight bloom ===
        # Bright areas should GLOW and bleed
        bright_mask = np.clip(lum_norm * 3 - 1.5, 0, 1)
        bright_mask = ndimage.gaussian_filter(bright_mask, sigma=15)
        bloom = ndimage.gaussian_filter(arr, sigma=[20, 20, 0])

        bloom_strength = bright_mask[:, :, np.newaxis] * strength * 0.6
        result = result + bloom * bloom_strength

        # === EFFECT 4: Crushed blacks with lifted shadows ===
        # The "cinematic" look - blacks aren't black, they're dark blue
        dark_mask = np.clip(1 - lum_norm * 3, 0, 1)
        dark_lift = dark_mask * strength * 25
        result[:, :, 2] = result[:, :, 2] + dark_lift  # Lift shadows to blue

        # === EFFECT 5: Dramatic contrast boost ===
        # Make everything POP
        mean_lum = np.mean(gray)
        contrast_factor = 1 + strength * 0.4
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
        """Apply AGGRESSIVE portrait-mode blur with atmospheric haze.

        MJ v4 blurred backgrounds into painterly oblivion while keeping
        subjects razor sharp. Plus that atmospheric haze/fog used for
        cheap drama. Even a photo of a sandwich gets the cinematic treatment.

        This should be OBVIOUS - backgrounds should dissolve into nothing.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Create radial gradient - AGGRESSIVE center focus
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        center_y, center_x = h // 2, w // 2

        dist = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        dist_norm = dist / max_dist

        # AGGRESSIVE blur falloff - sharp center, VERY blurry edges
        blur_mask = np.clip(dist_norm * 2.5 - 0.3, 0, 1)
        blur_mask = blur_mask ** 1.2  # Even more aggressive
        blur_mask = ndimage.gaussian_filter(blur_mask, sigma=15)

        # Create HEAVY blur levels
        blur_small = ndimage.gaussian_filter(arr, sigma=[3, 3, 0])
        blur_medium = ndimage.gaussian_filter(arr, sigma=[8, 8, 0])
        blur_large = ndimage.gaussian_filter(arr, sigma=[18, 18, 0])
        blur_massive = ndimage.gaussian_filter(arr, sigma=[30, 30, 0])

        blur_mask_3d = blur_mask[:, :, np.newaxis]

        # Progressive blur - edges get MASSIVE blur
        result = (
            arr * (1 - blur_mask_3d * strength) +
            blur_small * (blur_mask_3d * strength * 0.15) +
            blur_medium * (blur_mask_3d * strength * 0.25) +
            blur_large * (blur_mask_3d * strength * 0.3) +
            blur_massive * (blur_mask_3d * strength * 0.3)
        )

        # === ATMOSPHERIC HAZE ===
        # Add that cheap dramatic fog effect
        # More haze at edges/background
        haze_color = np.array([200, 210, 230])  # Bluish haze
        haze_mask = blur_mask * strength * 0.25
        haze = haze_color.reshape(1, 1, 3) * np.ones_like(arr)
        result = result * (1 - haze_mask[:, :, np.newaxis]) + haze * haze_mask[:, :, np.newaxis]

        # === Top-down depth gradient ===
        # Things at top are "further" - more blurred and hazy
        y_norm = y_coords / h
        top_haze = np.clip(1 - y_norm * 1.5, 0, 0.5) * strength
        top_haze_3d = top_haze[:, :, np.newaxis]
        result = result * (1 - top_haze_3d * 0.3) + haze * top_haze_3d * 0.3

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_textural_sharpening(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply AGGRESSIVE sharpening + waxy/plastic skin effect.

        MJ v4 made everything hyper-detailed AND slightly plastic. Textures
        razor sharp. Skin that's too smooth, too luminous, slightly CG.
        Every pore visible but also somehow unnaturally perfect.

        The combination of over-sharpening with over-smoothed skin creates
        that distinctive AI-generated look.
        """
        if strength < 0.01:
            return img

        # AGGRESSIVE unsharp mask
        img_sharp = img.filter(ImageFilter.UnsharpMask(
            radius=2,
            percent=int(250 * strength),  # Much more aggressive
            threshold=2
        ))

        # Second pass of sharpening for extra crispness
        img_sharp = img_sharp.filter(ImageFilter.UnsharpMask(
            radius=1,
            percent=int(100 * strength),
            threshold=1
        ))

        arr = np.array(img, dtype=np.float32)
        arr_sharp = np.array(img_sharp, dtype=np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Detect textured regions
        local_mean = ndimage.uniform_filter(gray, size=10)
        local_sq_mean = ndimage.uniform_filter(gray**2, size=10)
        local_var = local_sq_mean - local_mean**2
        local_var = np.clip(local_var, 0, None)

        if local_var.max() > 0:
            texture_mask = local_var / local_var.max()
        else:
            texture_mask = np.zeros_like(gray)

        texture_mask = ndimage.gaussian_filter(texture_mask, sigma=3)

        # === Apply HEAVY sharpening to textured areas ===
        texture_blend = texture_mask[:, :, np.newaxis] * strength * 0.8
        result = arr * (1 - texture_blend) + arr_sharp * texture_blend

        # === WAXY/PLASTIC SKIN EFFECT ===
        # Detect skin regions
        skin_mask = (
            (r > 80) & (g > 50) & (b > 30) &
            (r > g) & (g > b) &
            (r - b > 10) & (r - b < 120)
        ).astype(np.float32)
        skin_mask = ndimage.gaussian_filter(skin_mask, sigma=8)

        # Smooth skin while keeping it sharp at edges - the UNCANNY effect
        skin_smooth = ndimage.gaussian_filter(result, sigma=[4, 4, 0])

        # Blend smooth skin with sharp details
        skin_blend = skin_mask[:, :, np.newaxis] * strength * 0.5
        result = result * (1 - skin_blend) + skin_smooth * skin_blend

        # Add luminous quality to skin (too bright, too perfect)
        skin_luminance = skin_mask * strength * 20
        result[:, :, 0] = result[:, :, 0] + skin_luminance * 0.8
        result[:, :, 1] = result[:, :, 1] + skin_luminance * 0.7
        result[:, :, 2] = result[:, :, 2] + skin_luminance * 0.5

        # === AGGRESSIVE microcontrast ===
        for c in range(3):
            channel = result[:, :, c]
            local_mean = ndimage.uniform_filter(channel, size=15)
            deviation = channel - local_mean
            # HEAVY boost to local deviations
            enhanced = local_mean + deviation * (1 + strength * 0.6)
            blend = texture_mask * strength * 0.5
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
