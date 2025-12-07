"""International Gothic Era (1375-1425) / StyleGAN "This Person Does Not Exist" parallel.

Faces that are suspiciously beautiful. Smooth, porcelain skin. Technical proficiency
in service of idealization. International Gothic was the bridge between Byzantine
formalism and Renaissance naturalism, reaching for realistic faces but landing in
an elegant uncanny valley.

Key works: Simone Martini's Annunciation (1333), with the Virgin's face technically
accomplished but emotionally vacant. Gentile da Fabriano's Adoration of the Magi
(1423), where crowd faces feel like variations on the same seed. The Wilton Diptych
(c. 1395-1399), with eleven angels of nearly identical faces.

StyleGAN's tells: asymmetrical earrings, hair bleeding into backgrounds, eyes that
track slightly wrong. International Gothic's tells: the same face appearing across
figures, backgrounds that don't connect to figures, jewelry rendered with more
conviction than the faces wearing it.

Technical approach:
1. Generate portrait image using Stable Diffusion with medieval prompting
2. Apply StyleGAN-like artifacts as post-processing:
   - Porcelain smoothness (over-processed skin)
   - Hair bleeding into background
   - Eye tracking error (asymmetric gaze)
   - Asymmetric accessories
   - Background disconnect (sharp figure, confused background)
3. Apply International Gothic stylization:
   - Gold leaf background
   - Tempera texture
   - Courtly palette (rich blues, reds, gold)
   - Decorative precision (jewelry sharper than faces)
   - Emotional vacancy
"""

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

from hyperspeed.core.artifact_control import ArtifactControl
from hyperspeed.core.image_utils import resize_to_multiple
from hyperspeed.eras.base import EraMetadata, EraPipeline, EraRegistry


@EraRegistry.register
class InternationalGothicPipeline(EraPipeline):
    """International Gothic Era: StyleGAN's uncanny valley meets medieval elegance.

    This pipeline generates portraits with the characteristic "suspiciously beautiful"
    quality of StyleGAN faces, styled as International Gothic panel paintings.
    Faces that pass at thumbnail resolution but unsettle on closer inspection.

    Controls:
        porcelain_smoothness: Over-smooth skin (0-1)
        hair_bleeding: Hair dissolves into background (0-1)
        eye_tracking_error: Eyes that don't quite align (0-1)
        asymmetric_accessories: Earrings/jewelry that don't match (0-1)
        background_disconnect: Sharp figure, confused background (0-1)
        gold_background: Gold leaf void intensity (0-1)
        tempera_texture: Medieval surface quality (0-1)
        courtly_palette: Rich blues/reds/gold color shift (0-1)
        decorative_precision: Jewelry sharper than faces (0-1)
        emotional_vacancy: Flatten expression areas (0-1)
    """

    metadata: ClassVar[EraMetadata] = EraMetadata(
        name="International Gothic",
        art_historical_parallel="International Gothic (1375-1425)",
        time_period="2019-2021",
        description=(
            "Faces that are suspiciously beautiful. Smooth, porcelain skin. "
            "The elegant uncanny valley where StyleGAN meets Simone Martini. "
            "Technical proficiency in service of idealization."
        ),
        characteristic_artifacts=[
            "Porcelain-smooth skin",
            "Asymmetrical earrings/accessories",
            "Hair bleeding into backgrounds",
            "Eyes that track slightly wrong",
            "Jewelry more convincing than faces",
            "Emotionally vacant expressions",
            "Gold void backgrounds",
            "Same face, different hats",
        ],
    )

    def __init__(self, model_path: Path | None = None, device: str = "mps"):
        super().__init__(model_path, device)
        self._pipe = None

    def load_model(self) -> None:
        """Load Stable Diffusion for portrait generation."""
        if self._pipe is not None:
            return

        from diffusers import StableDiffusionPipeline

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        self._pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        self._pipe = self._pipe.to(self.device)
        self._pipe.enable_attention_slicing()
        self._pipe.safety_checker = None

        self._model = True

    def get_default_params(self) -> dict[str, Any]:
        return {
            # StyleGAN artifact simulation
            "porcelain_smoothness": 0.7,
            "hair_bleeding": 0.6,
            "eye_tracking_error": 0.4,
            "asymmetric_accessories": 0.5,
            "background_disconnect": 0.5,

            # International Gothic stylization
            "gold_background": 0.5,
            "gold_style": "metallic",
            "tempera_texture": 0.4,
            "courtly_palette": 0.6,
            "decorative_precision": 0.5,
            "emotional_vacancy": 0.4,

            # SD generation parameters
            "inference_steps": 25,
            "guidance_scale": 7.5,
        }

    def _build_prompt(self, subject: str) -> str:
        """Build International Gothic style prompt from subject."""
        return (
            f"portrait of {subject}, International Gothic style painting, "
            "medieval courtly costume, ornate gold jewelry, rich blue and red robes, "
            "gold leaf background, tempera painting on wood panel, "
            "Simone Martini style, 1400s Italian art, elegant, idealized face, "
            "detailed decorative elements, religious icon aesthetic"
        )

    def generate(
        self,
        prompt: str | None = None,
        source_image: Image.Image | None = None,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Generate an image with International Gothic / StyleGAN artifacts.

        Args:
            prompt: Text prompt (subject description). Will be enhanced with
                   International Gothic styling.
            source_image: Optional source image (if provided, skips generation)
            control: Artifact control parameters
            **era_params: porcelain_smoothness, hair_bleeding, eye_tracking_error,
                         asymmetric_accessories, background_disconnect, gold_background,
                         tempera_texture, courtly_palette, decorative_precision,
                         emotional_vacancy, inference_steps, guidance_scale

        Returns:
            Generated portrait with StyleGAN/International Gothic artifacts
        """
        control = control or ArtifactControl()
        params = {**self.get_default_params(), **era_params}

        # Generation parameters
        num_steps = params.get("inference_steps", 25)
        guidance = params.get("guidance_scale", 7.5)

        # Generate or use source image
        if source_image is None:
            if prompt is None:
                prompt = "noble lady"  # Default subject

            self.ensure_loaded()

            # Build full prompt with International Gothic styling
            full_prompt = self._build_prompt(prompt)

            generator = None
            if control.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(control.seed)

            width = era_params.get("width", 512)
            height = era_params.get("height", 768)  # Portrait orientation

            result = self._pipe(
                prompt=full_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
            )
            img = result.images[0]
        else:
            img = source_image.copy()

        original = img.copy()

        # Get effect strengths scaled by intensity
        intensity = control.intensity
        default_params = self.get_default_params()

        def get_effect_strength(param_name: str) -> float:
            """Get effect strength - use raw value if explicitly set, else scale by intensity."""
            value = params[param_name]
            default = default_params[param_name]
            if value != default:
                return value
            else:
                return value * intensity * 1.5

        # Set seed for reproducibility
        if control.seed is not None:
            np.random.seed(control.seed)

        # Apply StyleGAN-like artifacts first
        img = self._apply_porcelain_smoothness(img, get_effect_strength("porcelain_smoothness"))
        img = self._apply_hair_bleeding(img, get_effect_strength("hair_bleeding"))
        img = self._apply_eye_tracking_error(img, get_effect_strength("eye_tracking_error"))
        img = self._apply_asymmetric_accessories(img, get_effect_strength("asymmetric_accessories"))
        img = self._apply_background_disconnect(img, get_effect_strength("background_disconnect"))

        # Apply International Gothic stylization
        img = self._apply_gold_background(img, get_effect_strength("gold_background"), params["gold_style"])
        img = self._apply_tempera_texture(img, get_effect_strength("tempera_texture"))
        img = self._apply_courtly_palette(img, get_effect_strength("courtly_palette"))
        img = self._apply_decorative_precision(img, get_effect_strength("decorative_precision"))
        img = self._apply_emotional_vacancy(img, get_effect_strength("emotional_vacancy"))

        # Apply final sharpening to restore StyleGAN's photographic crispness
        img = self._apply_stylegan_sharpness(img, intensity)

        # Apply placement mask
        img = control.apply_mask(img, original)

        # Upscale if requested
        upscale = era_params.get("upscale", 1)
        if upscale > 1:
            new_size = (img.width * upscale, img.height * upscale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _apply_porcelain_smoothness(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply uncanny over-smoothing to skin areas.

        StyleGAN faces have that plastic, over-processed quality where skin
        looks impossibly smooth. Like a beauty filter cranked too high.
        Edge-aware to preserve eyes, lips, and jewelry detail.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        # Detect skin-tone regions for targeted smoothing
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        skin_mask = (
            (r > 80) & (r < 255) &
            (g > 40) & (g < 200) &
            (b > 20) & (b < 180) &
            (r > g) & (r > b)
        ).astype(np.float32)
        skin_mask = ndimage.gaussian_filter(skin_mask, sigma=5)

        # Bilateral-like filter using multiple gaussian passes
        smooth = ndimage.gaussian_filter(arr, sigma=(strength * 4, strength * 4, 0))

        # Edge detection to preserve detail
        gray = np.mean(arr, axis=2)
        edges = ndimage.sobel(gray)
        edges = ndimage.gaussian_filter(edges, sigma=2)
        edge_mask = np.clip(edges / (edges.max() + 1e-8) * 3, 0, 1)

        # Combine masks: smooth skin but preserve edges
        smooth_mask = skin_mask * (1 - edge_mask)
        smooth_mask = smooth_mask[:, :, np.newaxis]

        # Blend
        result = arr * (1 - smooth_mask * strength * 0.8) + smooth * (smooth_mask * strength * 0.8)

        # Add slight plastic sheen (boost highlights in skin areas)
        highlights = np.clip((arr - 180) / 75, 0, 1)
        result = result + highlights * skin_mask[:, :, np.newaxis] * strength * 15

        img_out = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

        # Slight contrast boost for HDR skin look
        enhancer = ImageEnhance.Contrast(img_out)
        img_out = enhancer.enhance(1 + strength * 0.15)

        return img_out

    def _apply_hair_bleeding(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply hair bleeding into background effect.

        StyleGAN often couldn't cleanly separate hair from background.
        Hair edges dissolve, blend, leak into the surrounding area.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Detect dark regions (likely hair) - top half of image weighted
        gray = np.mean(arr, axis=2)
        dark_mask = 1 - np.clip(gray / 100, 0, 1)

        # Weight toward top of image where hair usually is
        y_weight = np.linspace(1, 0.3, h)[:, np.newaxis]
        hair_mask = dark_mask * y_weight
        hair_mask = ndimage.gaussian_filter(hair_mask, sigma=5)

        # Find edges of hair region
        hair_edges = ndimage.sobel(hair_mask)
        hair_edges = ndimage.gaussian_filter(hair_edges, sigma=3)
        hair_edges = np.clip(hair_edges * 5, 0, 1)

        # Create bleeding effect - smear colors at hair boundaries
        bleed_distance = int(strength * 8) + 2

        # Multi-directional smear
        smeared = arr.copy()
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
        for dy, dx in directions:
            shifted = ndimage.shift(arr, [dy * bleed_distance, dx * bleed_distance, 0], mode='reflect')
            weight = hair_edges * strength * 0.15
            weight = weight[:, :, np.newaxis]
            smeared = smeared * (1 - weight) + shifted * weight

        # Color averaging at hair-background boundary
        local_avg = ndimage.uniform_filter(arr, size=(int(strength * 6) + 1, int(strength * 6) + 1, 1))
        hair_edges_3d = hair_edges[:, :, np.newaxis]
        result = smeared * (1 - hair_edges_3d * strength * 0.3) + local_avg * (hair_edges_3d * strength * 0.3)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_eye_tracking_error(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply subtle eye misalignment.

        StyleGAN eyes often don't quite track together. One eye slightly higher,
        slight rotation difference. Subtle enough to unsettle, not cartoonish.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Detect eye-like regions (dark spots in the upper-middle area)
        gray = np.mean(arr, axis=2)

        # Focus on upper half (eye region)
        eye_region = np.zeros_like(gray)
        eye_region[h//4:h//2, w//4:3*w//4] = 1
        eye_region = ndimage.gaussian_filter(eye_region, sigma=20)

        # Find dark spots (eyes are darker than surroundings)
        local_mean = ndimage.uniform_filter(gray, size=25)
        local_dark = local_mean - gray
        eye_candidates = np.clip(local_dark / (local_dark.max() + 1e-8) * 2, 0, 1) * eye_region

        # Look for bilateral symmetry
        left_shift = ndimage.shift(eye_candidates, [0, 25], mode='constant')
        right_shift = ndimage.shift(eye_candidates, [0, -25], mode='constant')
        bilateral = np.minimum(left_shift, right_shift)
        bilateral = ndimage.gaussian_filter(bilateral, sigma=8)

        # Apply asymmetric vertical shift
        drift_amount = strength * 8
        center_x = w // 2

        result = arr.copy()
        for c in range(3):
            # Left side shifts up, right side shifts down
            left_shifted = ndimage.shift(arr[:, :, c], [-drift_amount, 0], mode='reflect')
            right_shifted = ndimage.shift(arr[:, :, c], [drift_amount, 0], mode='reflect')

            # Create position-based blend
            x_coord = np.arange(w)[np.newaxis, :]
            left_mask = (x_coord < center_x).astype(np.float32)
            right_mask = 1 - left_mask

            shifted = left_shifted * left_mask + right_shifted * right_mask

            # Only apply in eye regions
            result[:, :, c] = arr[:, :, c] * (1 - bilateral * strength * 0.7) + shifted * (bilateral * strength * 0.7)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_asymmetric_accessories(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply asymmetric distortion to accessories.

        StyleGAN's famous tell: earrings that don't match, glasses slightly warped.
        Apply different transformations to left and right sides.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Focus on ear regions (sides of face, upper half)
        left_ear = np.zeros((h, w), dtype=np.float32)
        right_ear = np.zeros((h, w), dtype=np.float32)

        left_ear[h//4:h//2, :w//4] = 1
        right_ear[h//4:h//2, 3*w//4:] = 1

        left_ear = ndimage.gaussian_filter(left_ear, sigma=15)
        right_ear = ndimage.gaussian_filter(right_ear, sigma=15)

        # Detect high-detail regions (likely jewelry/accessories)
        gray = np.mean(arr, axis=2)
        detail = ndimage.sobel(gray)
        detail = ndimage.gaussian_filter(detail, sigma=2)
        detail_mask = np.clip(detail / (detail.max() + 1e-8) * 2, 0, 1)

        # Accessory regions are detailed areas in ear zones
        left_acc = left_ear * detail_mask
        right_acc = right_ear * detail_mask

        result = arr.copy()

        # Apply different distortions to each side
        # Left side: slight scale and brightness change
        if np.any(left_acc > 0.1):
            # Brightness asymmetry
            result[:, :w//3, :] = result[:, :w//3, :] * (1 + left_acc[:, :w//3, np.newaxis] * strength * 0.3)

            # Color tint asymmetry (slightly warmer on one side)
            result[:, :w//3, 0] = result[:, :w//3, 0] + left_acc[:, :w//3] * strength * 20

        # Right side: slight geometric distortion
        if np.any(right_acc > 0.1):
            # Create subtle warp
            x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
            warp_x = x_coords + np.sin(y_coords / h * np.pi * 2) * right_acc * strength * 10
            warp_y = y_coords + np.cos(x_coords / w * np.pi * 2) * right_acc * strength * 5

            warp_x = np.clip(warp_x, 0, w - 1)
            warp_y = np.clip(warp_y, 0, h - 1)

            for c in range(3):
                warped = ndimage.map_coordinates(arr[:, :, c], [warp_y, warp_x], order=1, mode='reflect')
                # Only apply in right accessory region
                result[:, :, c] = result[:, :, c] * (1 - right_acc * strength * 0.5) + warped * (right_acc * strength * 0.5)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_background_disconnect(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply background disconnect effect.

        StyleGAN figures are sharp, but backgrounds become confused.
        Sharp but incoherent - the latent space "runs out" of training data.
        Creates that "figure doesn't belong here" feeling.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Detect foreground (high detail, skin tones, center-weighted)
        gray = np.mean(arr, axis=2)
        detail = ndimage.sobel(gray)
        detail = ndimage.gaussian_filter(detail, sigma=3)
        detail_mask = np.clip(detail / (detail.max() + 1e-8) * 2, 0, 1)

        # Center weight (figures usually centered)
        cy, cx = h // 2, w // 2
        y_grid, x_grid = np.ogrid[:h, :w]
        center_dist = np.sqrt(((y_grid - cy) / h) ** 2 + ((x_grid - cx) / w) ** 2)
        center_weight = 1 - np.clip(center_dist * 1.5, 0, 1)

        # Foreground mask
        fg_mask = np.clip(detail_mask * 2 + center_weight * 0.5, 0, 1)
        fg_mask = ndimage.gaussian_filter(fg_mask, sigma=8)
        bg_mask = 1 - fg_mask

        # Apply confusion to background
        # 1. Color channel bleeding
        shifted_r = ndimage.shift(arr[:, :, 0], [strength * 4, strength * 3], mode='reflect')
        shifted_b = ndimage.shift(arr[:, :, 2], [-strength * 3, strength * 4], mode='reflect')

        result = arr.copy()
        result[:, :, 0] = arr[:, :, 0] * (1 - bg_mask * strength * 0.4) + shifted_r * (bg_mask * strength * 0.4)
        result[:, :, 2] = arr[:, :, 2] * (1 - bg_mask * strength * 0.4) + shifted_b * (bg_mask * strength * 0.4)

        # 2. Local color averaging (sharp but meaningless)
        local_avg = ndimage.uniform_filter(arr, size=(int(strength * 10) + 1, int(strength * 10) + 1, 1))
        bg_mask_3d = bg_mask[:, :, np.newaxis]
        result = result * (1 - bg_mask_3d * strength * 0.3) + local_avg * (bg_mask_3d * strength * 0.3)

        # 3. Texture confusion - slight patchwork
        patch_size = int(strength * 20) + 8
        patches = np.zeros_like(arr)
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                if np.random.random() < 0.3:  # Some patches get rotated colors
                    patches[y:y_end, x:x_end] = np.roll(arr[y:y_end, x:x_end], 1, axis=2)
                else:
                    patches[y:y_end, x:x_end] = arr[y:y_end, x:x_end]

        result = result * (1 - bg_mask_3d * strength * 0.2) + patches * (bg_mask_3d * strength * 0.2)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_gold_background(self, img: Image.Image, strength: float, style: str) -> Image.Image:
        """Apply gold leaf background effect.

        International Gothic paintings often had gold leaf backgrounds.
        Transcendent void that separates the figure from earthly space.
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

        # Create gold based on style
        if style == "metallic":
            # Rich gold leaf
            gold_base = np.array([[[218, 180, 80]]], dtype=np.float32)
            gold_highlight = np.array([[[255, 223, 120]]], dtype=np.float32)

            # Gold leaf texture variation
            small_h, small_w = max(1, h // 8), max(1, w // 8)
            texture = np.random.rand(small_h, small_w)
            texture = np.array(Image.fromarray((texture * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)) / 255.0
            texture = ndimage.gaussian_filter(texture, sigma=2)

            # Add fine cracks/variation typical of gold leaf
            cracks = np.random.rand(h, w) > 0.995
            cracks = ndimage.maximum_filter(cracks.astype(np.float32), size=2)
            cracks = ndimage.gaussian_filter(cracks, sigma=1)

            texture = texture[:, :, np.newaxis]
            gold = gold_base * (1 - texture * 0.25) + gold_highlight * (texture * 0.25)
            gold = gold * (1 - cracks[:, :, np.newaxis] * 0.3)
        else:
            # Symbolic warm tones
            base_warm = np.array([[[200, 165, 95]]], dtype=np.float32)

            small_h, small_w = max(1, h // 16), max(1, w // 16)
            variation = np.random.rand(small_h, small_w, 3)
            var_resized = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                var_resized[:, :, c] = np.array(Image.fromarray((variation[:, :, c] * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)) / 255.0
            variation = ndimage.gaussian_filter(var_resized, sigma=(6, 6, 0))

            gold = base_warm + variation * 35 - 17

        # Blend gold into background
        bg_mask_3d = bg_mask[:, :, np.newaxis]
        result = arr * (1 - bg_mask_3d * strength) + gold * (bg_mask_3d * strength)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_tempera_texture(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply egg tempera surface texture.

        International Gothic paintings were done in egg tempera on wood panels.
        Fine grain, slightly matte, warm undertones from the egg yolk medium.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Fine grain noise (tempera has subtle surface texture)
        grain = np.random.randn(h, w) * strength * 8
        grain = ndimage.gaussian_filter(grain, sigma=0.5)
        grain = grain[:, :, np.newaxis]

        result = arr + grain

        # Warm color cast from egg yolk medium
        warm_shift = np.array([[[5, 3, -5]]], dtype=np.float32) * strength
        result = result + warm_shift

        # Slight matte finish (reduce specular highlights)
        highlights = np.clip((arr - 200) / 55, 0, 1)
        result = result - highlights * strength * 20

        # Subtle wood panel texture (very fine horizontal grain)
        wood_grain = np.sin(np.arange(h)[:, np.newaxis] * 0.5) * strength * 3
        result = result + wood_grain[:, :, np.newaxis]

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_courtly_palette(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply International Gothic courtly color palette.

        Rich ultramarine blues, vermillion reds, gold - the expensive pigments
        of courtly patronage. Skin tones slightly desaturated, costume saturated.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        # Detect skin vs non-skin regions
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        skin_mask = (
            (r > 80) & (r < 255) &
            (g > 40) & (g < 200) &
            (b > 20) & (b < 180) &
            (r > g) & (r > b)
        ).astype(np.float32)
        skin_mask = ndimage.gaussian_filter(skin_mask, sigma=5)

        # Desaturate skin slightly (courtly pallor)
        skin_gray = np.mean(arr, axis=2, keepdims=True)
        skin_desat = arr * (1 - strength * 0.2) + skin_gray * (strength * 0.2)

        # Boost saturation in non-skin areas (rich robes)
        costume_mask = 1 - skin_mask
        costume_gray = np.mean(arr, axis=2, keepdims=True)
        costume_sat = costume_gray + (arr - costume_gray) * (1 + strength * 0.4)

        # Combine
        skin_mask_3d = skin_mask[:, :, np.newaxis]
        result = skin_desat * skin_mask_3d + costume_sat * (1 - skin_mask_3d)

        # Push blues toward ultramarine (deeper, more purple)
        blue_areas = (b > r) & (b > g)
        blue_mask = blue_areas.astype(np.float32) * (1 - skin_mask)
        blue_mask = ndimage.gaussian_filter(blue_mask, sigma=3)

        result[:, :, 0] = result[:, :, 0] + blue_mask * strength * 20  # Add red to blue
        result[:, :, 2] = result[:, :, 2] + blue_mask * strength * 15  # Boost blue

        # Push reds toward vermillion (more orange)
        red_areas = (r > b) & (r > g * 1.2)
        red_mask = red_areas.astype(np.float32) * (1 - skin_mask)
        red_mask = ndimage.gaussian_filter(red_mask, sigma=3)

        result[:, :, 0] = result[:, :, 0] + red_mask * strength * 15
        result[:, :, 1] = result[:, :, 1] + red_mask * strength * 5  # Slight orange

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_decorative_precision(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply decorative precision effect.

        In International Gothic, jewelry and decorative elements were rendered
        with more precision than faces. The gold thread, the gemstones, the
        intricate patterns - all sharper than the idealized features.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)

        # Detect high-frequency decorative areas (jewelry, fabric patterns)
        gray = np.mean(arr, axis=2)
        high_freq = ndimage.laplace(gray)
        high_freq = np.abs(high_freq)
        decorative_mask = np.clip(high_freq / (high_freq.max() + 1e-8) * 3, 0, 1)
        decorative_mask = ndimage.gaussian_filter(decorative_mask, sigma=2)

        # Detect skin areas to exclude
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        skin_mask = (
            (r > 80) & (r < 255) &
            (g > 40) & (g < 200) &
            (b > 20) & (b < 180) &
            (r > g) & (r > b)
        ).astype(np.float32)
        skin_mask = ndimage.gaussian_filter(skin_mask, sigma=5)

        # Decorative = high detail but not skin
        decorative_mask = decorative_mask * (1 - skin_mask)
        decorative_mask = decorative_mask[:, :, np.newaxis]

        # Sharpen decorative areas
        sharpened = arr + ndimage.laplace(arr) * (-strength * 0.5)

        # Slightly blur skin areas
        skin_blur = ndimage.gaussian_filter(arr, sigma=(strength * 2, strength * 2, 0))
        skin_mask_3d = skin_mask[:, :, np.newaxis]

        # Combine: sharp decorations, soft skin
        result = arr.copy()
        result = result * (1 - decorative_mask * strength * 0.6) + sharpened * (decorative_mask * strength * 0.6)
        result = result * (1 - skin_mask_3d * strength * 0.3) + skin_blur * (skin_mask_3d * strength * 0.3)

        # Boost contrast in decorative areas
        result = result + (result - 128) * decorative_mask * strength * 0.2

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_emotional_vacancy(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply emotional vacancy effect.

        International Gothic faces are technically accomplished but emotionally
        vacant. Flatten the midtones in expression areas, reduce contrast around
        mouth and brows. That "technically correct but vacant" look.
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        # Detect face region (center upper area, skin tones)
        face_region = np.zeros((h, w), dtype=np.float32)
        face_region[h//6:2*h//3, w//4:3*w//4] = 1
        face_region = ndimage.gaussian_filter(face_region, sigma=30)

        # Skin detection
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        skin_mask = (
            (r > 80) & (r < 255) &
            (g > 40) & (g < 200) &
            (b > 20) & (b < 180) &
            (r > g) & (r > b)
        ).astype(np.float32)
        skin_mask = ndimage.gaussian_filter(skin_mask, sigma=3)

        # Face = skin in face region
        face_mask = face_region * skin_mask
        face_mask = ndimage.gaussian_filter(face_mask, sigma=5)

        # Expression areas: mouth region and brow region
        mouth_region = np.zeros((h, w), dtype=np.float32)
        mouth_region[h//2:2*h//3, w//3:2*w//3] = 1
        mouth_region = ndimage.gaussian_filter(mouth_region, sigma=15)

        brow_region = np.zeros((h, w), dtype=np.float32)
        brow_region[h//4:h//3, w//4:3*w//4] = 1
        brow_region = ndimage.gaussian_filter(brow_region, sigma=15)

        expression_mask = (mouth_region + brow_region) * face_mask
        expression_mask = np.clip(expression_mask, 0, 1)
        expression_mask = expression_mask[:, :, np.newaxis]

        # Flatten midtones in expression areas
        local_mean = ndimage.uniform_filter(arr, size=(15, 15, 1))
        flattened = arr * 0.3 + local_mean * 0.7

        # Reduce contrast
        gray = np.mean(arr, axis=2, keepdims=True)
        low_contrast = gray + (arr - gray) * 0.6

        # Combine
        result = arr * (1 - expression_mask * strength * 0.5) + flattened * (expression_mask * strength * 0.3) + low_contrast * (expression_mask * strength * 0.2)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_stylegan_sharpness(self, img: Image.Image, intensity: float) -> Image.Image:
        """Apply StyleGAN's characteristic photographic sharpness.

        StyleGAN outputs were notably crisp - sharp edges on clothing, hair,
        jewelry. This photographic quality contrasted with the smooth skin
        to create the uncanny valley effect. We restore this after other
        effects may have softened edges.
        """
        arr = np.array(img, dtype=np.float32)

        # Detect skin to preserve smoothness there
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        skin_mask = (
            (r > 80) & (r < 255) &
            (g > 40) & (g < 200) &
            (b > 20) & (b < 180) &
            (r > g) & (r > b)
        ).astype(np.float32)
        skin_mask = ndimage.gaussian_filter(skin_mask, sigma=8)

        # Non-skin areas get sharpened
        sharpen_mask = 1 - skin_mask
        sharpen_mask = sharpen_mask[:, :, np.newaxis]

        # Unsharp mask sharpening
        blurred = ndimage.gaussian_filter(arr, sigma=(1.5, 1.5, 0))
        sharpened = arr + (arr - blurred) * 1.2 * intensity

        # Only sharpen non-skin areas
        result = arr * (1 - sharpen_mask * 0.8) + sharpened * (sharpen_mask * 0.8)

        # Add subtle micro-contrast for that photographic punch
        # Local contrast enhancement
        local_mean = ndimage.uniform_filter(arr, size=(10, 10, 1))
        local_contrast = arr + (arr - local_mean) * 0.15 * intensity * sharpen_mask

        result = result * 0.7 + local_contrast * 0.3

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
