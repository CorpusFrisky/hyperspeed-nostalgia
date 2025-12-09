"""Neoclassicism Era (1760-1850) / Photorealistic AI Era (2023-2024) parallel.

Idealized, smooth, suspiciously beautiful. The return to classical perfection
meets AI's tendency to make everyone too attractive and surfaces too clean.

The Neoclassical movement saw painters like Jacques-Louis David and Jean-Auguste-
Dominique Ingres return to Greek and Roman ideals: perfect proportions, marble-
smooth skin, heroic poses, theatrical staging. They pursued an idealized beauty
that was suspiciously perfect, anatomically "correct" in ways that real bodies
aren't.

The AI photorealistic era (2023-2024) does the same thing by accident. SDXL,
Midjourney v5-v6, and DALL-E 3 produce people who are suspiciously attractive,
surfaces that are too clean, proportions that are "idealized" rather than real.

Key works:
- Jacques-Louis David's Oath of the Horatii (1784): idealized bodies, theatrical staging
- Ingres's La Grande Odalisque (1814): anatomically "incorrect" (extra vertebrae)
  but aesthetically smooth

Technical approach:
1. Generate image with SDXL via Replicate API (fast) or local (fallback)
2. Apply characteristic Neoclassicism / photorealistic AI tells locally:
   - Marble-smooth skin (frequency-selective smoothing)
   - Golden ratio symmetry (facial symmetry enhancement)
   - Surface flattening (non-skin texture reduction)
   - Heroic staging (subtle compositional centering)
   - Anatomical liberty (Ingres-style proportion distortion)
   - Classical palette (muted, restrained colors)

The tell isn't "too epic" (that's High Renaissance / MJ v4).
The tell is "too perfect."

Environment:
- Set REPLICATE_API_TOKEN to use Replicate for fast generation
- Falls back to local SDXL if token not set or API fails
"""

import io
import os
import time
from pathlib import Path
from typing import Any, ClassVar
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

from hyperspeed.core.artifact_control import ArtifactControl
from hyperspeed.eras.base import EraMetadata, EraPipeline, EraRegistry


# Replicate model ID for SDXL
REPLICATE_SDXL_MODEL = "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc"


def _generate_via_replicate(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 30,
    guidance: float = 8.0,
    seed: int | None = None,
) -> Image.Image | None:
    """Generate image via Replicate API. Returns None if unavailable."""
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return None

    try:
        import replicate
    except ImportError:
        print("Replicate package not installed. Run: pip install replicate")
        return None

    try:
        input_params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance,
            "scheduler": "K_EULER",
            "refine": "no_refiner",
            "high_noise_frac": 0.8,
        }

        if seed is not None:
            input_params["seed"] = seed

        print("Generating via Replicate API...")
        output = replicate.run(REPLICATE_SDXL_MODEL, input=input_params)

        # Output is a list of FileOutput objects or URLs
        if output and len(output) > 0:
            result = output[0]
            # Handle both FileOutput objects and plain URL strings
            image_url = str(result) if hasattr(result, '__str__') else result
            print("Downloading from Replicate...")
            with urlopen(image_url) as response:
                image_data = response.read()
            return Image.open(io.BytesIO(image_data)).convert("RGB")

    except Exception as e:
        print(f"Replicate API error: {e}")
        return None

    return None


def _submit_replicate_async(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 30,
    guidance: float = 8.0,
    seed: int | None = None,
) -> "replicate.prediction.Prediction | None":
    """Submit a generation job to Replicate without waiting. Returns prediction object."""
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        return None

    try:
        import replicate
    except ImportError:
        print("Replicate package not installed. Run: pip install replicate")
        return None

    try:
        input_params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance,
            "scheduler": "K_EULER",
            "refine": "no_refiner",
            "high_noise_frac": 0.8,
        }

        if seed is not None:
            input_params["seed"] = seed

        # Parse model string into model and version
        model_name, version_id = REPLICATE_SDXL_MODEL.split(":")

        # Create prediction without waiting - use version parameter
        prediction = replicate.predictions.create(
            version=version_id,
            input=input_params,
        )
        return prediction

    except Exception as e:
        print(f"Replicate API error submitting job: {e}")
        return None


def _download_replicate_result(prediction: "replicate.prediction.Prediction") -> Image.Image | None:
    """Download result from a completed Replicate prediction."""
    try:
        # Reload to get latest status
        prediction.reload()

        if prediction.status == "succeeded" and prediction.output:
            result = prediction.output[0]
            image_url = str(result) if hasattr(result, '__str__') else result
            with urlopen(image_url) as response:
                image_data = response.read()
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        elif prediction.status == "failed":
            print(f"Prediction failed: {prediction.error}")
            return None
        else:
            return None

    except Exception as e:
        print(f"Error downloading result: {e}")
        return None


def batch_generate_replicate(
    jobs: list[dict],
    poll_interval: float = 2.0,
    max_wait: float = 300.0,
    submit_delay: float = 0.5,
) -> list[tuple[dict, Image.Image | None]]:
    """Submit multiple jobs to Replicate in parallel, return results as they complete.

    This function:
    1. Submits ALL jobs to Replicate (with small delay to avoid rate limits)
    2. Polls for completion
    3. Returns results in completion order

    Args:
        jobs: List of dicts with keys: prompt, width, height, num_steps, guidance, seed, output_path, era_params
        poll_interval: Seconds between status checks
        max_wait: Maximum seconds to wait for all jobs
        submit_delay: Seconds to wait between job submissions (to avoid rate limits)

    Returns:
        List of (job_dict, Image or None) tuples in completion order
    """
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        print("No REPLICATE_API_TOKEN set. Cannot batch generate.")
        return [(job, None) for job in jobs]

    try:
        import replicate
    except ImportError:
        print("Replicate package not installed. Run: pip install replicate")
        return [(job, None) for job in jobs]

    # Submit all jobs with small delays to avoid rate limiting
    print(f"Submitting {len(jobs)} jobs to Replicate...")
    pending: list[tuple[dict, replicate.prediction.Prediction]] = []
    failed_jobs: list[dict] = []

    for i, job in enumerate(jobs):
        # Small delay between submissions to avoid rate limits
        if i > 0:
            time.sleep(submit_delay)

        prediction = _submit_replicate_async(
            prompt=job["prompt"],
            width=job.get("width", 1024),
            height=job.get("height", 1024),
            num_steps=job.get("num_steps", 30),
            guidance=job.get("guidance", 8.0),
            seed=job.get("seed"),
        )
        if prediction:
            pending.append((job, prediction))
            print(f"  [{i+1}/{len(jobs)}] Submitted: {job['prompt'][:50]}...")
        else:
            failed_jobs.append(job)
            print(f"  [{i+1}/{len(jobs)}] FAILED to submit: {job['prompt'][:50]}...")

    if not pending:
        return [(job, None) for job in jobs]

    # Poll for completion
    print(f"\nWaiting for {len(pending)} jobs to complete...")
    results: list[tuple[dict, Image.Image | None]] = []
    # Include already-failed jobs
    for job in failed_jobs:
        results.append((job, None))
    start_time = time.time()

    while pending and (time.time() - start_time) < max_wait:
        still_pending = []

        for job, prediction in pending:
            prediction.reload()

            if prediction.status == "succeeded":
                print(f"  Completed: {job['prompt'][:50]}...")
                img = _download_replicate_result(prediction)
                results.append((job, img))
            elif prediction.status == "failed":
                print(f"  FAILED: {job['prompt'][:50]}... - {prediction.error}")
                results.append((job, None))
            elif prediction.status == "canceled":
                print(f"  CANCELED: {job['prompt'][:50]}...")
                results.append((job, None))
            else:
                # Still processing
                still_pending.append((job, prediction))

        pending = still_pending

        if pending:
            elapsed = time.time() - start_time
            print(f"  [{len(results)}/{len(jobs)} done, {len(pending)} pending, {elapsed:.0f}s elapsed]")
            time.sleep(poll_interval)

    # Handle any remaining timeouts
    for job, prediction in pending:
        print(f"  TIMEOUT: {job['prompt'][:50]}...")
        results.append((job, None))

    return results


@EraRegistry.register
class NeoclassicismPipeline(EraPipeline):
    """Neoclassicism Era: The suspicious perfection of photorealistic AI.

    This pipeline generates images with the characteristic "too perfect"
    quality of 2023-2024 photorealistic AI, mapped onto the idealized
    aesthetics of Neoclassical painters like David and Ingres.

    Unlike High Renaissance (over-dramatization), Neoclassicism is about
    over-idealization. The tell isn't "too epic" - it's "too perfect."

    Controls:
        marble_smooth_skin: Frequency-selective skin smoothing (0-1)
        golden_ratio_symmetry: Facial symmetry enhancement (0-1)
        surface_flattening: Non-skin texture flattening (0-1)
        heroic_staging: Compositional centering/staging (0-1)
        anatomical_liberty: Ingres-style proportion distortion (0-1)
        classical_palette: Muted Neoclassical color harmony (0-1)
    """

    metadata: ClassVar[EraMetadata] = EraMetadata(
        name="Neoclassicism",
        art_historical_parallel="Neoclassicism (1760-1850)",
        time_period="2023-2024",
        description=(
            "Idealized, smooth, suspiciously beautiful. "
            "The return to classical perfection meets AI's tendency to "
            "make everyone too attractive and surfaces too clean."
        ),
        characteristic_artifacts=[
            "Marble-smooth skin (frequency-selective smoothing)",
            "Suspiciously attractive (golden ratio symmetry)",
            "Too-clean surfaces (texture flattening)",
            "Heroic poses (theatrical staging)",
            "Anatomical liberty (Ingres-style proportion distortion)",
            "Classical palette (muted, restrained colors)",
        ],
    )

    def __init__(self, model_path: Path | None = None, device: str = "mps"):
        super().__init__(model_path, device)
        self._pipe = None

    def load_model(self) -> None:
        """Load SDXL for generation - high quality base for photorealistic output."""
        if self._pipe is not None:
            return

        from diffusers import StableDiffusionXLPipeline

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        # Use SDXL for photorealistic quality
        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32,
            use_safetensors=True,
            variant="fp16",
        )
        self._pipe = self._pipe.to(self.device)
        self._pipe.enable_attention_slicing()

        self._model = True

    def get_default_params(self) -> dict[str, Any]:
        return {
            "marble_smooth_skin": 0.7,       # THE signature effect
            "golden_ratio_symmetry": 0.4,    # Keep subtle
            "surface_flattening": 0.5,
            "heroic_staging": 0.3,           # Very subtle
            "anatomical_liberty": 0.5,
            "classical_palette": 0.6,
            "inference_steps": 30,
            "guidance_scale": 8.0,           # Slightly lower for natural look
        }

    def generate(
        self,
        prompt: str | None = None,
        source_image: Image.Image | None = None,
        control: ArtifactControl | None = None,
        **era_params: Any,
    ) -> Image.Image:
        """Generate an image with Neoclassicism / photorealistic AI tells.

        Args:
            prompt: Text prompt for generation. Required.
            source_image: Not used (txt2img only for this era)
            control: Artifact control parameters
            **era_params: marble_smooth_skin, golden_ratio_symmetry, etc.
                         use_local: Force local generation (skip Replicate)

        Returns:
            Generated image with Neoclassicism / photorealistic tells
        """
        control = control or ArtifactControl()
        params = {**self.get_default_params(), **era_params}

        if prompt is None:
            raise ValueError("Neoclassicism pipeline requires a prompt")

        # Generation parameters
        num_steps = params.get("inference_steps", 30)
        guidance = params.get("guidance_scale", 8.0)
        use_local = era_params.get("use_local", False)

        width = era_params.get("width", 1024)
        height = era_params.get("height", 1024)

        # Ensure dimensions are multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        img = None

        # Try Replicate first (much faster than local SDXL)
        if not use_local:
            img = _generate_via_replicate(
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

        marble_smooth_skin = get_effect_strength("marble_smooth_skin")
        golden_ratio_symmetry = get_effect_strength("golden_ratio_symmetry")
        surface_flattening = get_effect_strength("surface_flattening")
        heroic_staging = get_effect_strength("heroic_staging")
        anatomical_liberty = get_effect_strength("anatomical_liberty")
        classical_palette = get_effect_strength("classical_palette")

        # Apply Neoclassicism tells in sequence
        # Order: palette first, then skin/face, then body, then composition
        img = self._apply_classical_palette(img, classical_palette)
        img = self._apply_marble_smooth_skin(img, marble_smooth_skin)
        img = self._apply_golden_ratio_symmetry(img, golden_ratio_symmetry)
        img = self._apply_anatomical_liberty(img, anatomical_liberty)
        img = self._apply_surface_flattening(img, surface_flattening)
        img = self._apply_heroic_staging(img, heroic_staging)

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
        """Apply Neoclassicism / photorealistic AI tells to an existing image.

        This is the post-processing step only. Use this when you already have
        a base image (e.g., from batch Replicate generation) and want to apply
        the characteristic tells without re-generating.

        Args:
            img: Base image to apply tells to
            control: Artifact control parameters
            **era_params: marble_smooth_skin, golden_ratio_symmetry, etc.

        Returns:
            Image with Neoclassicism / photorealistic tells applied
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

        marble_smooth_skin = get_effect_strength("marble_smooth_skin")
        golden_ratio_symmetry = get_effect_strength("golden_ratio_symmetry")
        surface_flattening = get_effect_strength("surface_flattening")
        heroic_staging = get_effect_strength("heroic_staging")
        anatomical_liberty = get_effect_strength("anatomical_liberty")
        classical_palette = get_effect_strength("classical_palette")

        # Apply Neoclassicism tells in sequence
        img = self._apply_classical_palette(img, classical_palette)
        img = self._apply_marble_smooth_skin(img, marble_smooth_skin)
        img = self._apply_golden_ratio_symmetry(img, golden_ratio_symmetry)
        img = self._apply_anatomical_liberty(img, anatomical_liberty)
        img = self._apply_surface_flattening(img, surface_flattening)
        img = self._apply_heroic_staging(img, heroic_staging)

        # Final blend with original based on placement
        img = control.apply_mask(original, img)

        # Upscale if requested
        upscale = era_params.get("upscale", 1)
        if upscale > 1:
            new_size = (img.width * upscale, img.height * upscale)
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        return img

    def _apply_marble_smooth_skin(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply marble-smooth skin effect.

        Unlike the porcelain smoothness of International Gothic (uniform),
        this is FREQUENCY-SELECTIVE: preserve some texture (pores) while
        removing all irregularities (blemishes, texture variation). The
        result looks like skin photographed with expensive equipment AND
        retouched, but there's no actual human texture. Just... marble.

        The AI photorealistic tell: skin that's TOO smooth, TOO even, but
        still has just enough detail to not look obviously fake.

        Technique:
        - Detect skin regions
        - Apply bilateral filter (edge-preserving smoothing)
        - Multi-pass averaging for that airbrushed look
        - Add subtle luminous quality (waxy sheen)
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        result = arr.copy()

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # === SKIN DETECTION ===
        # Broad skin detection to catch faces and bodies
        skin_mask = (
            (r > 60) & (g > 40) & (b > 20) &
            (r > g * 0.8) & (g > b * 0.7) &
            (r - b > 5) & (r - b < 150) &
            (np.abs(r - g) < 80)
        ).astype(np.float32)

        # Expand and smooth the mask
        skin_mask = ndimage.gaussian_filter(skin_mask, sigma=15)
        skin_mask = np.clip(skin_mask * 1.8, 0, 1)

        # === BILATERAL-LIKE SMOOTHING ===
        # Multiple passes of gaussian blur for marble effect
        smooth_fine = ndimage.gaussian_filter(arr, sigma=[2, 2, 0])
        smooth_medium = ndimage.gaussian_filter(arr, sigma=[5, 5, 0])
        smooth_heavy = ndimage.gaussian_filter(arr, sigma=[10, 10, 0])

        # Blend smoothed versions for marble-like quality
        marble_smooth = (
            smooth_fine * 0.2 +
            smooth_medium * 0.4 +
            smooth_heavy * 0.4
        )

        # === PRESERVE SOME HIGH FREQUENCY (pores, but not texture) ===
        # Extract very fine detail to add back (keeps it from looking like plastic)
        very_fine = arr - ndimage.gaussian_filter(arr, sigma=[1, 1, 0])
        # Only keep the absolute finest detail
        very_fine = very_fine * 0.15

        marble_smooth = marble_smooth + very_fine

        # Apply to skin regions
        skin_blend = skin_mask[:, :, np.newaxis] * strength * 0.85
        result = arr * (1 - skin_blend) + marble_smooth * skin_blend

        # === LUMINOUS QUALITY (waxy sheen) ===
        # Add subtle glow to skin areas
        skin_glow = ndimage.gaussian_filter(skin_mask, sigma=12)
        glow_strength = skin_glow * strength * 25

        # Warm luminosity (marble has that warm inner glow)
        result[:, :, 0] = result[:, :, 0] + glow_strength * 0.9
        result[:, :, 1] = result[:, :, 1] + glow_strength * 0.8
        result[:, :, 2] = result[:, :, 2] + glow_strength * 0.6

        # === EVEN OUT SKIN TONES ===
        # Push toward uniform tone (the over-retouched look)
        skin_pixels = skin_mask > 0.3
        if np.any(skin_pixels):
            skin_mean = np.zeros(3)
            for c in range(3):
                skin_mean[c] = np.mean(result[:, :, c][skin_pixels])

            # Blend toward mean skin tone (subtle)
            evenness = skin_mask[:, :, np.newaxis] * strength * 0.25
            skin_target = np.ones_like(result) * skin_mean.reshape(1, 1, 3)
            result = result * (1 - evenness) + skin_target * evenness

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_golden_ratio_symmetry(self, img: Image.Image, strength: float) -> Image.Image:
        """Enhance facial symmetry toward idealized proportions.

        The AI photorealistic tell: everyone is slightly more symmetrical
        than real humans. Eyes perfectly aligned. Nose perfectly centered.
        Jawlines that would make a plastic surgeon jealous.

        This should be SUBTLE - we're not creating monsters, just making
        normal people suspiciously beautiful.

        Technique:
        - Detect approximate face region (upper-center of image)
        - Mirror-blend left and right halves toward symmetry
        - Keep strength low (0.3-0.5) to avoid uncanny valley
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        result = arr.copy()

        # Face is typically in upper-center portion
        face_top = int(h * 0.05)
        face_bottom = int(h * 0.6)
        face_left = int(w * 0.2)
        face_right = int(w * 0.8)

        # Create face region mask (soft edges)
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Vertical component (face is in upper portion)
        y_norm = (y_coords - face_top) / (face_bottom - face_top + 1)
        y_mask = np.clip(1 - np.abs(y_norm - 0.5) * 2.5, 0, 1)
        y_mask = np.clip(y_mask * 2, 0, 1)  # Boost center

        # Horizontal component (centered)
        x_center = w // 2
        x_dist = np.abs(x_coords - x_center) / (w * 0.4)
        x_mask = np.clip(1 - x_dist, 0, 1)

        # Combined face mask
        face_mask = y_mask * x_mask
        face_mask = ndimage.gaussian_filter(face_mask, sigma=20)

        # === SYMMETRY ENHANCEMENT ===
        # Flip horizontally and blend
        flipped = np.flip(arr, axis=1)

        # Blend toward symmetry (average of original and flipped)
        symmetric = (arr + flipped) * 0.5

        # Apply with face mask and strength
        sym_blend = face_mask[:, :, np.newaxis] * strength * 0.6
        result = arr * (1 - sym_blend) + symmetric * sym_blend

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_surface_flattening(self, img: Image.Image, strength: float) -> Image.Image:
        """Flatten textures on non-skin surfaces.

        AI photorealism in 2023-2024 produces surfaces that are TOO CLEAN.
        Marble floors without dust. Fabrics without wrinkles. Hair without
        flyaways. This is the "stock photo" quality.

        Technique:
        - Detect non-skin regions (inverse of skin mask)
        - Apply texture flattening (reduce local variance)
        - Preserve edges while killing small-scale texture
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        result = arr.copy()

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Detect skin regions (to exclude)
        skin_mask = (
            (r > 60) & (g > 40) & (b > 20) &
            (r > g * 0.8) & (g > b * 0.7) &
            (r - b > 5) & (r - b < 150)
        ).astype(np.float32)
        skin_mask = ndimage.gaussian_filter(skin_mask, sigma=10)

        # Non-skin mask
        non_skin_mask = 1 - skin_mask

        # === LOCAL VARIANCE REDUCTION ===
        # Calculate local mean
        local_mean = ndimage.uniform_filter(arr, size=[8, 8, 1])

        # Blend toward local mean (reduces texture variation)
        texture_reduced = arr * 0.4 + local_mean * 0.6

        # === PRESERVE EDGES ===
        # Detect edges to preserve
        edges_x = np.abs(ndimage.sobel(gray, axis=1))
        edges_y = np.abs(ndimage.sobel(gray, axis=0))
        edges = np.sqrt(edges_x**2 + edges_y**2)
        if edges.max() > 0:
            edges = edges / edges.max()

        # Edge preservation mask
        edge_preserve = ndimage.gaussian_filter(edges, sigma=3)
        edge_preserve = np.clip(edge_preserve * 2, 0, 1)

        # Apply flattening only to non-edge, non-skin areas
        flatten_mask = non_skin_mask * (1 - edge_preserve)
        flatten_blend = flatten_mask[:, :, np.newaxis] * strength * 0.7

        result = arr * (1 - flatten_blend) + texture_reduced * flatten_blend

        # === ADDITIONAL SMOOTHING PASS ===
        # Light gaussian blur on non-skin areas
        smooth = ndimage.gaussian_filter(result, sigma=[2, 2, 0])
        smooth_blend = non_skin_mask[:, :, np.newaxis] * strength * 0.3
        result = result * (1 - smooth_blend) + smooth * smooth_blend

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_heroic_staging(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply subtle compositional centering and theatrical staging.

        Neoclassical paintings have a distinctive "stage" quality - figures
        arranged in deliberate, heroic poses against simplified backgrounds.
        AI photorealism similarly tends to center subjects and simplify
        compositions toward "stock photo" aesthetics.

        Technique:
        - Gentle radial pull toward center
        - BUT more subtle than High Renaissance - this is about "positioning"
        - Slight background simplification at periphery
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
        dist_norm = np.sqrt(dy_norm**2 + dx_norm**2)

        # Pull strength increases with distance from center
        # BUT more subtle than High Renaissance
        pull_factor = dist_norm ** 2 * strength * 0.08

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

        # === PERIPHERAL SIMPLIFICATION ===
        # Slightly blur/simplify the edges
        peripheral_mask = np.clip(dist_norm - 0.5, 0, 0.5) * 2
        peripheral_mask = ndimage.gaussian_filter(peripheral_mask, sigma=30)

        blurred = ndimage.gaussian_filter(result, sigma=[5, 5, 0])
        periph_blend = peripheral_mask[:, :, np.newaxis] * strength * 0.3
        result = result * (1 - periph_blend) + blurred * periph_blend

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_anatomical_liberty(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply Ingres-style anatomical liberties.

        Ingres gave his Odalisque three extra vertebrae because it looked
        better. AI models do the same - torsos too long, necks too elegant,
        proportions that feel "idealized" rather than "real."

        Technique:
        - Detect approximate torso region (mid-height of image)
        - Apply subtle vertical stretch (elongation)
        - This should be VERY subtle (like Ingres - you don't notice until
          you count vertebrae)
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Torso region is roughly middle third of image
        torso_center = h * 0.45  # Slightly above center
        torso_extent = h * 0.25

        # Distance from torso center (normalized)
        y_from_torso = (y_coords - torso_center) / torso_extent
        torso_mask = np.exp(-y_from_torso**2 * 0.5)  # Gaussian falloff

        # Horizontal center bias (elongation should be centered)
        x_center = w // 2
        x_dist = np.abs(x_coords - x_center) / (w * 0.4)
        x_mask = np.clip(1 - x_dist, 0, 1)

        # Combined mask
        stretch_mask = torso_mask * x_mask

        # === VERTICAL STRETCH ===
        # Displace pixels to create slight elongation
        # Pixels above torso center move up, below move down
        stretch_direction = np.sign(y_coords - torso_center)
        stretch_amount = stretch_mask * strength * 8  # Subtle

        # Displacement
        new_y = y_coords - stretch_direction * stretch_amount
        new_y = np.clip(new_y, 0, h - 1)

        result = np.zeros_like(arr)
        for c in range(3):
            result[:, :, c] = ndimage.map_coordinates(
                arr[:, :, c],
                [new_y, x_coords.astype(float)],
                order=1,
                mode='reflect'
            )

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    def _apply_classical_palette(self, img: Image.Image, strength: float) -> Image.Image:
        """Apply Neoclassical color palette.

        Neoclassical paintings have a distinctive color harmony:
        - Warm flesh tones (peachy/marble, not golden)
        - Cool, muted backgrounds (grays, muted blues)
        - Limited saturation (not the HDR of MJ v4)
        - Overall "classical" restraint

        AI photorealism in 2023-2024 has a similar quality - colors are
        "correct" but somehow lack the vibrancy of real life. The "stock
        photo" desaturation.

        Technique:
        - Global saturation reduction (pull toward grayscale)
        - Skin regions: warm neutral shift (peach/marble)
        - Background regions: cool neutral shift (gray-blue)
        - Preserve contrast while reducing color intensity
        """
        if strength < 0.01:
            return img

        arr = np.array(img, dtype=np.float32)
        result = arr.copy()

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        lum_norm = gray / 255.0

        # === GLOBAL DESATURATION ===
        # Pull colors toward gray (the "classical restraint")
        gray_3d = gray[:, :, np.newaxis].repeat(3, axis=2)
        desat_amount = strength * 0.35
        result = result * (1 - desat_amount) + gray_3d * desat_amount

        # === SKIN DETECTION ===
        skin_mask = (
            (r > 60) & (g > 40) & (b > 20) &
            (r > g * 0.8) & (g > b * 0.7) &
            (r - b > 5) & (r - b < 150)
        ).astype(np.float32)
        skin_mask = ndimage.gaussian_filter(skin_mask, sigma=12)

        # Non-skin (background)
        non_skin_mask = 1 - skin_mask

        # === WARM SKIN TONES (peachy marble) ===
        skin_shift = skin_mask * strength * 20
        result[:, :, 0] = result[:, :, 0] + skin_shift * 0.8   # Warm red
        result[:, :, 1] = result[:, :, 1] + skin_shift * 0.5   # Peachy
        result[:, :, 2] = result[:, :, 2] - skin_shift * 0.2   # Reduce blue

        # === COOL BACKGROUND (gray-blue) ===
        bg_shift = non_skin_mask * strength * 15
        result[:, :, 0] = result[:, :, 0] - bg_shift * 0.3  # Reduce red
        result[:, :, 1] = result[:, :, 1] - bg_shift * 0.1  # Slight green reduction
        result[:, :, 2] = result[:, :, 2] + bg_shift * 0.4  # Add blue/cool

        # === CONTRAST PRESERVATION ===
        # Boost contrast slightly to compensate for desaturation
        mean_lum = np.mean(gray)
        contrast_factor = 1 + strength * 0.15
        result = mean_lum + (result - mean_lum) * contrast_factor

        # === SHADOW LIFT (Neoclassical shadows aren't black) ===
        dark_mask = np.clip(1 - lum_norm * 2.5, 0, 1)
        shadow_lift = dark_mask * strength * 20
        result = result + shadow_lift[:, :, np.newaxis] * 0.8

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
