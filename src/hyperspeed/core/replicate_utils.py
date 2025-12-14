"""Shared utilities for Replicate API integration.

This module provides common functions for generating images via the Replicate
API using SDXL. Used by multiple eras (High Renaissance, Neoclassicism,
Early Renaissance) for fast cloud-based generation.

Environment:
- Set REPLICATE_API_TOKEN to enable cloud generation
- Falls back gracefully if token not set or API fails
"""

import io
import os
import time
from typing import TYPE_CHECKING
from urllib.request import urlopen

from PIL import Image

if TYPE_CHECKING:
    import replicate.prediction


# Replicate model ID for SDXL
REPLICATE_SDXL_MODEL = "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc"


def generate_via_replicate(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 30,
    guidance: float = 8.0,
    seed: int | None = None,
    scheduler: str = "K_EULER",
) -> Image.Image | None:
    """Generate image via Replicate API. Returns None if unavailable.

    Args:
        prompt: Text prompt for generation
        width: Output width (default 1024)
        height: Output height (default 1024)
        num_steps: Number of inference steps (default 30)
        guidance: CFG guidance scale (default 8.0)
        seed: Random seed for reproducibility
        scheduler: Scheduler type (K_EULER, K_EULER_ANCESTRAL, DDIM, etc.)

    Returns:
        PIL Image if successful, None if API unavailable or error
    """
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
            "scheduler": scheduler,
            "refine": "no_refiner",
            "high_noise_frac": 0.8,
        }

        if seed is not None:
            input_params["seed"] = seed

        print(f"Generating via Replicate API (steps={num_steps}, guidance={guidance}, scheduler={scheduler})...")
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


def submit_replicate_async(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_steps: int = 30,
    guidance: float = 8.0,
    seed: int | None = None,
    scheduler: str = "K_EULER",
) -> "replicate.prediction.Prediction | None":
    """Submit a generation job to Replicate without waiting.

    Args:
        prompt: Text prompt for generation
        width: Output width (default 1024)
        height: Output height (default 1024)
        num_steps: Number of inference steps (default 30)
        guidance: CFG guidance scale (default 8.0)
        seed: Random seed for reproducibility
        scheduler: Scheduler type (K_EULER, K_EULER_ANCESTRAL, DDIM, etc.)

    Returns:
        Prediction object if successful, None if unavailable
    """
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
            "scheduler": scheduler,
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


def download_replicate_result(prediction: "replicate.prediction.Prediction") -> Image.Image | None:
    """Download result from a completed Replicate prediction.

    Args:
        prediction: Replicate prediction object to download from

    Returns:
        PIL Image if successful, None if failed or not ready
    """
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
        poll_interval: Seconds between status checks (default 2.0)
        max_wait: Maximum seconds to wait for all jobs (default 300.0)
        submit_delay: Seconds to wait between job submissions (default 0.5)

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

        prediction = submit_replicate_async(
            prompt=job["prompt"],
            width=job.get("width", 1024),
            height=job.get("height", 1024),
            num_steps=job.get("num_steps", 30),
            guidance=job.get("guidance", 8.0),
            seed=job.get("seed"),
            scheduler=job.get("scheduler", "K_EULER"),
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
                img = download_replicate_result(prediction)
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
