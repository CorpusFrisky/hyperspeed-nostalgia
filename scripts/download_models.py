#!/usr/bin/env python3
"""Download required models for HYPERSPEED NOSTALGIA.

This script downloads and caches the models needed for each era.
Run once before first use.
"""

import argparse
import sys
from pathlib import Path


def download_inception():
    """Download InceptionV3 for DeepDream."""
    print("Downloading InceptionV3 for DeepDream...")
    try:
        from torchvision import models

        # This triggers the download and caching
        models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        print("  InceptionV3 downloaded successfully!")
        return True
    except Exception as e:
        print(f"  Failed to download InceptionV3: {e}")
        return False


def download_vgg19():
    """Download VGG19 (alternative for DeepDream)."""
    print("Downloading VGG19 (alternative DeepDream model)...")
    try:
        from torchvision import models

        models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        print("  VGG19 downloaded successfully!")
        return True
    except Exception as e:
        print(f"  Failed to download VGG19: {e}")
        return False


def download_sd15():
    """Download Stable Diffusion 1.5 for diffusion-based eras."""
    print("Downloading Stable Diffusion 1.5...")
    print("  (This is ~2.5GB and may take a while)")
    try:
        from diffusers import StableDiffusionPipeline
        import torch

        # Just download, don't load into memory
        StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        print("  Stable Diffusion 1.5 downloaded successfully!")
        return True
    except Exception as e:
        print(f"  Failed to download SD 1.5: {e}")
        return False


def verify_mps():
    """Verify MPS (Apple Silicon) support."""
    print("Checking MPS (Apple Silicon) support...")
    try:
        import torch

        if torch.backends.mps.is_available():
            print("  MPS is available!")
            if torch.backends.mps.is_built():
                print("  MPS backend is built and ready.")
            return True
        else:
            print("  MPS not available. Will fall back to CPU.")
            return False
    except Exception as e:
        print(f"  Error checking MPS: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download models for HYPERSPEED NOSTALGIA"
    )
    parser.add_argument(
        "--model",
        choices=["all", "deepdream", "sd15", "vgg19"],
        default="deepdream",
        help="Which models to download",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip MPS verification",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("HYPERSPEED NOSTALGIA - Model Download")
    print("=" * 60)
    print()

    # Verify MPS first
    if not args.skip_verify:
        verify_mps()
        print()

    success = True

    if args.model in ("all", "deepdream"):
        success = download_inception() and success
        print()

    if args.model in ("all", "vgg19"):
        success = download_vgg19() and success
        print()

    if args.model in ("all", "sd15"):
        success = download_sd15() and success
        print()

    print("=" * 60)
    if success:
        print("All requested models downloaded successfully!")
        print()
        print("You can now run:")
        print("  hyperspeed generate --era deepdream --source your_image.jpg")
    else:
        print("Some downloads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
