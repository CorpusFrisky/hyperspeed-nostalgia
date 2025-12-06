"""Image loading, saving, and preprocessing utilities."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(path: str | Path, max_size: int | None = None) -> Image.Image:
    """Load an image from disk.

    Args:
        path: Path to the image file
        max_size: If set, resize largest dimension to this value

    Returns:
        PIL Image in RGB mode
    """
    img = Image.open(path).convert("RGB")

    if max_size is not None:
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img


def save_image(
    img: Image.Image,
    path: str | Path,
    format: str | None = None,
    quality: int = 95,
) -> Path:
    """Save an image to disk.

    Args:
        img: PIL Image to save
        path: Output path
        format: Output format (inferred from extension if None)
        quality: JPEG/WebP quality (1-100)

    Returns:
        Path to saved file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {}
    if format is None:
        format = path.suffix.lower().lstrip(".")

    if format in ("jpg", "jpeg"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    elif format == "webp":
        save_kwargs["quality"] = quality
    elif format == "png":
        save_kwargs["compress_level"] = 6

    img.save(path, **save_kwargs)
    return path


def preprocess_for_model(
    img: Image.Image,
    size: int | tuple[int, int] | None = None,
    normalize: bool = True,
    device: str = "mps",
) -> torch.Tensor:
    """Convert PIL Image to tensor for model input.

    Args:
        img: PIL Image
        size: Target size (int for square, tuple for w,h)
        normalize: Apply ImageNet normalization
        device: Target device

    Returns:
        Tensor of shape (1, 3, H, W)
    """
    if size is not None:
        if isinstance(size, int):
            size = (size, size)
        img = img.resize(size, Image.Resampling.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

    return tensor.to(device)


def tensor_to_image(tensor: torch.Tensor, denormalize: bool = True) -> Image.Image:
    """Convert model output tensor back to PIL Image.

    Args:
        tensor: Tensor of shape (1, 3, H, W) or (3, H, W)
        denormalize: Undo ImageNet normalization

    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.detach().cpu()

    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean

    tensor = torch.clamp(tensor, 0, 1)
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return Image.fromarray(arr)


def create_noise_image(
    size: tuple[int, int],
    seed: int | None = None,
    color_noise: bool = True,
) -> Image.Image:
    """Create a noise image as starting point for some pipelines.

    Args:
        size: (width, height)
        seed: Random seed for reproducibility
        color_noise: If True, RGB noise; if False, grayscale

    Returns:
        PIL Image of noise
    """
    if seed is not None:
        np.random.seed(seed)

    w, h = size
    if color_noise:
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    else:
        gray = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        arr = np.stack([gray, gray, gray], axis=-1)

    return Image.fromarray(arr)


def resize_to_multiple(img: Image.Image, multiple: int = 8) -> Image.Image:
    """Resize image so dimensions are multiples of a given value.

    Many models require dimensions divisible by 8 or 16.

    Args:
        img: PIL Image
        multiple: Dimensions will be rounded down to this multiple

    Returns:
        Resized PIL Image
    """
    w, h = img.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple

    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return img
