# Claude Instructions for aiSlopArtHistory

## Output Directory

**ALWAYS save generated images to `examples/<era>/` subdirectories, NOT `examples/` root or `outputs/`.**

Images are organized by era:
- `examples/impressionism/` - Impressionism era images
- `examples/neoclassicism/` - Neoclassicism era images
- `examples/high_renaissance/` - High Renaissance era images
- `examples/early_renaissance/` - Early Renaissance era images
- `examples/international_gothic/` - International Gothic era images
- `examples/byzantine/` - Byzantine/Early Diffusion era images
- `examples/deepdream/` - DeepDream era images
- `examples/cave/` - Cave painting era images
- `examples/egyptian/` - Egyptian era images
- `examples/reference/` - Reference images (source photos, etc.)

When using `--output`, include the era subdirectory:
```bash
hyperspeed generate "..." --era impressionism --output examples/impressionism/my_image.png
```

The `hyperspeed batch` command defaults to `examples/<era>/` automatically.

## Project Structure

- `src/hyperspeed/` - Main source code
- `examples/<era>/` - Generated example images organized by era
- `*_GUIDE.md` - Era-specific documentation
