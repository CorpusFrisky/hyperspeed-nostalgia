"""Command-line interface for HYPERSPEED NOSTALGIA."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from hyperspeed.core.artifact_control import ArtifactControl, ArtifactPlacement
from hyperspeed.core.image_utils import load_image, save_image
from hyperspeed.eras.base import EraRegistry

# Import eras to register them
from hyperspeed.eras import deepdream  # noqa: F401
from hyperspeed.eras import early_gan  # noqa: F401
from hyperspeed.eras import early_diffusion  # noqa: F401
from hyperspeed.eras import international_gothic  # noqa: F401
from hyperspeed.eras import early_renaissance  # noqa: F401
from hyperspeed.eras import high_renaissance  # noqa: F401

app = typer.Typer(
    name="hyperspeed",
    help="HYPERSPEED NOSTALGIA: Reclaiming AI artifacts through art historical parallels.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def generate(
    prompt: Annotated[
        Optional[str],
        typer.Argument(help="Text prompt (optional, not all eras use it)"),
    ] = None,
    era: Annotated[
        str,
        typer.Option("--era", "-e", help="Era to use for generation"),
    ] = "deepdream",
    source: Annotated[
        Optional[Path],
        typer.Option("--source", "-s", help="Source image to transform"),
    ] = None,
    strength: Annotated[
        float,
        typer.Option("--strength", min=0.0, max=1.0, help="[img2img] How much to transform source (0=keep original, 1=fully regenerate)"),
    ] = 0.6,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path"),
    ] = Path("output.png"),
    intensity: Annotated[
        float,
        typer.Option("--intensity", "-i", min=0.0, max=1.0, help="Artifact intensity"),
    ] = 0.5,
    placement: Annotated[
        str,
        typer.Option("--placement", "-p", help="Artifact placement: global, edges, etc."),
    ] = "global",
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="Random seed for reproducibility"),
    ] = None,
    layer: Annotated[
        Optional[str],
        typer.Option("--layer", help="[DeepDream] Network layer to target"),
    ] = None,
    octaves: Annotated[
        Optional[int],
        typer.Option("--octaves", help="[DeepDream] Number of octaves"),
    ] = None,
    iterations: Annotated[
        Optional[int],
        typer.Option("--iterations", help="[DeepDream] Iterations per octave"),
    ] = None,
    # Early Diffusion params
    tile_size: Annotated[
        Optional[int],
        typer.Option("--tile-size", help="[Early Diffusion] Mosaic tile size in pixels"),
    ] = None,
    tile_style: Annotated[
        Optional[str],
        typer.Option("--tile-style", help="[Early Diffusion] 'obvious' or 'subtle'"),
    ] = None,
    gold_style: Annotated[
        Optional[str],
        typer.Option("--gold-style", help="[Early Diffusion] 'metallic' or 'symbolic'"),
    ] = None,
    inference_steps: Annotated[
        Optional[int],
        typer.Option("--inference-steps", help="[Early Diffusion] Diffusion steps (lower=more undercooked)"),
    ] = None,
    guidance_scale: Annotated[
        Optional[float],
        typer.Option("--guidance-scale", help="[Early Diffusion] CFG scale (higher=more saturated)"),
    ] = None,
    # Individual effect intensities (Early Diffusion)
    hallucination: Annotated[
        Optional[float],
        typer.Option("--hallucination", help="[Early Diffusion] Background hallucination intensity"),
    ] = None,
    halo_bleed: Annotated[
        Optional[float],
        typer.Option("--halo-bleed", help="[Early Diffusion] Halo bleed intensity"),
    ] = None,
    edge_fizz: Annotated[
        Optional[float],
        typer.Option("--edge-fizz", help="[Early Diffusion] Edge fizz intensity"),
    ] = None,
    eye_drift: Annotated[
        Optional[float],
        typer.Option("--eye-drift", help="[Early Diffusion] Eye drift intensity"),
    ] = None,
    almost_text: Annotated[
        Optional[float],
        typer.Option("--almost-text", help="[Early Diffusion] Almost-text intensity"),
    ] = None,
    finger_ambiguity: Annotated[
        Optional[float],
        typer.Option("--finger-ambiguity", help="[Early Diffusion] Finger ambiguity intensity"),
    ] = None,
    # International Gothic params
    porcelain_smoothness: Annotated[
        Optional[float],
        typer.Option("--porcelain-smoothness", help="[International Gothic] Porcelain skin smoothness"),
    ] = None,
    hair_bleeding: Annotated[
        Optional[float],
        typer.Option("--hair-bleeding", help="[International Gothic] Hair bleeding into background"),
    ] = None,
    eye_tracking_error: Annotated[
        Optional[float],
        typer.Option("--eye-tracking-error", help="[International Gothic] Eye misalignment"),
    ] = None,
    asymmetric_accessories: Annotated[
        Optional[float],
        typer.Option("--asymmetric-accessories", help="[International Gothic] Asymmetric earrings/jewelry"),
    ] = None,
    background_disconnect: Annotated[
        Optional[float],
        typer.Option("--background-disconnect", help="[International Gothic] Background confusion"),
    ] = None,
    gold_background: Annotated[
        Optional[float],
        typer.Option("--gold-background", help="[International Gothic] Gold leaf background intensity"),
    ] = None,
    tempera_texture: Annotated[
        Optional[float],
        typer.Option("--tempera-texture", help="[International Gothic] Tempera surface texture"),
    ] = None,
    courtly_palette: Annotated[
        Optional[float],
        typer.Option("--courtly-palette", help="[International Gothic] Rich courtly color palette"),
    ] = None,
    decorative_precision: Annotated[
        Optional[float],
        typer.Option("--decorative-precision", help="[International Gothic] Jewelry sharper than faces"),
    ] = None,
    emotional_vacancy: Annotated[
        Optional[float],
        typer.Option("--emotional-vacancy", help="[International Gothic] Flatten expression areas"),
    ] = None,
    # Early Renaissance params
    hand_failure: Annotated[
        Optional[float],
        typer.Option("--hand-failure", help="[Early Renaissance] Hand anatomical errors"),
    ] = None,
    foreshortening_error: Annotated[
        Optional[float],
        typer.Option("--foreshortening-error", help="[Early Renaissance] Depth representation failures"),
    ] = None,
    perspective_error: Annotated[
        Optional[float],
        typer.Option("--perspective-error", help="[Early Renaissance] Spatial contradictions"),
    ] = None,
    proportion_shift: Annotated[
        Optional[float],
        typer.Option("--proportion-shift", help="[Early Renaissance] Scale variation"),
    ] = None,
    edge_ambiguity: Annotated[
        Optional[float],
        typer.Option("--edge-ambiguity", help="[Early Renaissance] Figure-ground integration"),
    ] = None,
    wooden_face: Annotated[
        Optional[float],
        typer.Option("--wooden-face", help="[Early Renaissance] Stiff mannequin-like faces, misaligned eyes"),
    ] = None,
    # High Renaissance params
    blue_orange_cast: Annotated[
        Optional[float],
        typer.Option("--blue-orange-cast", help="[High Renaissance] THE MJ v4 color grading"),
    ] = None,
    overdramatized_lighting: Annotated[
        Optional[float],
        typer.Option("--overdramatized-lighting", help="[High Renaissance] Rim light, volumetric rays"),
    ] = None,
    hypersaturation: Annotated[
        Optional[float],
        typer.Option("--hypersaturation", help="[High Renaissance] Push colors beyond natural"),
    ] = None,
    epic_blur: Annotated[
        Optional[float],
        typer.Option("--epic-blur", help="[High Renaissance] Shallow DOF everywhere"),
    ] = None,
    textural_sharpening: Annotated[
        Optional[float],
        typer.Option("--textural-sharpening", help="[High Renaissance] Over-detailed surfaces"),
    ] = None,
    compositional_centering: Annotated[
        Optional[float],
        typer.Option("--compositional-centering", help="[High Renaissance] Pull subjects center"),
    ] = None,
    warm_halo: Annotated[
        Optional[float],
        typer.Option("--warm-halo", help="[High Renaissance] Glow around edges"),
    ] = None,
    device: Annotated[
        str,
        typer.Option("--device", help="Device: mps, cuda, cpu"),
    ] = "mps",
    width: Annotated[
        Optional[int],
        typer.Option("--width", "-W", help="Output width (default 1024)"),
    ] = None,
    height: Annotated[
        Optional[int],
        typer.Option("--height", "-H", help="Output height (default 1024)"),
    ] = None,
    upscale: Annotated[
        int,
        typer.Option("--upscale", "-U", help="Upscale factor after generation (1-4)"),
    ] = 1,
):
    """Generate an image with era-specific artifacts.

    Examples:

        hyperspeed generate --era deepdream --source photo.jpg

        hyperspeed generate --era deepdream --intensity 0.8 --layer mixed6

        hyperspeed generate "fractal dogs" --era deepdream --octaves 6
    """
    try:
        placement_enum = ArtifactPlacement(placement)
    except ValueError:
        console.print(f"[red]Unknown placement: {placement}[/red]")
        raise typer.Exit(1)

    # Load source image if provided
    source_image = None
    if source is not None:
        if not source.exists():
            console.print(f"[red]Source image not found: {source}[/red]")
            raise typer.Exit(1)
        console.print(f"Loading source image: {source}")
        source_image = load_image(source, max_size=1024)

    # Create artifact control
    control = ArtifactControl(
        intensity=intensity,
        placement=placement_enum,
        seed=seed,
    )

    # Build era params
    era_params = {}
    # DeepDream params
    if layer is not None:
        era_params["layer"] = layer
    if octaves is not None:
        era_params["octaves"] = octaves
    if iterations is not None:
        era_params["iterations"] = iterations
    # Early Diffusion params
    if tile_size is not None:
        era_params["tile_size"] = tile_size
    if tile_style is not None:
        era_params["tile_style"] = tile_style
    if gold_style is not None:
        era_params["gold_style"] = gold_style
    if inference_steps is not None:
        era_params["inference_steps"] = inference_steps
    if guidance_scale is not None:
        era_params["guidance_scale"] = guidance_scale
    # Individual effect intensities
    if hallucination is not None:
        era_params["background_hallucination"] = hallucination
    if halo_bleed is not None:
        era_params["halo_bleed"] = halo_bleed
    if edge_fizz is not None:
        era_params["edge_fizz"] = edge_fizz
    if eye_drift is not None:
        era_params["eye_drift"] = eye_drift
    if almost_text is not None:
        era_params["almost_text"] = almost_text
    if finger_ambiguity is not None:
        era_params["finger_ambiguity"] = finger_ambiguity
    # International Gothic params
    if porcelain_smoothness is not None:
        era_params["porcelain_smoothness"] = porcelain_smoothness
    if hair_bleeding is not None:
        era_params["hair_bleeding"] = hair_bleeding
    if eye_tracking_error is not None:
        era_params["eye_tracking_error"] = eye_tracking_error
    if asymmetric_accessories is not None:
        era_params["asymmetric_accessories"] = asymmetric_accessories
    if background_disconnect is not None:
        era_params["background_disconnect"] = background_disconnect
    if gold_background is not None:
        era_params["gold_background"] = gold_background
    if tempera_texture is not None:
        era_params["tempera_texture"] = tempera_texture
    if courtly_palette is not None:
        era_params["courtly_palette"] = courtly_palette
    if decorative_precision is not None:
        era_params["decorative_precision"] = decorative_precision
    if emotional_vacancy is not None:
        era_params["emotional_vacancy"] = emotional_vacancy
    # Early Renaissance params
    if hand_failure is not None:
        era_params["hand_failure"] = hand_failure
    if foreshortening_error is not None:
        era_params["foreshortening_error"] = foreshortening_error
    if perspective_error is not None:
        era_params["perspective_error"] = perspective_error
    if proportion_shift is not None:
        era_params["proportion_shift"] = proportion_shift
    if edge_ambiguity is not None:
        era_params["edge_ambiguity"] = edge_ambiguity
    if wooden_face is not None:
        era_params["wooden_face"] = wooden_face
    # High Renaissance params
    if blue_orange_cast is not None:
        era_params["blue_orange_cast"] = blue_orange_cast
    if overdramatized_lighting is not None:
        era_params["overdramatized_lighting"] = overdramatized_lighting
    if hypersaturation is not None:
        era_params["hypersaturation"] = hypersaturation
    if epic_blur is not None:
        era_params["epic_blur"] = epic_blur
    if textural_sharpening is not None:
        era_params["textural_sharpening"] = textural_sharpening
    if compositional_centering is not None:
        era_params["compositional_centering"] = compositional_centering
    if warm_halo is not None:
        era_params["warm_halo"] = warm_halo
    # img2img strength (only matters when --source is provided)
    era_params["img2img_strength"] = strength
    # Common params
    if width is not None:
        era_params["width"] = width
    if height is not None:
        era_params["height"] = height
    if upscale > 1:
        era_params["upscale"] = upscale

    # Get era pipeline
    try:
        pipeline_class = EraRegistry.get(era)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Era:[/bold] {pipeline_class.metadata.name}")
    console.print(f"[dim]{pipeline_class.metadata.art_historical_parallel}[/dim]")
    console.print()

    # Create and run pipeline
    with console.status("[bold green]Generating...[/bold green]"):
        pipeline = pipeline_class(device=device)
        result = pipeline.generate(
            prompt=prompt,
            source_image=source_image,
            control=control,
            **era_params,
        )

    # Save output
    output_path = save_image(result, output)
    console.print(f"[green]Saved to:[/green] {output_path}")


@app.command()
def eras():
    """List available eras with their art historical parallels."""
    table = Table(title="Available Eras")
    table.add_column("Era", style="cyan", no_wrap=True)
    table.add_column("Art Historical Parallel", style="magenta")
    table.add_column("Period", style="green")
    table.add_column("Artifacts", style="yellow")

    for metadata in EraRegistry.list_eras():
        artifacts = ", ".join(metadata.characteristic_artifacts[:3])
        if len(metadata.characteristic_artifacts) > 3:
            artifacts += "..."
        table.add_row(
            metadata.name,
            metadata.art_historical_parallel,
            metadata.time_period,
            artifacts,
        )

    console.print(table)


@app.command()
def params(
    era: Annotated[str, typer.Argument(help="Era to show parameters for")],
):
    """Show available parameters for an era."""
    try:
        pipeline_class = EraRegistry.get(era)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    metadata = pipeline_class.metadata
    console.print(f"[bold cyan]{metadata.name}[/bold cyan]")
    console.print(f"[dim]{metadata.description}[/dim]")
    console.print()

    # Create temp instance to get defaults
    pipeline = pipeline_class()
    defaults = pipeline.get_default_params()

    table = Table(title="Era Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Default", style="green")

    for name, value in defaults.items():
        table.add_row(f"--{name}", str(value))

    console.print(table)

    console.print()
    console.print("[bold]Characteristic Artifacts:[/bold]")
    for artifact in metadata.characteristic_artifacts:
        console.print(f"  - {artifact}")


@app.command()
def version():
    """Show version information."""
    from hyperspeed import __version__

    console.print(f"hyperspeed-nostalgia v{__version__}")


if __name__ == "__main__":
    app()
