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
    device: Annotated[
        str,
        typer.Option("--device", help="Device: mps, cuda, cpu"),
    ] = "mps",
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
    if layer is not None:
        era_params["layer"] = layer
    if octaves is not None:
        era_params["octaves"] = octaves
    if iterations is not None:
        era_params["iterations"] = iterations

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
