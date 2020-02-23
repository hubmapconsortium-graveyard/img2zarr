"""Console script for img2zarr."""
import sys
import click
from .tile_zarr import tile_zarr


@click.command()
@click.argument(
    "base", type=click.Path(),
)
@click.option(
    "--max_level",
    help="Number of pyramidal layers to create.",
    type=int,
    default=None,
)
@click.option(
    "--tile_size",
    help="""
    Default is 512. If different from full resolution
    chunk sizes, a new full resolution array will be
    created with desired tile size.""",
    type=int,
    default=512,
)
def main(base, max_level, tile_size):
    """Console script for img2zarr."""
    tile_zarr(base_image_path=base, max_level=max_level, tile_size=tile_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
