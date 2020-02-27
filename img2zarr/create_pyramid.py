import click

import numpy as np
import dask.array as da
import zarr
from numcodecs import Zlib

from pathlib import Path

PYRAMID_GROUP_NAME = "sub-resolutions"
DEFAULT_COMPRESSOR = Zlib(level=1)


def pad_axis(array, dim, pad_width):
    padding = [(0, 0) if i != dim else (0, pad_width) for i in range(len(array.shape))]
    padded = da.pad(array, padding, "constant")
    return padded


def _create_pyramid(
    base_image_path, max_level=None, compressor=None,
):
    pyramid_path = Path(base_image_path).parent
    root = zarr.open_group(str(pyramid_path), mode="a")
    root.create_group(PYRAMID_GROUP_NAME, overwrite=True)

    # Gather metadata about store
    z = zarr.open(base_image_path)
    y_size, x_size = z.shape[-2:]
    y_dim = len(z.shape) - 2
    x_dim = len(z.shape) - 1
    tile_size = z.chunks[x_dim]
    chunks = z.chunks
    dtype = z.dtype

    # We want to read the image from zarr with a "good" chunksizes for computation.
    # Creating many small chunks is not ideal for dask and has large overhead.
    # https://docs.dask.org/en/latest/array-best-practices.html#select-a-good-chunk-size
    img = da.from_zarr(base_image_path, chunks="auto")

    if max_level is None:
        # create all levels up to 512 x 512
        max_level = int(
            np.ceil(np.log2(np.maximum(y_size, x_size))) - np.log2(tile_size)
        )
    if compressor is None:
        compressor = DEFAULT_COMPRESSOR

    # Halving of the last two dims per round
    downsample_scheme = {
        y_dim: 2,
        x_dim: 2,
    }

    for i in range(1, max_level):
        img = da.coarsen(np.mean, img, downsample_scheme, trim_excess=True)

        # Edge Case: Need to pad smallest thumbnail sometimes.
        if img.shape[y_dim] < tile_size:
            img = pad_axis(img, y_dim, tile_size - img.shape[y_dim])

        if img.shape[x_dim] < tile_size:
            img = pad_axis(img, x_dim, tile_size - img.shape[x_dim])

        # Define pyramid level path
        out_path = str(pyramid_path / PYRAMID_GROUP_NAME / str(i).zfill(2))

        # Write to zarr store
        # Ensure correct dtype and chunksizes for store
        img.astype(dtype).rechunk(chunks).to_zarr(out_path, compressor=compressor)

        # Read from last store so dask doesn't need to re-compute starting at base.
        img = da.from_zarr(out_path, chunks="auto")


@click.command()
@click.argument(
    "base", type=click.Path(),
)
@click.option(
    "--n_layers", help="Number of pyramidal layers to create.", type=int, default=None,
)
def create_pyramid(base, n_layers):
    max_level = None
    if n_layers is not None:
        max_level = n_layers + 1
    _create_pyramid(base_image_path=base, max_level=max_level)
