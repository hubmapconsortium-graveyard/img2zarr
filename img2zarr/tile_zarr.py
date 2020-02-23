import numpy as np
import dask.array as da
import zarr
from numcodecs import Zlib

from pathlib import Path


def pad_axis(array, dim, pad_width):
    padding = [
        (0, 0) if i != dim else (0, pad_width) for i in range(len(array.shape))
    ]
    padded = da.pad(array, padding, "constant")
    return padded


def tile_zarr(
    base_image_path,
    max_level=None,
    dtype=None,
    chunks=None,
    pyramid_group="downsampled",
    compressor=Zlib(level=1),
    tile_size=512,
):
    z = zarr.open(base_image_path)
    pyramid_path = Path(base_image_path).parent

    img = da.from_zarr(z)
    y_size, x_size = img.shape[-2:]
    y_dim = len(img.shape) - 2
    x_dim = len(img.shape) - 1

    if dtype is None:
        dtype = img.dtype
    if max_level is None:
        # create all levels up to 512 x 512
        max_level = int(np.ceil(np.log2(np.maximum(y_size, x_size)))) - 9
    if chunks is None:
        chunks = img.chunksize

    # Halving of the last two dims per round
    downsample_scheme = {
        y_dim: 2,
        x_dim: 2,
    }

    for i in range(1, max_level):
        img = da.coarsen(np.mean, img, downsample_scheme, trim_excess=True)

        # Edge Case: Need to pad smallest thumbnail sometimes.
        #
        # If a dimension of an array is larger than TILE_SIZE,
        # zarr will respect the chunk size requested an automatically
        # pad with zeros in the store. However, if an array dimension
        # is smaller than the tile size, `da.to_zarr` will change the
        # chunking and not pad with zeros. We need sometimes need to pad
        # for the smallest tiles because x and y might not be square.
        if img.shape[y_dim] < tile_size:
            img = pad_axis(img, y_dim, tile_size - img.shape[y_dim])

        if img.shape[x_dim] < tile_size:
            img = pad_axis(img, x_dim, tile_size - img.shape[x_dim])

        # Define pyramid level path
        out_path = str(pyramid_path / pyramid_group / str(i).zfill(2))

        # Write to zarr store
        img.astype(dtype).rechunk(chunks).to_zarr(
            out_path, compressor=compressor
        )

        # Read from last store so dask doesn't need to re-compute
        # task graph starting at base.
        img = da.from_zarr(out_path)
