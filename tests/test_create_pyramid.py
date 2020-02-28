#!/usr/bin/env python

"""Tests for `img2zarr` package."""

import pytest
from functools import reduce

import zarr
import numpy as np

from click.testing import CliRunner

from img2zarr import cli


def _write_base_zarr(store_fp, shape, chunks, dtype, name="base"):
    store = zarr.DirectoryStore(store_fp)
    root = zarr.group(store=store)
    base = root.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype)
    arr = np.arange(reduce(lambda x, y: y * x, shape)).reshape(shape)
    base[:] = arr


@pytest.fixture(scope="session")
def zarr_base_image_2d(tmp_path_factory):
    shape = (200, 2000)
    chunks = (32, 32)
    dtype = "uint16"
    fp = tmp_path_factory.mktemp("data2d.zarr")
    _write_base_zarr(store_fp=str(fp), shape=shape, chunks=chunks, dtype=dtype)
    return fp / "base"


@pytest.fixture(scope="session")
def zarr_base_image_rgb(tmp_path_factory):
    shape = (200, 2000, 3)
    chunks = (32, 32, 3)
    dtype = "uint16"
    fp = tmp_path_factory.mktemp("dataRgb.zarr")
    _write_base_zarr(store_fp=str(fp), shape=shape, chunks=chunks, dtype=dtype)
    return fp / "base"


# @pytest.fixture(scope="session")
# def zarr_base_image_3d(tmp_path_factory):
#     shape = (3, 200, 2000)
#     chunks = (1, 64, 64)
#     dtype = "uint8"
#     fp = tmp_path_factory.mktemp("data3d.zarr")
#     _write_base_zarr(store_fp=str(fp), shape=shape, chunks=chunks, dtype=dtype)
#     return fp / "base"


def test_2d_default(zarr_base_image_2d):
    runner = CliRunner()
    runner.invoke(cli.main, args=["create-pyramid", str(zarr_base_image_2d)])
    g = zarr.open(str(zarr_base_image_2d.parent / "sub-resolutions"))
    assert ["01", "02", "03", "04", "05"] == [k for k in g.array_keys()]


def test_2_layers_2d(zarr_base_image_2d):
    runner = CliRunner()
    runner.invoke(
        cli.main, args=["create-pyramid", "--n_layers", 2, str(zarr_base_image_2d)]
    )
    g = zarr.open(str(zarr_base_image_2d.parent / "sub-resolutions"))
    assert ["01", "02"] == [k for k in g.array_keys()]


def test_2d_rgb(zarr_base_image_2d):
    runner = CliRunner()
    runner.invoke(cli.main, args=["create-pyramid", str(zarr_base_image_2d)])
    g = zarr.open(str(zarr_base_image_2d.parent / "sub-resolutions"))
    assert ["01", "02", "03", "04", "05"] == [k for k in g.array_keys()]
