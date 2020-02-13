#!/usr/bin/env python

"""Tests for `img2zarr` package."""


import unittest
from click.testing import CliRunner

from img2zarr import img2zarr
from img2zarr import cli


class TestImg2zarr(unittest.TestCase):
    """Tests for `img2zarr` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'img2zarr.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
