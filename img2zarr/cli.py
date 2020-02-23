import sys
import click

from img2zarr.create_pyramid import create_pyramid


@click.group()
def main():
    pass


main.add_command(create_pyramid)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
