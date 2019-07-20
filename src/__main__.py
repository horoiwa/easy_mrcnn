import click
import os
import shutil

from .generator import dataset_generator


@click.group()
def cli():
    pass


@cli.command()
@click.option('--dataset_dir', '-d', required=True,
              help='Dataset folder path',
              type=click.Path(exists=True))
@click.option('--out_dir', '-o', default='logs',
              help='Dataset folder path')
def prep(dataset_dir, out_dir):
    out_dir = os.path.join(dataset_dir, out_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    dataset_dir = os.path.abspath(dataset_dir)
    out_dir = os.path.abspath(out_dir)

    dataset_generator(dataset_dir, out_dir)
    print("Finish prep dataset")


@cli.command()
@click.option('--dataset_path', '-d', required=True,
              help='Dataset folder path',
              type=click.Path(exists=True))
def training(dataset_path):
    dataset_path = os.path.abspath(dataset_path)
    print(dataset_path)


if __name__ == '__main__':
    cli()
