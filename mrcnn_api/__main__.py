import click
import os


@click.group()
def cli():
    pass


@cli.command()
@click.option('--dataset_path', '-d', required=True,
              help='Dataset folder path',
              type=click.Path(exists=True))
def create_dataset(dataset_path):
    dataset_path = os.path.abspath(dataset_path)
    print(dataset_path)


if __name__ == '__main__':
    cli()
