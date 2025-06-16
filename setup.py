# setup.py example
from setuptools import setup, find_packages

setup(
    name='unified_tile_extractor',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'Pillow',
        'PyYAML',
        'scikit-image',
        'rasterio',
        'shapely',
        'torch',
        'torchvision',
        'h5py',
    ],
    entry_points={
        'console_scripts': [
            'tile-extract = unified_tile_extractor.cli.preprocess_cli:main'
        ]
    },
)
