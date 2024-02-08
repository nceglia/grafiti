from setuptools import setup, find_packages

setup(
    name='sappy',
    version='0.0.1',
    description='Spatially Aware Phenotyping Python Package',
    packages=find_packages(include=['sappy','sappy.model','sappy.tools','sappy.plotting','sappy.datasets']),
)
