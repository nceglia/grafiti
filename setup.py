from setuptools import setup, find_packages

setup(
    name='grafiti',
    version='0.0.1',
    description='Graph Autoencoder For Imaging and Transcriptomic Inference',
    packages=find_packages(include=['grafiti','grafiti.model','grafiti.tools','grafiti.plotting','grafiti.datasets']),
)
