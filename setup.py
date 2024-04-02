# File: setup.py

from setuptools import setup, find_packages

setup(
    name="CosmoHMC",
    version="0.1.0",
    author="Sayan Saha",
    author_email="sayan.saha@students.iiserpune.ac.in",
    description="A Hamiltonian Monte Carlo sampler for cosmology.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourgithub/CosmoHMC",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tqdm',
    ],
)