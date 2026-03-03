from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

requirements += ["setuptools"]

setup(
    name="xrdmaptools",
    version="0.1.0",
    author="Evan J. Musterman",
    url="https://github.com/emusterman/xrdmaptools.git",
    packages=find_packages(),
    install_requires=requirements,
    description="Tools for analyzing XRD mapping data.",
    python_requires='>=3.9',
)