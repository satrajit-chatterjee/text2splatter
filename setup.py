from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()


setup(
    name="text2splatter",
    version="0.0.1",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
)