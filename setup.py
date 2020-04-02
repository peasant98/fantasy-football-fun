"""

Setup for fantasy_football_fun package.

"""
import os
from setuptools import setup, find_packages


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


with open('README.md', 'rb') as f:
    readme = f.read().decode('utf8', 'ignore')

setup(
    name='fantasy_football_fun',
    version='0.0.1',
    description='Codes for robotic skin project',
    long_description=readme,
    author='Matthew Strong',
    author_email='matthew.strong@colorado.edu',
    url='https://github.com/peasant98/fantasy-football-fun',
        install_requires=read_requirements(),
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)
