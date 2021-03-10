from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='dragoman',
    version='0.4dev',
    packages=['dragoman',],
    license='Apache 2.0',
    author='Philipp Eller',
    author_email='peller.phys@gmail.com',
    url='https://github.com/philippeller/dragoman',
    description='Look at data in different ways',
    long_description=long_description,
    long_description_content_type='text/markdown',

    #setup_requires=[
    #    'python>=3.6',
    #    'pip>=1.8',
    #    'setuptools>18.5',
    #],

    install_requires=[
        'scipy>=0.17',
        'matplotlib>=2.0',
        'KDEpy',
        'tabulate',
        'numpy_indexed',
        'numpy>=1.11',
    ],
)
