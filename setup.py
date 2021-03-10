from distutils.core import setup

setup(
    name='dragoman',
    version='0.4dev',
    packages=['dragoman',],
    license='Apache 2.0',
    author='Philipp Eller',
    author_email='peller.phys@gmail.com',
    url='https://github.com/philippeller/dragoman',
    description='Look at data in different ways',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    setup_requires=[
        'python>=3.0',
        'pip>=1.8',
        'setuptools>18.5',
    ],

    install_requires=[
        'scipy>=0.17',
        'matplotlib>=2.0',
        'KDEpy',
        'tabulate',
        'numpy_indexed',
        'numpy>=1.11',
    ],
)
