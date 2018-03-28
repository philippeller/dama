from distutils.core import setup

setup(
    name='MilleFeuille',
    version='0.1dev',
    packages=['millefeuille',],
    license='Apache 2.0',
    author='Philipp Eller',
    long_description=open('README.md').read(),

    setup_requires=[
        'pip>=1.8',
        'setuptools>18.5',
        'numpy>=1.11'
    ],

    install_requires=[
        'scipy>=0.17',
        'matplotlib>=2.0',
    ],
)
