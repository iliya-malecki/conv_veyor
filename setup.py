from setuptools import setup, find_packages

from conv_veyor import __version__

setup(
    name='conv_veyor',
    version=__version__,

    url='https://github.com/iliya-malecki/conv_veyor',
    author='Iliya Malecki',
    author_email='iliyamalecki@gmail.com',
    
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'matplotlib',
        'Pillow',
        'wikipedia',
        'opencv-python >= 4.4',
        'pandas >= 1.0',
        'numpy >= 1.20',
    ],
)