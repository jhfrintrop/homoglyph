# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


setup(
    name='homoglyph_cnn',
    version='0.0.1',
    author='Jan-Hendrik Frintrop',
    author_email='jan@jhfrintrop.de',
    description='Detecting Homoglyph Attacks with a Siamese Neural Network',
    long_description=open('README.md').read(),
    license='LICENSE',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'keras==2.2.0',
        'tensorflow==1.5.0',
        'numpy',
        'matplotlib',
        'pillow',
        'scikit-learn',
        'h5py',
        'fonttools',
        'click',
    ],
    entry_points={
        'console_scripts': [
            'homoglyph-cnn=cli:main',
        ],
    },
)
