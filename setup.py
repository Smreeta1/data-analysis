from setuptools import setup, find_packages

setup(
    name='csv-merge',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
    ],

    author='Smreeta',
    description='Merging CSV files using pandas',
    license='MIT',
)
