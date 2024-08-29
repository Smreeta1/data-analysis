from setuptools import setup, find_packages

setup(
    name='Classify',
    version='0.1',
    description='Topics-Classify',
    author='Smreeta',
    packages=find_packages(),
    install_requires=[
        'nltk>=3.6.3',  
    ], 
)
