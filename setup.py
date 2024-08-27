from setuptools import setup, find_packages

setup(
    name='POS',
    version='0.1',
    description='POS detection',
    author='Smreeta',
    packages=find_packages(),
    install_requires=[
        'nltk>=3.6.3',  
    ], 
)
