from setuptools import setup, find_packages

setup(
    name='processing',
    version='0.1',
    description='NLP',
    author='Smreeta',
    packages=find_packages(),
    install_requires=[
        'nltk>=3.6.3',  
    ], 
)
