from setuptools import setup, find_packages

setup(
    name='potnet',  
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',  
        'pandas',
        'matplotlib',
        'torch==2.4.1', 
        'pot==0.9.3', 
        'sklearn',
        'tqdm'
    ],
    python_requires='>=3.9',  
    author="Wenhui Sophia Lu", 
    author_email="sophialu@stanford.edu", 
    description="This package provides an implementation of the POTNet model for synthetic data generation using PyTorch, as described in the corresponding paper 'Efficient Generative Modeling via Penalized Optimal Transport Network'.",
    long_description=open('README.md').read(),
    keywords="generative modeling, marginal penalization, optimal transport, tabular data, Wasserstein distance",
    long_description_content_type='text/markdown',
    url="https://github.com/sophial05/POTNet",  
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://github.com/sophial05/POTNet#readme", 
        "Source Code": "https://github.com/sophial05/POTNet",  
        "Paper": "https://arxiv.org/abs/2402.10456",
    },
)