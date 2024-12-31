from setuptools import setup, find_packages

setup(
    name='llm4ad',
    version='0.0.0',
    author='LLM4AD, Rui, City University of Hong Kong',
    description='Large language model for algorithm design platform.',
    long_description='',
    long_description_content_type='text/markdown',
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=[
        'numpy<2',
        'scipy',
    ],
    author_email='rzhang.cs@gmail.com',
    url='https://github.com/RayZhhh/py-llm4ad',
)
