# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from setuptools import setup, find_packages

setup(
    name='llm4ad',
    version='1.0',
    author='LLM4AD Developers',
    description='Large Language Model for Algorithm Design Platform ',
    packages=find_packages(),
    package_dir={'': '.'},
    python_requires='>=3.9',
    install_requires=[
        'numpy<2',
        'torch',
        'tensorboardX',
        'scipy',
        'tqdm',
        'numba',
        'requests',
        'openai',
        'pytz',
        'matplotlib',
        'python-docx',
        'ttkbootstrap'
    ]
)
