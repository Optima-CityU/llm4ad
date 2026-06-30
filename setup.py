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

CORE_REQUIRES = [
    'numpy<2',
    'matplotlib',
    'pytz',
    'tqdm',
    'ttkbootstrap',
]

EXTRAS_REQUIRE = {
    'numba': ['numba'],
    'tensorboard': ['torch', 'tensorboard'],
    'wandb': ['wandb'],
    'openai': ['openai'],
    'local-vllm': [
        'requests',
        'psutil',
        'torch',
        'flask',
        'flask-cors',
        'transformers',
        'vllm',
    ],
    'local-ollama': ['langchain-ollama'],
    'funsearch': ['scipy'],
    'multi-objective': ['pymoo'],
    'meoh': [
        'pymoo',
        'codebleu',
        'tree-sitter-python==0.23',
    ],
    'partevo': [
        'codebleu',
        'tree-sitter-python==0.23',
        'scikit-learn',
        'seaborn',
        'torch',
        'transformers',
    ],
    'llamea': [
        'llamea @ git+https://github.com/XAI-liacs/LLaMEA.git@main',
        'ConfigSpace',
    ],
    'machine-learning': ['gymnasium[box2d]'],
    'science-discovery': ['pandas', 'scipy', 'sympy'],
    'pymoo-task': ['pymoo'],
    'co-bench': [
        'datasets',
        'huggingface_hub',
        'httpx',
        'httpcore',
        'networkx',
        'scipy',
    ],
    'tsp-gls': ['numba', 'scipy'],
}

# Install every optional feature with:
#   pip install ".[all]"
EXTRAS_REQUIRE['all'] = sorted({
    package
    for packages in EXTRAS_REQUIRE.values()
    for package in packages
})

setup(
    name='llm4ad',
    version='1.0',
    author='LLM4AD Developers',
    description='Large Language Model for Algorithm Design Platform ',
    packages=find_packages(),
    package_dir={'': '.'},
    python_requires='>=3.9,<3.13',
    install_requires=CORE_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
