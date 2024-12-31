
# Py-LLM4AD: LLM for Algorithm Design Platform

> [!note] 
> Developer branch for: [LLM4AD](https://github.com/Optima-CityU/llm4ad), this repo is only for developing and debugging usage. 



<img src="./figs/llm-eps.png" alt="llm-eps" style="zoom:25%;" />

## 🪪 Licence

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for details. Part of this project, specifically the [`alevo/method/funsearch/programs_database.py`](llm4ad/method/funsearch/programs_database.py) file, uses code licensed under the Apache License 2.0.

## 🎁 Requirements (Python version >= 3.9)

> [!important]
>
> The Python version must be larger or equal to Python 3.9. Since we need `ast.unparse()`.

- numpy

- numba (if you want to use numba accelerate)
- tensorboard (if you want to use the tensorboard logger)

- wandb (if you want to use wandb logger)

## 📦 About this repository

This repository implements state-of-the-art large language model-based automated algorithm design methods. **We also provide LLM-free examples to help you understand the working pipeline of these methods! See usage below.** 

| Methods                                               | Paper title                                                  |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| RandomSampling                                        | Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models (PPSN 2024) |
| FunSearch                                             | Mathematical Discoveries from Program Search with Large Language Models (Nature 2023) |
| EoH<font color=red>*</font>                           | Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model (ICML 2024) |
| (1+1)-EPS<font color=red>*</font> <br/>(HillClimbing) | Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models (PPSN 2024) |
| RegEvo                                                | The algorithm is based on: Regularized Evolution for Image Classifier Architecture Search, Real et. al. |

<font color=red>*</font>The implementation has some minor differences from the original method (demonstrated in their original paper), considering generality and multithreading acceleration.

## 💡 Features of our package

| Feature                                                      | Support / To be supported    |
| ------------------------------------------------------------ | ---------------------------- |
| Unified interfaces for FOUR LLM-EPS methods                  | 🔥Support now                 |
| Evaluation acceleration: multiprocessing evaluation, add numba wrapper for heuristic | 🔥Support now                 |
| Secure evaluation: main process protection, timeout interruption | 🔥Support now                 |
| Log: Wandb and Tensorboard support                           | 🔥Support now                 |
| Secure evaluation: maximum RAM usage restriction             | 🚀Will be updated soon        |
| Resume run                                                   | 🔥Support now                 |
| More local LLM examples                                      | 🔥Support DeepSeek, Llama now |
| More AHD task examples                                       | 🚀Will be updated soon        |

## 🛠️ Project structure

|----**[alevo]**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----**[base]** basic package for modifying code, secure evaluation, and profiling experiments.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`code.py` implements two classes, Function and Program, to record the evolved heuristic.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`evaluate.py` implements the evaluation interface for the secure evaluator.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`evaluate_multi_program.py` a generic interface and evaluator to evaluate multiple programs.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`modify_code.py` tools for modifying code.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`sample.py` implements the llm interface, and trimmer.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----**[methods]** package for implementation classes of various LLM-EPS methods.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----**[tools]**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----**[llm]** package for examples to use local LLMs, and use API interfaces.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----**[profiler]** package for base implementations of Tensorboard and WandB profiler (logger).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;~~|----**[tasks]** package for AHD task examples (to be updated).~~

## ⚙️ Usage of this package

### Use [base] package to modify code and perform the secure evaluation

The `base` package is the basic package for all methods, which provides **useful tools** for manipulating the code for Python programs and functions, and methods to extract the valid part (Python code) from LLM-generated contents. Please refer to the tutorials (jupyter notebooks) in the `tutorials_for_base_package` folder. 

You are encouraged to implement your own EPS method using this package! 

### Use [method] package to perform an AD task

We provide LLM-free examples for you. Please look for examples in `example/online_bin_packing/fake_xxx.py`.  These examples provide a "fake sampler" that randomly selects code from the database to simulate the sampling process, which can help you test and debug our pipeline more easily. 

Run in terminal:

```shell
cd example/online_bin_packing
python fake_funsearch.py
```

We also provide tutorials for procedures to customize your own AHD tasks. An example of online bin packing problems (LLM-free or using your API key) is demonstrated on the Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RayZhhh/py-llm4ad/blob/main/online_bin_packing/online_bin_packing_tutorial.ipynb). 

## 🔗 Citation

```la
@article{liu2024llm4ad,
      title={LLM4AD: A Platform for Algorithm Design with Large Language Model}, 
      author={Fei Liu and Rui Zhang and Zhuoliang Xie and Rui Sun and Kai Li and Xi Lin and Zhenkun Wang and Zhichao Lu and Qingfu Zhang},
      year={2024},
      eprint={2412.17287},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.17287}, 
}
```

## ❌ Issues

If you encounter any difficulty using the code, please do not hesitate to submit an issue!

