Installation
============

Python version requirements
---------------------------
.. important::
    Your Python version should >= Python 3.9 to ensure `ast.unparse()` function.

We provide three ways to install our package.

Run this project without installing the package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: console

    $ pip install numpy
    $ pip install scipy
    $ pip install torch  # if using tensorboard profiler
    $ pip install tensorboard # if using tensorboard profiler
    $ pip install wandb  # if using W&B profiler
    $ pip install numba  # if using number to for acceleration

Or simply install packages from requirements.txt

.. code-block:: console

    $ pip install -r requirements.txt


Locally build the package
~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: console

    $ git clone https://github.com/Optima-CityU/LLM4AD
    $ cd llm4ad
    $ pip install -e .

PyPI installation
~~~~~~~~~~~~~~~~~
.. code-block:: console

    $ pip install llm4ad


