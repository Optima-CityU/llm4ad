Run examples
=================

**Step 1**, clone the repository from GitHub.

.. code-block:: console

    $ git clone https://github.com/Optima-CityU/LLM4AD

**Step 2**, make sure you have installed all requirements (Please refer to :doc:`installation`).

**Step 3**, find the corresponded python script and execute the script.
The script started with the word "fake" refers to LLM-free examples, as the Sampler randomly select a function from the database to imitate a sampling process.
If you have prepared an LLM API, Please goto **Step 4** and **Step 5**.

.. code-block:: console

    $ cd examples/online_bin_packing/
    $ python fake_randsample.py

**Step 4**, fill your api_endpoint and api_key in `randsample.py`.

.. code-block:: python

    api_endpoint: str = ''  # the ip of your API provider, no "https://", such as "api.bltcy.top".
    api_key: str = ''  # your API key which may start with "sk-......"

.. note::
    The `api_endpoint` has no "https://", e.g., "api.bltcy.top".

.. note::
    The API key may start with "sk-".

**Step 5**, run the script.

.. code-block:: console

    $ python randsample.py

