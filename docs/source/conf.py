# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LLM4AD'
copyright = '2024, LLM4AD Team'
author = 'LLM4AD Team'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'recommonmark',
    'nbsphinx',
    'sphinx_markdown_tables',
    'myst_parser',
    # 'IPython.sphinxext.ipython_console_highlighting'
]

html_logo = 'assets/figs/logo_dark.jpg'

html_theme_options = {
    'path_to_docs': 'docs/source',
    'repository_url': 'https://github.com/Optima-CityU/LLM4AD',
    'use_repository_button': True,
    'use_edit_page_button': True,
    'home_page_in_toc': True,
}

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = ['_index.md', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']


# 添加 JavaScript 来动态切换 logo
# def setup(app):
#     app.add_js_file('_static/theme_switcher.js')
