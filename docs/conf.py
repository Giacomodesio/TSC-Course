# Configuration file for the Sphinx documentation builder.
# For full documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))  # Adatta il path alla root del progetto

# -- Project information -----------------------------------------------------
project = 'TSC_course'
copyright = '2024, De Sio'
author = 'De Sio'
release = '1.0.0'  # Versione del progetto

# -- General configuration ---------------------------------------------------
extensions = [
    'autoapi.extension',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

autoapi_type = 'python'
autoapi_dirs = ['../src']  # Root directory del codice Python
autoapi_ignore = ['*conf.py']
autoapi_generate_api_docs = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for AutoAPI -----------------------------------------------------
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
]
