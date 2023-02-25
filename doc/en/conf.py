# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
#
project = "FedML"
copyright = "2022-2023, FedML, Inc"
author = "fedml"

# The full version, including alpha/beta/rc tags
# release = 'FedML'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["myst_parser"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "pt_lightning_sphinx_theme"
html_theme_path = ["../theme/lightning_sphinx_theme"]
html_logo = "./_static/image/fedml_logo.png"
html_theme_options = {
    # 'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
    "analytics_anonymize_ip": False,
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "white",
    # Toc options
    # 'collapse_navigation': True,
    # 'sticky_navigation': True,
    "navigation_depth": 4,
    # 'includehidden': True,
    # 'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# html_sidebars = {
#     "**": [
#         "sidebar/scroll-start.html",
#         "sidebar/brand.html",
#         "sidebar/search.html",
#         "sidebar/navigation.html",
#         "sidebar/ethical-ads.html",
#         "sidebar/scroll-end.html",
#     ]
# }
html_context = {
    # 'css_files': [
    #     'https://fonts.googleapis.com/css?family=Lato',
    #     '_static/css/pytorch_theme.css'
    # ],
}

html_theme_options = {
    "light_css_variables": {
        "font-stack": "Arial, sans-serif",
        "font-stack--monospace": "Courier, monospace",
    },
}
