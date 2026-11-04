import submitit

project = "submitit"
copyright = "Meta Platforms, Inc."
release = submitit.__version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_custom_sections = [("Update parameters", "params_style")]
myst_heading_anchors = 3
exclude_patterns = ["_build"]

html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/facebookincubator/submitit",
    "source_branch": "main",
    "source_directory": "docs/",
}
