from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "CorrDim"
author = "kduxin"
release = "0.2.0.dev2"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_member_order = "groupwise"
autodoc_typehints = "description"
autodoc_mock_imports = [
    "numpy",
    "sklearn",
    "torch",
    "tqdm",
    "transformers",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_title = f"{project} documentation"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}
