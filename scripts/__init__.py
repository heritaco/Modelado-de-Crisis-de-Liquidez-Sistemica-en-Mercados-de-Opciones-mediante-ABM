"""
Script package initialization.
------------------------------
This module exposes functions to apply styles to matplotlib and plotly visualizations.
"""

from style import mpl_apply, set_style, plotly_apply

__all__ = [
    "mpl_apply",
    "set_style",
    "plotly_apply"
]


# version
__version__ = "1.1.0"

"""
Updates:
1.1.0 - 2025-12-01
    - Better mpl plots  
"""