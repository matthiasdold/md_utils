"""Compatibility layer for optional plotly-resampler support.

This module provides a fallback when plotly-resampler is not installed,
allowing the package to work with plain plotly figures but with a performance warning.
"""

import logging
import warnings

logger = logging.getLogger(__name__)

# Try to import plotly-resampler, fallback to plain plotly if unavailable
try:
    from plotly_resampler import FigureResampler as _FigureResampler

    HAS_RESAMPLER = True
    FigureResampler = _FigureResampler
except ImportError:
    HAS_RESAMPLER = False
    import plotly.graph_objects as go

    class FigureResampler(go.Figure):
        """Fallback FigureResampler that behaves like a regular plotly Figure.

        When plotly-resampler is not installed, this class provides a compatible
        interface but without the resampling performance benefits.
        """

        def __init__(self, *args, **kwargs):
            # Issue warning on first instantiation
            warnings.warn(
                "plotly-resampler is not installed. Large time series plots may "
                "have poor performance. Install with: pip install mdu[resampler] "
                "or pip install mdu[all] for better performance.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "Using plain plotly Figure instead of FigureResampler. "
                "Performance may be degraded for large datasets."
            )
            super().__init__(*args, **kwargs)

        def add_trace(self, trace, hf_x=None, hf_y=None, **kwargs):
            """Add trace with compatibility for hf_x and hf_y parameters.

            Parameters
            ----------
            trace : plotly trace object
                The trace to add
            hf_x : array-like, optional
                High-frequency x data (used by resampler, ignored in fallback)
            hf_y : array-like, optional
                High-frequency y data (used by resampler, ignored in fallback)
            **kwargs
                Additional arguments passed to parent add_trace
            """
            # If hf_x and hf_y are provided, use them instead of trace data
            if hf_x is not None and hf_y is not None:
                trace.x = hf_x
                trace.y = hf_y
            return super().add_trace(trace, **kwargs)

        def replace(self, figure):
            """Replace the current figure with a new one.

            Parameters
            ----------
            figure : plotly.graph_objects.Figure
                The figure to replace with
            """
            # Clear current figure data and layout
            self.data = []
            self.layout = figure.layout


def get_figure_resampler(*args, **kwargs):
    """Factory function to create a FigureResampler or fallback Figure.

    Returns
    -------
    FigureResampler or go.Figure
        A FigureResampler if available, otherwise a compatible fallback Figure.
    """
    return FigureResampler(*args, **kwargs)


def warn_if_no_resampler():
    """Issue a warning if plotly-resampler is not available."""
    if not HAS_RESAMPLER:
        warnings.warn(
            "plotly-resampler is not installed. Large time series plots may "
            "have poor performance. Install with: pip install mdu[resampler] "
            "or pip install mdu[all]",
            UserWarning,
            stacklevel=2,
        )
