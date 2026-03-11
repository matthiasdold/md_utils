"""Example demonstrating optional plotly-resampler functionality."""

import numpy as np
import plotly.graph_objects as go

# This will work without plotly-resampler installed (with a warning)
# Or will use FigureResampler if installed with: pip install mdu[resampler]
from mdu.plotly.resampler_compat import FigureResampler, HAS_RESAMPLER

# Generate some example data
times = np.linspace(0, 10, 10000)
data = np.sin(times * 2 * np.pi) + np.random.randn(len(times)) * 0.1

# Create figure - will use resampler if available, plain plotly otherwise
fig = FigureResampler()

# Add trace (works with both resampler and fallback)
fig.add_trace(
    go.Scattergl(name="signal"),
    hf_x=times,
    hf_y=data,
)

fig.update_layout(title=f"Example plot (using {'resampler' if HAS_RESAMPLER else 'plain plotly'})")

print(f"Using plotly-resampler: {HAS_RESAMPLER}")
print("To enable resampler, install with: pip install mdu[resampler] or pip install mdu[all]")

# fig.show()  # Uncomment to display
