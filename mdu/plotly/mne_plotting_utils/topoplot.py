# Utilities to create topo plots from mne instances
import mne
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CloughTocher2DInterpolator


def create_plotly_topoplot(
    data: np.ndarray,
    inst: mne.io.Raw | mne.Epochs | mne.Evoked,
    contour_kwargs: dict = {"colorscale": "Viridis"},
    show: bool = False,
) -> go.FigureWidget:
    """Plot a topoplot from data and an mne instance for meta data information


    Parameters
    ----------
    data : np.ndarray
        the data for the topoplot, one value for each channel in inst.ch_names

    inst : mne.io.Raw | mne.Epochs | mne.Evoked
        the mne instance to get the channel meta information from


    Returns
    -------
    go.FigureWidget
        topo plot figure in plotly

    """
    pos = mne.channels.layout._find_topomap_coords(
        inst.info, inst.ch_names, to_sphere=True
    )
    r = get_radius(pos, scale_range=1.2)
    origin = get_origin(pos, inst.ch_names)

    fig = go.Figure()
    fig = plot_contour_heatmap(
        fig,
        data,
        inst,
        pos,
        origin=origin,
        radius=r,
        contour_kwargs=contour_kwargs,
    )
    fig = plot_sensors_at_topo_pos(fig, inst, pos=pos)
    fig = plot_head_sphere_nose_and_ears(
        fig, pos, inst.ch_names, radius=r, origin=origin
    )

    fig = fig.update_layout(
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    if show:
        fig.show()

    return fig


# ---------- no longer use the matplotlib hack around
def get_radius(pos: np.ndarray, scale_range: float = 1.2) -> float:
    return (
        max(
            pos[:, 0].max() - pos[:, 0].min(),
            pos[:, 1].max() - pos[:, 1].min(),
        )
        * scale_range
        / 2
    )


def get_origin(pos: np.ndarray, ch_names: list[str]) -> np.ndarray:
    """Use Cz coordinate of present, else chose (0, 0)"""
    if "Cz" in ch_names:
        assert len(ch_names) == len(pos)
        origin = pos[ch_names.index("Cz")]
    else:
        origin = np.asarray([0, 0])

    return origin


def plot_contour_heatmap(
    fig: go.Figure,
    data: np.ndarray,
    inst: mne.io.Raw | mne.Epochs | mne.Evoked,
    pos: np.ndarray,
    origin: np.ndarray = np.asarray([0, 0]),
    radius: float = 1,
    blank_scaling: float = 0.2,
    show: bool = False,
    contour_kwargs: dict = {"colorscale": "Viridis"},
) -> go.FigureWidget:
    # 2D interpolation
    blank_scaling = 0.2
    fig = go.Figure()
    interp = CloughTocher2DInterpolator(pos, data)
    xx, yy = np.meshgrid(
        np.linspace(-1 * radius + origin[0], 1 * radius + origin[0], 101),
        np.linspace(-1 * radius + origin[1], 1 * radius + origin[1], 101),
    )

    z = interp(xx, yy)

    # mask out internal points on the grid which are further away from any
    # channel than a fraction of the radius defined by blank_scaling
    gridpoints = np.stack([xx, yy], axis=-1)
    dist_tensor = np.asarray(
        [np.linalg.norm(gridpoints - p, axis=-1) for p in pos]
    )
    blank_msk = np.all(dist_tensor >= radius * blank_scaling, axis=0)
    z[blank_msk] = np.nan

    fig = fig.add_contour(
        z=z,
        x=xx[0, :],
        y=yy[:, 0],
        hoverinfo=None,
        coloraxis="coloraxis",
        **contour_kwargs,
    )

    # Using coloraxis above is used to unify in subplots with multiple axis
    # but removes any colorscale arguments -> manually fix here
    if "colorscale" in contour_kwargs:
        fig = fig.update_layout(
            coloraxis=dict(colorscale=contour_kwargs["colorscale"])
        )

    if show:
        fig.show()

    return fig


def plot_sensors_at_topo_pos(
    fig: go.Figure,
    inst: mne.io.Raw | mne.Epochs | mne.Evoked,
    show: bool = False,
    plot_outlines: bool = True,
    pos: np.ndarray = None,
) -> go.FigureWidget:
    """
    Plot the sensors at the topoplot positions given a mne.instances
    Note: the _find_topomap_coords projects to a 2d sphere instead of just
          using the x,y coordinates of a 3d vector
    """

    if pos is None:
        pos = mne.channels.layout._find_topomap_coords(
            inst.info, inst.ch_names, to_sphere=True
        )
    for chn, (pos_x, pos_y) in zip(inst.ch_names, pos):
        fig.add_scatter(
            x=[pos_x],
            y=[pos_y],
            text=[chn],
            name=chn,
            mode="markers",
            marker_size=5,
            marker_color="black",
            showlegend=False,
            hovertemplate="<b>%{text}</b><extra></extra>",
        )

    fig = fig.update_layout(height=500, width=500)

    if show:
        fig.show()

    return fig


def plot_head_sphere_nose_and_ears(
    fig: go.Figure,
    pos: np.ndarray,
    ch_names: list[str],
    scale_range: float = 1.20,
    radius: float = None,
    origin: np.ndarray = None,
) -> go.Figure:
    """Plot a sphere around Cz if present in the ch_names, else around (0, 0)"""
    if origin is None:
        origin = get_origin(pos, ch_names)
    if radius is None:
        radius = get_radius(pos, scale_range=scale_range)

    ll = np.linspace(0, np.pi * 2, 101)
    head_x = np.cos(ll) * radius + origin[0]
    head_y = np.sin(ll) * radius + origin[1]

    fig.add_scatter(
        x=head_x,
        y=head_y,
        mode="lines",
        opacity=0.5,
        hoverinfo=None,
        showlegend=False,
        name="head_line",
        line_color="#222",
    )

    fig = plot_ears(fig, radius, origin)
    fig = plot_nose(fig, radius, origin)

    return fig


def plot_nose(fig: go.Figure, r: float, origin: np.ndarray) -> go.Figure:
    dm = r * 0.1  # distance from circle in middle
    ddeg = 5  # width of nose in degree

    yside = (
        np.sin(np.pi / 2 - ddeg * np.pi / 180) * r
    )  # point where nose meets the circle
    xside = np.cos(np.pi / 2 - ddeg * np.pi / 180) * r

    fig.add_scatter(
        x=np.asarray([-xside, 0, xside]) + origin[0],
        y=np.asarray([yside, r + dm, yside]) + origin[1],
        name="nose",
        mode="lines",
        line_color="#222",
        hoverinfo=None,
        showlegend=False,
    )

    return fig


def plot_ears(fig: go.Figure, r: float, origin: np.ndarray) -> go.Figure:
    # coordinates from mne, scaled and translated, should result in the same
    # ear shape
    ear_x = (
        np.array(
            [
                0.497,
                0.510,
                0.518,
                0.5299,
                0.5419,
                0.54,
                0.547,
                0.532,
                0.510,
                0.489,
            ]
        )
        * (r * 2)
        + origin[0]
    )
    ear_y = (
        np.array(
            [
                0.0555,
                0.0775,
                0.0783,
                0.0746,
                0.0555,
                -0.0055,
                -0.0932,
                -0.1313,
                -0.1384,
                -0.1199,
            ]
        )
        * (r * 2)
        + origin[1]
    )

    fig.add_scatter(
        x=ear_x,
        y=ear_y,
        mode="lines",
        line_color="#222",
        name="ear_right",
        hoverinfo=None,
        showlegend=False,
    )
    fig.add_scatter(
        x=ear_x * -1,
        y=ear_y,
        mode="lines",
        line_color="#222",
        name="ear_left",
        hoverinfo=None,
        showlegend=False,
    )
    return fig
