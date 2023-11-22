import plotly.graph_objects as go


def apply_default_styles(
    fig: go.Figure,
    row: int | None = None,
    col: int | None = None,
    xzero: bool = True,
    yzero: bool = True,
    showgrid: bool = True,
    ygrid: bool = True,
    xgrid: bool = True,
    gridoptions: dict
    | None = dict(
        gridcolor="#444444", gridwidth=1, griddash="dot"
    ),  # options for dash are 'solid', 'dot', 'dash', 'longdash', 'dashdot',
    # or 'longdashdot'
    gridoptions_y: dict | None = None,
    gridoptions_x: dict | None = None,
) -> go.Figure:
    fig.update_xaxes(
        showgrid=showgrid,
        gridcolor="#444444",
        linecolor="#444444",
        col=col,
        row=row,
    )
    if xzero:
        fig.update_xaxes(zerolinecolor="#444444", row=row, col=col)

    fig.update_yaxes(
        showgrid=showgrid,
        gridcolor="#444444",
        zerolinecolor="#444444",
        linecolor="#444444",
        row=row,
        col=col,
    )
    if yzero:
        fig.update_yaxes(zerolinecolor="#444444", row=row, col=col)

    # Narrow margins large ticks for better readability
    tickfontsize = 20
    fig.update_layout(
        font=dict(size=tickfontsize),
        margin=dict(l=40, r=5, t=40, b=40),
        title=dict(x=0.5, xanchor="center"),
    )

    # clean background
    fig.update_layout(
        plot_bgcolor="#ffffff",  # 'rgba(0,0,0,0)',   # transparent bg
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=16),
    )

    # grid
    if showgrid:
        ygrid = True
        xgrid = True
    if gridoptions:
        gridoptions_x = gridoptions
        gridoptions_y = gridoptions

    if ygrid:
        fig.update_yaxes(
            **gridoptions_y,
            row=row,
            col=col,
        )

    if xgrid:
        fig.update_xaxes(
            **gridoptions_x,
            row=row,
            col=col,
        )

    return fig
