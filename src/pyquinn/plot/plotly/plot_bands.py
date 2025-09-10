from pyquinn.types import AtomGraph, BandStructure
from pyquinn.types.atom_graph import BackendAtomGraph
from pyquinn.types.bandstructure import BackendBandStructure
from pyquinn.util.geometry import (
    get_special_k_pts, 
    get_special_kpath_folds
)
from pyquinn.plot.plotter import DEFAULT_COLOR

import plotly.graph_objects as go
import numpy as np

PLOTLY_BAND_COLORS = {
        "red": (1, 0, 0, 1),
        "green": (0, 1, 0, 1),
        "blue": (0, 0, 1, 1),
        "black": (0, 0, 0, 1),
        "white": (1, 1, 1, 1),
        "gray": (0.5, 0.5, 0.5, 1),
}

def _normalize_color(c: str | tuple | list) -> tuple[float, float, float, float]:
    
    if isinstance(c, str):
        c = c.lower()
        if c.startswith("#"):
            hex = c.lstrip("#")
            if len(hex) == 6:
                r, g, b = [int(hex[i:i+2], 16)/255.0 for i in (0, 2, 4)]
                return (r, g, b, 1.0)
            elif len(hex) == 8:
                r, g, b, a = [int(hex[i:i+2], 16)/255.0 for i in (0, 2, 4, 6)]
                return (r, g, b, a)
        elif c in PLOTLY_BAND_COLORS:
            return PLOTLY_BAND_COLORS[c]
        else:
            raise ValueError(f"Unknown color string: {c}")
    elif isinstance(c, (tuple, list, np.ndarray)):
        c = np.asarray(c, dtype=float)
        if np.any(c > 1):
            c = c / 255.0
        if c.shape[0] == 3:
            return (*c, 1.0)
        elif c.shape[0] == 4:
            return tuple(c)
        else:
            raise ValueError(f"Color must have 3 or 4 components: got {c}")
    else:
        raise TypeError(f"Unsupported color format: {type(c)}")


def _plot_colored_line(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    point_colors: np.ndarray | list,
    linewidth: float = 2.0,
) -> go.Figure:

    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    rgba = np.array([_normalize_color(c) for c in point_colors])

    if rgba.shape[0] != len(x):
        raise ValueError("Length of point_colors must match x and y")

    for i in range(len(x) - 1):
        color = 'rgba({},{},{},{})'.format(*(rgba[i] + rgba[i+1]) * 0.5 * 255)
        trace = go.Scatter(
            x=x[i:i+2],
            y=y[i:i+2],
            mode='lines',
            line=dict(color=color, width=linewidth),
            showlegend=False,
            hoverinfo='skip'  # optional: disables tooltip for segments
        )
        fig.add_trace(trace)

    fig.update_layout(
        xaxis_title='x',
        yaxis_title='y',
        margin=dict(t=20, b=40, l=40, r=20),
    )
    return fig

def plot_band_path(
        bands: BandStructure | BackendBandStructure,
        atoms: AtomGraph | BackendAtomGraph | None = None,
        special_k_pts: dict | str | None = None,
        energy_units: str | None = None,
        fig: go.Figure | None = None,
        color: str = DEFAULT_COLOR,
        proj_color_map: list[tuple | str] | None = None,
        proj_color_weights: np.ndarray | None = None,
        proj_color_labels: list[str] | None = None,
        show: bool = True):

    bands = bands.extract()

    k_pts = bands.k_pts
    k_pts_x = np.arange(len(k_pts))
    k_path = None

    if atoms is not None:
        atoms = atoms.extract()

    energy_units_str = f' [{energy_units}]' if energy_units else ''

    if isinstance(special_k_pts, str):
        special_k_pts = None
        k_path = special_k_pts

    if special_k_pts is None:
        special_k_pts = get_special_k_pts(
            k_pts=k_pts,
            unit_cell=None if atoms is None else atoms.unit_cell, 
            periodic=None if atoms is None else atoms.periodic,  
            path=k_path)

    folds, fold_labels = get_special_kpath_folds(
        special_k_pts, k_pts_x, 2.1)

    data = [
        go.Scatter(
            x=k_pts_x,
            y=bands.energies[:, i],
            mode='lines',
            name=f'E({i})',
            line_color=color)
        for i in range(bands.energies.shape[1])
    ]

    if fig is None:
        fig = go.Figure(data=data)

    fig.update_layout(
        showlegend=False,
        autosize=True,
        width=600,
        height=500,
        margin=dict(
            l=0,
            r=0,
            b=100,
            t=100,
            pad=4
        ),
        xaxis=dict(
            title='Band Path',
            tickvals=folds,  # Custom tick positions
            ticktext=fold_labels,  # Custom tick labels
            tickmode='array',
            zeroline=False,
        ),
        yaxis=dict(
            title='Energy'+energy_units_str
        )
    )

    if show:
        fig.show()

    return fig
