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


def plot_band_path(
        bands: BandStructure | BackendBandStructure,
        atoms: AtomGraph | BackendAtomGraph | None = None,
        special_k_pts: dict | str | None = None,
        energy_units: str | None = None,
        fig: go.Figure | None = None,
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
            line_color=DEFAULT_COLOR)
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
