from pyquinn.atoms.atom_graph import AtomGraph, BackendAtomGraph
from pyquinn.atoms.bandstructure import BandStructure, BackendBandStructure
from pyquinn.atoms.geometry import get_special_k_pts, get_special_kpath_folds

import plotly.graph_objects as go
import numpy as np


# def _get_special_k_pts(
#     atoms: AtomGraph,
#     k_pts: np.ndarray,
#     path: str | None = None,
#     special_radius: float = None):
    
#     if special_radius is None:
#         special_radius = 1.1*np.median(
#             np.linalg.norm(k_pts[:-1] - k_pts[1:], axis=-1))

#     # get the canonical band path and k-points (in fractional coordinates):
#     ase_atoms = atoms.to_ase()
#     rcell = ase_atoms.cell.reciprocal()
#     band_path = ase_atoms.cell.bandpath(path=path, npoints=256)

#     path_pts = {}
#     for name, pt in band_path.special_points.items():
#         path_pts[name] = rcell.cartesian_positions(pt)

#     # get the special points of the first Brillouin zone:
#     special_kpts = {
#         name.replace('G', 'Γ'): 
#             np.nonzero(
#                 np.linalg.norm(k_pts - pt, axis=-1) <= 
#                 special_radius)[0]
#         for name, pt in path_pts.items()
#     }
    
#     return special_kpts

# def _get_special_kpath_folds(
#     special_kpts: dict, 
#     kpts_x: np.ndarray | None = None, 
#     x_radius: float | None = None):
    
#     # construct fold points
#     fold_pts = []
#     for label, idxs in special_kpts.items():
#         for i in idxs:
#             x = i if kpts_x is None else kpts_x[i]
#             fold_pts.append((x, label))
    
#     fold_pts.sort()
#     folds = [f[0] for f in fold_pts]
#     fold_tokens = [(f[1],) for f in fold_pts]

#     # Handle if the same special point was labeled twice consecutively
#     if x_radius is not None:
#         dx = (np.diff(folds) <= x_radius).nonzero()[0]
#         for i in dx:
#             if fold_tokens[i][-1] != fold_tokens[i+1][-1]:
#                 fold_tokens[i+1] = (fold_tokens[i] + fold_tokens[i+1])
#             else:
#                 fold_tokens[i+1] = fold_tokens[i]
#             fold_tokens[i] = tuple()

#     # Gather all nonempty fold labels
#     folds = np.array(folds)
#     fold_labels = np.array(['|'.join(t) for t in fold_tokens])
#     labeled_idxs = (fold_labels != '')
    
#     return folds[labeled_idxs], fold_labels[labeled_idxs]

def plot_bandstructure(
        bands: BandStructure | BackendBandStructure,
        atoms: AtomGraph | BackendAtomGraph | None = None,
        special_k_pts: dict | str | None = None,
        energy_units: str | None = None,
        show: bool = True):

    atoms = atoms.extract()
    k_pts = bands.k_pts
    k_pts_x = np.arange(len(k_pts))
    k_path = None

    energy_units_str = f' [{energy_units}]' if energy_units else ''

    if isinstance(special_k_pts, str):
        special_k_pts = None
        k_path = special_k_pts

    if special_k_pts is None:
        special_k_pts = get_special_k_pts(
            unit_cell=atoms.unit_cell, 
            periodic=atoms.periodic, 
            k_pts=k_pts, 
            path=k_path)

    folds, fold_labels = get_special_kpath_folds(
        special_k_pts, k_pts_x, 2.1)

    data = [
        go.Scatter(
            x=k_pts_x,
            y=bands.energies[:,i],
            mode='lines',
            name=f'E({i})')
        for i in range(bands.energies.shape[1])
    ]

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