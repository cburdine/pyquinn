"""
Provides a plotting interface supporting multiple backends.
"""
from pyquinn.types.atom_graph import (
    AtomGraph, BackendAtomGraph
)
from pyquinn.types.bandstructure import (
    BandStructure, BackendBandStructure
)
from typing import Any

import numpy as np

PLOTTER_BACKENDS = [
    'plotly',
    'mpl'
]


DEFAULT_COLOR = '#5a38ce'


def _get_plotter_backend(backend: str) -> 'plotter': # noqa

    if backend == 'mpl':
        import pyquinn.plot.mpl as plotter
    elif backend == 'plotly':
        import pyquinn.plot.plotly as plotter
    else:
        raise ValueError(
            f'Unknown plotter backend type {backend} '
            f'(expected one of {PLOTTER_BACKENDS}).')

    return plotter


def plot_atoms(
        atoms: AtomGraph | BackendAtomGraph,
        backend: str = 'mpl',
        bond_factor: float = 1.1,
        radius_factor: float = 0.5,
        atom_resolution: int = 10,
        hide_atoms: bool = False,
        hide_bonds: bool = False,
        hide_unit_cell: bool = False,
        fig: Any | None = None,
        show: bool = True,
        **backend_kwargs) -> Any:

    plotter = _get_plotter_backend(backend)

    return plotter.plot_atoms(
        atoms=atoms,
        bond_factor=bond_factor,
        radius_factor=radius_factor,
        atom_resolution=atom_resolution,
        hide_atoms=hide_atoms,
        hide_bonds=hide_bonds,
        hide_unit_cell=hide_unit_cell,
        fig=fig,
        show=show,
        **backend_kwargs)


def plot_atomic_isosurface(
        mesh_xyz: np.ndarray,
        mesh_values: np.ndarray,
        backend: str = 'mpl',
        mode: str = 'real',
        atoms: AtomGraph | None = None,
        iso_value: float | None = None,
        color: str | None = None,
        alpha: float = 1.0,
        fig: Any | None = None,
        show: bool = True,
        **plot_atoms_kwargs) -> Any:

    plotter = _get_plotter_backend(backend)

    return plotter.plot_atomic_isosurface(
        mesh_xyz=mesh_xyz,
        mesh_values=mesh_values,
        mode=mode,
        atoms=atoms,
        iso_value=iso_value,
        color=color,
        alpha=alpha,
        fig=fig,
        show=show,
        **plot_atoms_kwargs)


def plot_band_path(
        bands: BandStructure | BackendBandStructure,
        backend: str = 'mpl',
        atoms: AtomGraph | BackendAtomGraph | None = None,
        special_k_pts: dict | str | None = None,
        energy_units: str | None = None,
        fig: Any | None = None,
        show: bool = True,
        **plot_band_path_kwargs):
    
    plotter = _get_plotter_backend(backend)

    return plotter.plot_band_path(
        bands=bands,
        atoms=atoms,
        special_k_pts=special_k_pts,
        energy_units=energy_units,
        fig=fig,
        show=show,
        **plot_band_path_kwargs)
    
