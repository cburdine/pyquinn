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
""" The list of backend modules supported."""


DEFAULT_COLOR = '#5a38ce'
"""Default primary color used for plot lines."""

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
    """Plots a 3D visualization of atoms using a plotter backend module.

    Args:
        atoms (AtomGraph | BackendAtomGraph): The AtomGraph to be plotted.
        backend (str, optional): The backend plotter module to use. Defaults 
            to 'mpl' (matplotlib).
        bond_factor (float, optional): Cutoff factor for bonds 
            (given as a multiple of the covalent radii between atoms). 
            Defaults to 1.1.
        radius_factor (float, optional): Size factor of atomic radii. 
            Defaults to 0.5.
        atom_resolution (int, optional): The polygon resolution of 
            atomic spheres. Defaults to 10.
        hide_atoms (bool, optional): If True, hides atoms in the plot.
            Defaults to False.
        hide_bonds (bool, optional): If True, hides bonds in the plot. 
            Defaults to False.
        hide_unit_cell (bool, optional): If True, hides the unit cell
            in the plot. Defaults to False.
        fig (Any | None, optional): A backend figure object 
            to which the plot is added. If None, a new backend
            figure object is created. Defaults to None.
        show (bool, optional): If True, the figure is displayed. 
            Defaults to True.

    Returns:
        Any: The backend figure object.
    """

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
    """Plots a 3D visualization of an atomic isosurface.

    Isosurface functions can be plotted with one of the three following modes:

    * `"real"`:  Plots a single real-valued isosurface at `iso_value`.
    * `"signed"`: Plots a red (positive) and blue (negative) isosurface where the
        absolute value of `mesh_values` agrees with the absolute value of 
        `iso_surface`.
    * `"complex"`: Plots a shaded isosurface where the shade color indicates
        the complex phase of `mesh_values`.

    Args:
        mesh_xyz (np.ndarray): A rectangular mesh of x,y,z coordinates
            (shape: [N, 3]).
        mesh_values (np.ndarray): The isosurface function evaluated at 
            the `mesh_xyz` points (shape: [N]).
        backend (str, optional): The backend plotter module to use. Defaults 
            to 'mpl' (matplotlib).
        mode (str, optional): Isosurface mode to use. Defaults to 'real'.
        atoms (AtomGraph | None, optional): An AtomGraph to be plotted
            along with the isosurface. Defaults to None.
        iso_value (float | None, optional): 
            The isosurface value to be plotted. If None, the value is selected
            automatically. Defaults to None.
        color (str | None, optional): Color of the isosurface, 
            if plotting using the `real` mode. Defaults to None.
        alpha (float, optional): Opacity of the isosurface on a 
            scale of 0.0 to 1.0. Defaults to 1.0.
        fig (Any | None, optional): A backend figure object 
            to which the plot is added. If None, a new backend
            figure object is created. Defaults to None.
        show (bool, optional): If True, the figure is displayed. 
            Defaults to True.

    Returns:
        Any: The backend figure object.
    """

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
    """Plots a band structure path diagram.

    Args:
        bands (BandStructure | BackendBandStructure): A BandStructure object
            containing the band path and corresponding energies.
        backend (str, optional): The backend plotter module to use. Defaults 
            to 'mpl' (matplotlib).
        atoms (AtomGraph | BackendAtomGraph | None, optional): An AtomGraph
            used to infer special high-symmetry k points. Defaults to None.
        special_k_pts (dict | str | None, optional): A dictionary of special
            high-symmetry k points. Defaults to None.
        energy_units (str | None, optional): Energy units label. Defaults to None.
        fig (Any | None, optional): A backend figure object 
            to which the plot is added. If None, a new backend
            figure object is created. Defaults to None.
        show (bool, optional): If True, the figure is displayed. 
            Defaults to True.

    Returns:
        Any: The backend figure object.
    """
    
    plotter = _get_plotter_backend(backend)

    return plotter.plot_band_path(
        bands=bands,
        atoms=atoms,
        special_k_pts=special_k_pts,
        energy_units=energy_units,
        fig=fig,
        show=show,
        **plot_band_path_kwargs)
    
