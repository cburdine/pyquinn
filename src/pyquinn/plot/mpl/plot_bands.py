from pyquinn.types import AtomGraph, BandStructure
from pyquinn.types.atom_graph import BackendAtomGraph
from pyquinn.types.bandstructure import BackendBandStructure
from pyquinn.util.geometry import (
    get_special_k_pts, 
    get_special_kpath_folds
)
from pyquinn.plot.plotter import DEFAULT_COLOR

DEFAULT_MPL_STYLE = 'seaborn-v0_8'

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def plot_band_path(
        bands: BandStructure | BackendBandStructure,
        atoms: AtomGraph | BackendAtomGraph | None = None,
        special_k_pts: dict | str | None = None,
        energy_units: str | None = None,
        fig: Figure | None = None,
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

    with plt.style.context(DEFAULT_MPL_STYLE):
        if fig is None:
            fig = plt.figure()

        if len(fig.axes) > 0:
            ax = fig.axes[-1]
        else:
            ax = fig.add_subplot()

        for band in bands.energies.T:
            ax.plot(k_pts_x, band, color=DEFAULT_COLOR)
        ax.set_xticks(folds, labels=fold_labels)

        ax.set_xlabel('Band Path')
        ax.set_ylabel('Energy' + energy_units_str)
        ax.set_xlim(left=k_pts_x.min(), right=k_pts_x.max())

        if show:
            fig.show()

    return fig
