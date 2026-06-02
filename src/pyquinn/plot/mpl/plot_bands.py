import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pyquinn.plot.plotter import DEFAULT_COLOR
from pyquinn.types import AtomGraph, BandStructure
from pyquinn.types.atom_graph import BackendAtomGraph
from pyquinn.types.bandstructure import BackendBandStructure
from pyquinn.util.geometry import get_special_k_pts, get_special_kpath_folds

DEFAULT_MPL_STYLE = "seaborn-v0_8"


def _plot_colored_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    point_colors: np.ndarray | list,
    linewidth: float = 2.0,
) -> LineCollection:
    """Plot a line on the given axes with explicit per-point RGBA colors."""

    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # Convert input colors to RGBA array of shape (N, 4)
    rgba = point_colors

    if rgba.shape[0] != x.shape[0]:
        raise ValueError("Number of point colors must match number of points")

    # Average colors across each line segment (between points)
    segment_colors = 0.5 * (rgba[:-1] + rgba[1:])

    # Create segments: shape (N-1, 2, 2)
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Construct and add the LineCollection
    lc = LineCollection(segments, colors=segment_colors, linewidth=linewidth)
    ax.add_collection(lc)

    return lc


def _average_colors(color_list, color_weights):
    rgba = mcolors.to_rgba_array(color_list)
    W = np.array(color_weights)

    return np.clip(W @ rgba, 1e-4, 1 - 1e-4)


def plot_band_path(
    bands: BandStructure | BackendBandStructure,
    atoms: AtomGraph | BackendAtomGraph | None = None,
    special_k_pts: dict | str | None = None,
    energy_units: str | None = None,
    fig: Figure | None = None,
    color: tuple | str = DEFAULT_COLOR,
    proj_color_map: list[tuple | str] | None = None,
    proj_color_weights: np.ndarray | None = None,
    proj_color_labels: list[str] | None = None,
    show: bool = True,
):

    bands = bands.extract()

    k_pts = bands.k_pts
    k_pts_x = np.arange(len(k_pts))
    k_path = None

    if atoms is not None:
        atoms = atoms.extract()

    energy_units_str = f" [{energy_units}]" if energy_units else ""

    if isinstance(special_k_pts, str):
        special_k_pts = None
        k_path = special_k_pts

    if special_k_pts is None:
        special_k_pts = get_special_k_pts(
            k_pts=k_pts,
            unit_cell=None if atoms is None else atoms.unit_cell,
            periodic=None if atoms is None else atoms.periodic,
            path=k_path,
        )

    folds, fold_labels = get_special_kpath_folds(special_k_pts, k_pts_x, 2.1)

    with plt.style.context(DEFAULT_MPL_STYLE):

        if fig is None:
            fig = plt.figure()

        if len(fig.axes) > 0:
            ax = fig.axes[-1]
        else:
            ax = fig.add_subplot()

        if proj_color_weights is not None:
            proj_color_weights = np.array(proj_color_weights)

        for i, band in enumerate(bands.energies.T):
            if proj_color_weights is not None:
                color_weights = proj_color_weights[:, i]
                point_colors = _average_colors(proj_color_map, color_weights)
                _plot_colored_line(ax, k_pts_x, band, point_colors=point_colors)
            else:
                ax.plot(k_pts_x, band, color=color)

        if proj_color_labels is not None:
            proj_elements = [
                Line2D([0], [0], color=color, label=label)
                for color, label in zip(proj_color_map, proj_color_labels, strict=False)
            ]
            ax.legend(handles=proj_elements)

        ax.set_xticks(folds, labels=fold_labels)

        ax.set_xlabel("Band Path")
        ax.set_ylabel("Energy" + energy_units_str)
        ax.set_xlim(left=k_pts_x.min(), right=k_pts_x.max())

        if show:
            fig.show()

    return fig
