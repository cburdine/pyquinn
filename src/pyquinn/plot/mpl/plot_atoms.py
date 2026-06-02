import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata

from pyquinn.types import AtomGraph
from pyquinn.types.atom_graph import BackendAtomGraph
from pyquinn.util.data import (
    ATOM_COLORS,
    ATOMIC_COVALENT_RADII,
    ATOMIC_NUMBERS,
)

ATOMIC_ISOSURFACE_MODES = [
    "real",
    "signed",
    "complex",
]


def _infer_shape_and_spacing(mesh_xyz):
    x_unique, x_counts = np.unique(mesh_xyz[:, 0], return_counts=True)
    y_unique, y_counts = np.unique(mesh_xyz[:, 1], return_counts=True)
    z_unique, z_counts = np.unique(mesh_xyz[:, 2], return_counts=True)

    x_spacing = np.unique(np.diff(np.sort(x_unique)))
    y_spacing = np.unique(np.diff(np.sort(y_unique)))
    z_spacing = np.unique(np.diff(np.sort(z_unique)))

    axis_data = [
        ("x", x_counts, x_spacing),
        ("y", y_counts, y_spacing),
        ("z", z_counts, z_spacing),
    ]

    for axis, counts, spacing in axis_data:
        if len(np.unique(counts)) > 1:
            raise ValueError(
                'Argument "mesh_xyz" must be a valid cubic mesh along the '
                f'"{axis}" direction.\n'
                f"Found different grid sizes along this axis: {counts}"
            )

        if not np.allclose(spacing, spacing[0]):
            raise ValueError(
                'Argument "mesh_xyz" must be a uniform cubic '
                f'mesh along the "{axis}" direction.\n'
                f"Found different step sizes along this axis: {spacing}"
            )

    shape = (x_unique.shape[0], y_unique.shape[0], z_unique.shape[0])
    spacing = (x_spacing[0], y_spacing[0], z_spacing[0])

    return shape, spacing


def _get_isosurface_mesh(mesh_xyz, mesh_values, iso_val, mesh_data=None):

    try:
        from skimage import measure
    except ImportError as err:
        raise ImportError(
            'Package "scikit-image" (skimage) must be installed to '
            "visualize isosurfaces with the mpl (matplotlib) backend. "
            'Please install with "pip install scikit-image".'
        ) from err

    mesh_shape, mesh_spacing = _infer_shape_and_spacing(mesh_xyz)
    V = mesh_values.reshape(mesh_shape)

    surface_xyz, surface_ijk, _, _ = measure.marching_cubes(V, level=iso_val, spacing=mesh_spacing)

    surface_xyz += np.min(mesh_xyz, axis=0, keepdims=True)

    if mesh_data is None:
        surface_data = None
    else:
        surface_data = griddata(points=mesh_xyz, values=mesh_data, xi=surface_xyz, method="nearest")

    return surface_xyz, surface_ijk, surface_data


def _make_sphere(xyz, r, resolution=10):
    u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]
    X = r * np.cos(u) * np.sin(v) + xyz[0]
    Y = r * np.sin(u) * np.sin(v) + xyz[1]
    Z = r * np.cos(v) + xyz[2]
    return (X, Y, Z)


def _make_bonds(positions, nvecs):

    x_lines, y_lines, z_lines = [], [], []
    for x0, dxs in zip(positions, nvecs, strict=False):
        for dx in dxs:
            x_lines.append([x0[0], x0[0] + dx[0]])
            y_lines.append([x0[1], x0[1] + dx[1]])
            z_lines.append([x0[2], x0[2] + dx[2]])

    x_lines = np.array(x_lines)
    y_lines = np.array(y_lines)
    z_lines = np.array(z_lines)

    return x_lines, y_lines, z_lines


def _make_unitcell(unit_cell, periodic=False, origin=None):

    if origin is None:
        origin = np.zeros(3)

    # compute unit cell vertices:
    v1, v2, v3, o = unit_cell[0], unit_cell[1], unit_cell[2], origin

    vertices = np.array(
        [
            o,
            o + v1,
            o + v2,
            o + v3,
            o + v1 + v2,
            o + v1 + v3,
            o + v2 + v3,
            o + v1 + v2 + v3,
        ]
    )

    edges = np.array(
        [
            (0, 1),
            (0, 2),
            (0, 3),  # Edges from origin
            (1, 4),
            (1, 5),  # Edges from v1
            (2, 4),
            (2, 6),  # Edges from v2
            (3, 5),
            (3, 6),  # Edges from v3
            (4, 7),
            (5, 7),
            (6, 7),  # Edges connecting the top face
        ]
    )

    # Create a list of x, y, z coordinates for edges
    x_lines, y_lines, z_lines = [], [], []

    for edge in edges:
        line = vertices[edge]
        x_lines.append(line[:, 0])
        y_lines.append(line[:, 1])
        z_lines.append(line[:, 2])

    x_lines = np.array(x_lines)
    y_lines = np.array(y_lines)
    z_lines = np.array(z_lines)

    return x_lines, y_lines, z_lines


def plot_atoms(
    atoms: AtomGraph | BackendAtomGraph,
    bond_factor: float = 1.1,
    radius_factor: float = 0.5,
    atom_resolution: int = 10,
    hide_atoms: bool = False,
    hide_bonds: bool = False,
    hide_unit_cell: bool = False,
    fig: Figure | None = None,
    show: bool = True,
) -> Figure:

    fig, xyz_min, xyz_max = _plot_atoms_with_aspect(
        atoms=atoms,
        bond_factor=bond_factor,
        radius_factor=radius_factor,
        atom_resolution=atom_resolution,
        hide_atoms=hide_atoms,
        hide_bonds=hide_bonds,
        hide_unit_cell=hide_unit_cell,
        fig=fig,
    )

    # fix aspect ratio
    if not np.allclose(xyz_min, xyz_max):
        fig.axes[-1].set_box_aspect(tuple(xyz_max[i] - xyz_min[i] for i in range(3)))

    if show:
        plt.show()

    return fig


def _plot_atoms_with_aspect(
    atoms: AtomGraph | BackendAtomGraph,
    bond_factor: float = 1.1,
    radius_factor: float = 0.5,
    atom_resolution: int = 10,
    hide_atoms: bool = False,
    hide_bonds: bool = False,
    hide_unit_cell: bool = False,
    fig: Figure | None = None,
) -> tuple:

    if fig is None:
        fig = plt.figure()

    if len(fig.axes) > 0:
        ax = fig.axes[-1]
    else:
        ax = fig.add_subplot(projection="3d")

    # record bounding box of all mesh values
    xyz_min = np.ones(3) * -1e-2
    xyz_max = np.ones(3) * 1e-2

    # render atoms
    if not hide_atoms:
        for i, symbol in enumerate(atoms.atoms):
            z = ATOMIC_NUMBERS[symbol] if symbol in ATOMIC_NUMBERS else 0

            # get color and radius of atom
            rgb = ATOM_COLORS[z]
            color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
            radius = ATOMIC_COVALENT_RADII[z] * radius_factor

            # make atom sphere
            xyz = atoms.positions[i]
            X, Y, Z = _make_sphere(xyz, radius, resolution=atom_resolution)

            # plot atom sphere
            color = tuple(rgb / 256 for rgb in ATOM_COLORS[z])
            ax.plot_surface(X, Y, Z, color=color)

            for i, M in enumerate((X, Y, Z)):
                xyz_min[i] = min(xyz_min[i], M.min())
                xyz_max[i] = max(xyz_max[i], M.max())

    if not hide_bonds:
        bond_cutoffs = [
            (bond_factor * ATOMIC_COVALENT_RADII[ATOMIC_NUMBERS[sym]] if sym in ATOMIC_NUMBERS else 0.0)
            for sym in atoms.atoms
        ]

        _, nvecs = atoms.get_neighbor_list(cutoffs=bond_cutoffs)
        x_lines, y_lines, z_lines = _make_bonds(atoms.positions, nvecs)
        for xl, yl, zl in zip(x_lines, y_lines, z_lines, strict=False):
            ax.plot(xl, yl, zl, color="grey", linewidth=3)
            ax.scatter(xl, yl, zl, color="grey")

        for i, lines in enumerate((x_lines, y_lines, z_lines)):
            if len(lines) > 0:
                xyz_min[i] = min(xyz_min[i], lines.min())
                xyz_max[i] = max(xyz_max[i], lines.max())

    if not hide_unit_cell:
        x_lines, y_lines, z_lines = _make_unitcell(atoms.unit_cell, atoms.periodic)

        for xl, yl, zl in zip(x_lines, y_lines, z_lines, strict=False):
            ax.plot(xl, yl, zl, color="black", linewidth=1)

        for i, lines in enumerate((x_lines, y_lines, z_lines)):
            if len(lines) > 0:
                xyz_min[i] = min(xyz_min[i], lines.min())
                xyz_max[i] = max(xyz_max[i], lines.max())

    return fig, xyz_min, xyz_max


def plot_atomic_isosurface(
    mesh_xyz: np.ndarray,
    mesh_values: np.ndarray,
    mode: str = "real",
    atoms: AtomGraph | None = None,
    iso_value: float | None = None,
    color: str | None = None,
    alpha: float = 1.0,
    fig: Figure | None = None,
    show: bool = True,
    **plot_atoms_kwargs,
):

    # record bounding box of all mesh values

    xyz_min = np.minimum(np.ones(3) * -1e-2, mesh_xyz.min(axis=0))
    xyz_max = np.maximum(np.ones(3) * -1e-2, mesh_xyz.max(axis=0))

    if mode not in ATOMIC_ISOSURFACE_MODES:
        raise ValueError(f"Expected mode to be one of {ATOMIC_ISOSURFACE_MODES}; " f'got: "{mode}".')

    # plot figure
    if fig is None:
        fig = plt.figure()
    if len(fig.axes) > 0:
        ax = fig.axes[-1]
    else:
        ax = fig.add_subplot(projection="3d")

    if mode == "signed":

        pos_mesh_values = np.maximum(0.0, np.real(mesh_values))
        neg_mesh_values = np.minimum(0.0, np.real(mesh_values))

        pos_iso_value = np.mean(pos_mesh_values) if iso_value is None else np.abs(iso_value)
        neg_iso_value = np.mean(neg_mesh_values) if iso_value is None else -np.abs(iso_value)
        comp_iso_value = 0.0 if iso_value is None else np.abs(iso_value)

        if np.min(neg_mesh_values) < -comp_iso_value:
            surf_xyz, surf_ijk, surf_data = _get_isosurface_mesh(
                mesh_xyz=mesh_xyz,
                mesh_values=pos_mesh_values,
                iso_val=pos_iso_value,
            )

            ax.add_collection(Poly3DCollection(surf_xyz[surf_ijk], facecolors="red", alpha=alpha))

        if np.max(pos_mesh_values) > comp_iso_value:
            surf_xyz, surf_ijk, surf_data = _get_isosurface_mesh(
                mesh_xyz=mesh_xyz,
                mesh_values=neg_mesh_values,
                iso_val=neg_iso_value,
            )

            ax.add_collection(Poly3DCollection(surf_xyz[surf_ijk], facecolors="blue", alpha=alpha))

    elif mode == "complex":
        iso_value = np.abs(mesh_values).mean() if iso_value is None else np.abs(iso_value)

        mesh_phases = mesh_values / np.abs(mesh_values)
        surf_xyz, surf_ijk, surf_data = _get_isosurface_mesh(
            mesh_xyz=mesh_xyz,
            mesh_values=np.abs(mesh_values),
            iso_val=iso_value,
            mesh_data=mesh_phases,
        )

        face_data = surf_data[surf_ijk].mean(axis=1)
        phases = (np.angle(face_data) / (2 * np.pi)) % 1.0
        phase_cmap = mpl.colormaps["hsv"]

        ax.add_collection(Poly3DCollection(surf_xyz[surf_ijk], facecolors=phase_cmap(phases), alpha=alpha))
    else:

        if iso_value is None:
            iso_value = np.abs(mesh_values).mean()

        surf_xyz, surf_ijk, surf_data = _get_isosurface_mesh(
            mesh_xyz=mesh_xyz, mesh_values=mesh_values, iso_val=iso_value
        )

        ax.add_collection(Poly3DCollection(surf_xyz[surf_ijk], facecolors=color, alpha=alpha))

    if atoms is not None:
        fig, atoms_xyz_min, atoms_xyz_max = _plot_atoms_with_aspect(
            atoms=atoms,
            fig=fig,
            **plot_atoms_kwargs,
        )

        xyz_min = np.minimum(xyz_min, atoms_xyz_min)
        xyz_max = np.maximum(xyz_max, atoms_xyz_max)

    # fix aspect ratio
    if not np.allclose(xyz_min, xyz_max):
        fig.axes[-1].set_box_aspect(tuple(xyz_max[i] - xyz_min[i] for i in range(3)))

    return fig
