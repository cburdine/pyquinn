import numpy as np
from ase import Atoms
from ase.build.supercells import lattice_points_in_supercell
from ase.dft.kpoints import monkhorst_pack
from ase.neighborlist import primitive_neighbor_list


def get_special_k_pts(
    k_pts: np.ndarray,
    unit_cell: np.ndarray | None,
    periodic: np.ndarray | None,
    path: str | None = None,
    special_radius: float = None,
):

    if special_radius is None:
        special_radius = 1.1 * np.median(np.linalg.norm(k_pts[:-1] - k_pts[1:], axis=-1))

    # get the canonical band path and k-points (in fractional coordinates)
    path_pts = {}
    if unit_cell is None:
        path_pts["G"] = np.zeros(3)
    else:
        ase_cell = Atoms(cell=unit_cell, pbc=periodic).cell
        band_path = ase_cell.bandpath(path=path, npoints=256)

        for name, pt in band_path.special_points.items():
            path_pts[name] = pt

    # get the special points of the first Brillouin zone
    special_kpts = {
        name.replace("G", "Γ").replace("S", "Σ"): np.nonzero(np.linalg.norm(k_pts - pt, axis=-1) <= special_radius)[0]
        for name, pt in path_pts.items()
    }

    return special_kpts


def get_special_kpath_folds(
    special_kpts: dict,
    kpts_x: np.ndarray | None = None,
    x_radius: float | None = None,
):

    # construct fold points
    fold_pts = []
    for label, idxs in special_kpts.items():
        for i in idxs:
            x = i if kpts_x is None else kpts_x[i]
            fold_pts.append((x, label))

    fold_pts.sort()
    folds = [f[0] for f in fold_pts]
    fold_tokens = [(f[1],) for f in fold_pts]

    # Handle if the same special point was labeled twice consecutively
    if x_radius is not None:
        dx = (np.diff(folds) <= x_radius).nonzero()[0]
        for i in dx:
            if fold_tokens[i][-1] != fold_tokens[i + 1][-1]:
                fold_tokens[i + 1] = fold_tokens[i] + fold_tokens[i + 1]
            else:
                fold_tokens[i + 1] = fold_tokens[i]
            fold_tokens[i] = tuple()

    # Gather all nonempty fold labels
    folds = np.array(folds)
    fold_labels = np.array(["|".join(t) for t in fold_tokens])
    labeled_idxs = fold_labels != ""

    return folds[labeled_idxs], fold_labels[labeled_idxs]


def build_neighbor_list(
    positions: list, unit_cell: list, periodic: tuple[bool], cutoffs: list
) -> tuple[list[np.ndarray], list[np.ndarray]]:

    # use ASE's neighbor list subroutine to build
    # a primitive neighbor list
    I, J, R, S = primitive_neighbor_list(
        quantities="ijdS",
        pbc=periodic,
        cell=unit_cell,
        positions=positions,
        cutoff=cutoffs,
    )

    # build lists of positions and neighbor indices
    n_positions = len(positions)
    neighbor_j = [[] for _ in range(n_positions)]
    neighbor_s = [[] for _ in range(n_positions)]
    neighbor_r = [[] for _ in range(n_positions)]
    for i, j, r, s in zip(I, J, R, S, strict=False):

        # ignore inexact sphere collisions:
        ij_cutoff = cutoffs[i] + cutoffs[j]
        if r <= ij_cutoff:
            neighbor_j[i].append(j)
            neighbor_s[i].append(s)
            neighbor_r[i].append(r)

    neighbor_idxs = [np.argsort(r_arr) for r_arr in neighbor_r]

    neighbors = [np.array(arr)[idx] for arr, idx in zip(neighbor_j, neighbor_idxs, strict=False)]
    unit_cell_images = [np.array(arr)[idx] for arr, idx in zip(neighbor_s, neighbor_idxs, strict=False)]

    return neighbors, unit_cell_images


def build_dense_neighbor_list(
    positions: list, unit_cell: list, periodic: tuple[bool], cutoffs: list
) -> tuple[np.ndarray, np.ndarray]:

    neighbors, unit_cell_images = build_neighbor_list(
        positions=positions,
        unit_cell=unit_cell,
        periodic=periodic,
        cutoffs=cutoffs,
    )

    n_positions = len(positions)
    n_neighbors = max(n.shape[0] for n in neighbors)
    dense_neighbors = np.full((n_positions, n_neighbors), -1)
    dense_unit_cell_images = np.zeros((n_positions, n_neighbors, 3))

    for i in range(n_positions):

        # get neighbor and images arrays (& verify shapes match)
        neighbors_i = neighbors[i]
        images_i = unit_cell_images[i].reshape(-1, 3)
        assert neighbors_i.shape[0] == images_i.shape[0]

        dense_neighbors[i, : neighbors_i.shape[0]] = neighbors_i
        dense_unit_cell_images[i, : neighbors_i.shape[0]] = images_i

    return dense_neighbors, dense_unit_cell_images


def supercell_lattice_points(size: np.ndarray) -> tuple[np.ndarray]:

    matrix = size
    if matrix.ndim < 1:
        matrix = matrix * np.eye(3)
    elif matrix.ndim == 1:
        matrix = np.diag(matrix)
    if matrix.shape != (3, 3):
        raise ValueError(
            "Expected size to have a shape expandable to a 3x3 matrix " f"(got expanded shape: {matrix.shape})."
        )

    frac_lattice_points = lattice_points_in_supercell(matrix)
    lattice_points = np.dot(frac_lattice_points, matrix)

    return matrix, lattice_points


def monkhorst_pack_mesh(size: tuple[int] | int = 5):

    if isinstance(size, int):
        size = (size, size, size)

    frac_mesh = monkhorst_pack(size)
    return frac_mesh
