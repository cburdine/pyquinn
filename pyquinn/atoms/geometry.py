import numpy as np
from ase.neighborlist import primitive_neighbor_list


def build_neighbor_list(
        positions: list,
        unit_cell: list,
        periodic: tuple[bool],
        cutoffs: list) -> tuple[list[np.ndarray], list[np.ndarray]]:

    # use ASE's neighbor list subroutine to build
    # a primitive neighbor list
    I, J, R, S = primitive_neighbor_list(
        quantities='ijdS',
        pbc=periodic,
        cell=unit_cell,
        positions=positions,
        cutoff=cutoffs
    )

    # build lists of positions and neighbor indices
    n_positions = len(positions)
    neighbor_j = [[] for _ in range(n_positions)]
    neighbor_s = [[] for _ in range(n_positions)]
    neighbor_r = [[] for _ in range(n_positions)]
    for i, j, r, s in zip(I, J, R, S):

        # ignore inexact sphere collisions:
        ij_cutoff = cutoffs[i] + cutoffs[j]
        if r <= ij_cutoff:
            neighbor_j[i].append(j)
            neighbor_s[i].append(s)
            neighbor_r[i].append(r)

    neighbor_idxs = [np.argsort(r_arr) for r_arr in neighbor_r]

    neighbors = [
        np.array(arr)[idx]
        for arr, idx in zip(neighbor_j, neighbor_idxs)
    ]
    unit_cell_images = [
        np.array(arr)[idx]
        for arr, idx in zip(neighbor_s, neighbor_idxs)
    ]

    return neighbors, unit_cell_images


def build_dense_neighbor_list(
        positions: list,
        unit_cell: list,
        periodic: tuple[bool],
        cutoffs: list) -> tuple[np.ndarray, np.ndarray]:

    neighbors, unit_cell_images = build_neighbor_list(
        positions=positions,
        unit_cell=unit_cell,
        periodic=periodic,
        cutoffs=cutoffs
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

        dense_neighbors[i, :neighbors_i.shape[0]] = neighbors_i
        dense_unit_cell_images[i, :neighbors_i.shape[0]] = images_i

    return dense_neighbors, dense_unit_cell_images
