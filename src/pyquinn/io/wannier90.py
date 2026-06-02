import numpy as np

from pyquinn.types import Operator


def read_wannier90_hamiltonian(
    file: str, site_indices: list[int], orbital_indices: list[int]
) -> Operator:
    """Reads a Wannier90 Hamiltonian file (typically of the form *_hr.dat).

    Args:
        file (str): Path to file.
        site_indices (list[int]): List of site indices of each Wannier orbital.
            These indices correspond to sites in an associated AtomGraph.
        orbital_indices (list[int]): List of orbital indices of each Wannier orbital.
            These indices correspond to the on-site orbitals in an associated AtomGraph.

    Raises:
        ValueError: If site indices are invalid

    Returns:
        Operator: The parsed real-space Hamiltonian, represented as
        an Operator.
    """

    with open(file) as f:

        # Read header (ignore header lines)
        header_lines = []

        while start_line := f.readline().strip():
            if start_line.isdigit():
                break

            header_lines.append(start_line)

        num_wann = int(start_line)
        nrpts = int(f.readline().strip())

        if len(site_indices) != num_wann:
            raise ValueError(
                f"Expected # wannier orbitals ({num_wann}) to equal len(site_indices) (got: {len(site_indices)})."
            )
        if len(orbital_indices) != num_wann:
            raise ValueError(
                f"Expected # wannier orbitals ({num_wann}) to equal len(orbital_indices) (got: {len(orbital_indices)})."
            )

        degeneracies = []
        while len(degeneracies) < nrpts:
            degeneracies.extend([int(x) for x in f.readline().split()])
        degeneracies = np.array(degeneracies)

        assert len(degeneracies) == nrpts

        H_entries = {}

        for i in range(nrpts):
            d_R = degeneracies[i]
            for _ in range(num_wann * num_wann):
                line = f.readline().split()
                Rx, Ry, Rz, m, n = map(int, line[:5])
                Re, Im = map(float, line[5:])
                idx = (m - 1, n - 1)
                idx_t = (n - 1, m - 1)
                if idx not in H_entries:
                    H_entries[idx] = {}
                # if idx_t not in H_entries:
                #     H_entries[idx_t] = {}

                z = complex((Re + 1j * Im) * d_R)
                z_t = complex((Re - 1j * Im) * d_R)

                H_entries[idx] |= {(Rx, Ry, Rz): z}
                # H_entries[idx_t] |= {(Rx, Ry, Rz): z_t}

        while line := f.readline():
            print("extra line: ", line)

        onsite_values = [0.0 for _ in range(num_wann)]
        couplings = []
        coupling_values = []
        images = []

        for i, j in sorted(H_entries.keys()):
            R_values = H_entries[(i, j)]
            for R, z in R_values.items():
                if i == j and R == (0, 0, 0):
                    onsite_values[i] += float(np.real(z))
                else:
                    couplings.append((i, j))
                    coupling_values.append(z)
                    images.append(R)

        return Operator(
            site_indices=np.array(site_indices),
            orbital_indices=np.array(orbital_indices),
            onsite_values=np.array(onsite_values),
            couplings=np.array(couplings),
            coupling_values=np.array(coupling_values),
            images=np.array(images),
        )
