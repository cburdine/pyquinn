from abc import abstractmethod

import torch
import torch.nn as nn

from pyquinn.torch.nn.eigensolver import Eigensolver
from pyquinn.torch.types import TorchOperator
from pyquinn.torch.types.operator import build_matrix, build_nonperiodic_matrix
from pyquinn.torch.util import get_activation_fn


class Hamiltonian(nn.Module):
    """Abstract base class for a Pytorch Hamiltonian model.

    This class represents a Hamiltonian of a quantum system
    that is parameterized by a neural network. All subclasses
    of this class must at a minimum provide an
    implementation of

    `get_hamiltonian_matrix()

    and

    `get_hamiltonian_key()`

    The default behavior of the `forward()` method is to return
    the eigenvalues of the quantum system, obtained through
    the diagonalization of the Hamiltonian matrix via a
    concrete instance of the `pyquinn.torch.Eigensolver` class.
    """

    def __init__(self, eigensolver: str | Eigensolver | None = None):
        """Constructor for a `Hamiltonian` instance with an associated
        `pyquinn.torch.Eigensolver` instance. All Hamiltonian instances are
        required to call this super constructor with the given eigensolver
        name.

        If a string is supplied, an eigensolver returned by
        `Eigensolver.from_name(eigensolver)` is used.

        Args:
            eigensolver (str | Eigensolver | None, optional): The Eigensolver
                associated with this Hamiltonian.
        """

        super().__init__()
        self.eigensolver = Eigensolver.from_name(eigensolver) if isinstance(eigensolver, str) else eigensolver

    @abstractmethod
    def hamiltonian_matrix(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Returns the Hamiltonian of an atomic system in matrix form.
        This abstract method provides the recommended signature of
        all subclasses that provide an implementation of this function.
        Note that some subclasses may change or add arguments to this
        signature.

        Args:
            atomic_numbers (torch.IntTensor):
                Atomic numbers of each site in the system (Hydrogen = 1, etc.).
                (shape: [N_sites])
            positions (torch.Tensor): Positions of atomic sites in 3D space
                (shape: [N_sites, 3])
            neighbors (torch.IntTensor): List of indices of neighboring
                                         (i.e. interacting) sites adjacent
                                         to each site in the lattice.
                                         Negative indices are ignored.
                                         (shape: [N_sites, max_neighbors])

        Raises:
            NotImplementedError: If this method is called directly
                                 (must be reimplemented in a subclass)

        Returns:
            torch.Tensor: Hamiltonian matrix (shape: [N_orbitals, N_orbitals])
        """
        raise NotImplementedError()

    @abstractmethod
    def orbital_names(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> list[str]:
        """Returns a list of string labels for the entries of the Hamiltonian
           matrix. Entries in the Hamiltonian matrix are expected to follow the
           ordering of `atomic_numbers` and `neighbors`, though each site may
           contain a variable number of orbitals.

        Args:
            atomic_numbers (torch.IntTensor):
                Atomic numbers of each site in the system (Hydrogen = 1, etc.).
                (shape: [N_sites])
            positions (torch.Tensor): Positions of atomic sites in 3D space
                (shape: [N_sites, 3])
            neighbors (torch.IntTensor): List of indices of neighboring
                                         (i.e. interacting) sites adjacent
                                         to each site in the lattice.
                                         Negative indices are ignored.
                                         (shape: [N_sites, max_neighbors])

        Raises:
            NotImplementedError: If this method is called directly
                                 (must be reimplemented in a subclass)

        Returns:
            list[str]: A list of string labels for the entries of the
                       Hamiltonian matrix. Must have the same length
                       as the dimension of the Hamiltonian matrix.
        """
        raise NotImplementedError()

    @abstractmethod
    def orbital_indices(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> tuple[torch.IntTensor, torch.IntTensor]:
        """Returns a list of on-site orbital indices for the entries of the
           Hamiltonian matrix. Entries in the Hamiltonian matrix are
           expected to follow the ordering of `atomic_numbers` and `neighbors`,
           though each site may contain a variable number of orbitals.

        Args:
            atomic_numbers (torch.IntTensor):
                Atomic numbers of each site in the system (Hydrogen = 1, etc.).
                (shape: [N_sites])
            positions (torch.Tensor): Positions of atomic sites in 3D space
                (shape: [N_sites, 3])
            neighbors (torch.IntTensor): List of indices of neighboring
                                         (i.e. interacting) sites adjacent
                                         to each site in the lattice.
                                         Negative indices are ignored.
                                         (shape: [N_sites, max_neighbors])

        Returns:
            tuple[torch.IntTensor, torch.IntTensor]: List of site indices i
                (0 < i < N_sites) and orbital indices j corresponding
                to entries in the Hamiltonian matrix. Both lists
                must have the same dimension as the Hamiltonian matrix.

        """
        raise NotImplementedError()

    @abstractmethod
    def orbital_quantum_numbers(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Returns a list of orbital quantum numbers for the entries of the
           Hamiltonian matrix. Entries in the matrix are expected to follow the
           ordering of `atomic_numbers` and `neighbors`, though each site may
           contain a variable number of orbitals.

           The type of quantum numbers returned depends on the

        Args:
            atomic_numbers (torch.IntTensor):
                Atomic numbers of each site in the system (Hydrogen = 1, etc.).
                (shape: [N_sites])
            positions (torch.Tensor): Positions of atomic sites in 3D space
                (shape: [N_sites, 3])
            neighbors (torch.IntTensor): List of indices of neighboring
                                         (i.e. interacting) sites adjacent
                                         to each site in the lattice.
                                         Negative indices are ignored.
                                         (shape: [N_sites, max_neighbors])

        Returns:
            torch.Tensor: List of orbital quantum numbers corresponding to
                entries in the Hamiltonian matrix. The first dimension
                of this tensor must equal the dimension of the Hamiltonian
                matrix.

        """
        raise NotImplementedError()

    def eigenstates(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the eigenenergies and associated eigenstates of the
        Hamiltonian matrix.

        Args:
            atomic_numbers (torch.IntTensor):
                Atomic numbers of each site in the system (Hydrogen = 1, etc.).
                (shape: [N_sites])
            positions (torch.Tensor): Positions of atomic sites in 3D space
                (shape: [N_sites, 3])
            neighbors (torch.IntTensor): List of indices of neighboring
                                         (i.e. interacting) sites adjacent
                                         to each site in the lattice.
                                         Negative indices are ignored.
                                         (shape: [N_sites, max_neighbors])

        Raises:
            ValueError: If an Eigensolver is not supplied.

        Returns:
            tuple[torch.Tensor,torch.Tensor]: Eigenenergies and associated
                                              matrix of eigenstates.
                                              (shape: [N_eigstates],
                                              [N_eigstates, N_orbitals])
        """

        H = self.hamiltonian_matrix(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            **kwargs,
        )

        if self.eigensolver is None:
            raise ValueError("An eigensolver must be supplied.")

        return self.eigensolver.diagonalize(H)

    def eigenenergies(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Returns the eigenenergies of an atomic system's Hamiltonian.

        Args:
            atomic_numbers (torch.IntTensor):
                Atomic numbers of each site in the system (Hydrogen = 1, etc.).
                (shape: [N_sites])
            positions (torch.Tensor): Positions of atomic sites in 3D space
                (shape: [N_sites, 3])
            neighbors (torch.IntTensor): List of indices of neighboring
                                         (i.e. interacting) sites adjacent
                                         to each site in the lattice.
                                         Negative indices are ignored.
                                         (shape: [N_sites, max_neighbors])

        Raises:
            ValueError: If an Eigensolver is not supplied.

        Returns:
            torch.Tensor: Eigenenergies of the system (shape: [N_eigstates])
        """
        H = self.hamiltonian_matrix(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            **kwargs,
        )

        if self.eigensolver is None:
            raise ValueError("An eigensolver must be supplied.")

        return self.eigensolver(H)

    def forward(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass function for a HamiltonianModel. Default behavior is
        to return the eigenenergies of an atomic system's Hamiltonian. In any
        concrete subclasses, this function should have a signature that
        matches `self.hamiltonian_matrix()`.

        Args:
            atomic_numbers (torch.IntTensor):
                Atomic numbers of each site in the system (Hydrogen = 1, etc.).
                (shape: [N_sites])
            positions (torch.Tensor): Positions of atomic sites in 3D space
                (shape: [N_sites, 3])
            neighbors (torch.IntTensor): List of indices of neighboring
                                         (i.e. interacting) sites adjacent
                                         to each site in the lattice.
                                         Negative indices are ignored.
                                         (shape: [N_sites, max_neighbors])

        Returns:
            torch.Tensor: Eigenenergies of the system (shape: [N_eigstates])
        """

        return self.eigenenergies(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            **kwargs,
        )


class ExponentialSKCoefficients(nn.Module):
    """Distance-parameterized Slater-Koster coefficients.

    This module predicts a small set of Slater-Koster integrals as a function of
    interatomic distance `r` and atomic numbers `(z_i, z_j)`:

        V_t(z_i, z_j, r) = c_t * s_t[z_i] * s_t[z_j] * exp(-beta_t * r)

    where `t` enumerates the supported integral types.

    Notes:
        - This is intentionally simple; for more expressive tight-binding fits,
          pass a custom `sk_coefficients` module to `SKHamiltonian`.
    """

    COEFF_NAMES = ("ss_sigma", "sp_sigma", "pp_sigma", "pp_pi")

    def __init__(
        self,
        z_max: int = 118,
        init_scale: float = 0.1,
        init_decay: float = 1.0,
        element_dependent: bool = True,
    ) -> None:
        super().__init__()

        self.z_max = int(z_max)
        n_coeff = len(self.COEFF_NAMES)

        self.coeff = nn.Parameter(torch.randn(n_coeff) * float(init_scale))

        # Non-negative decay via softplus; can be set near-zero by driving `raw_decay` negative.
        self.raw_decay = nn.Parameter(torch.full((n_coeff,), float(init_decay)))

        if element_dependent:
            self.element_scale = nn.Embedding(self.z_max + 1, n_coeff)
            with torch.no_grad():
                self.element_scale.weight.fill_(1.0)
        else:
            self.element_scale = None

    def forward(
        self,
        z_i: torch.IntTensor,
        z_j: torch.IntTensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Return coefficients with shape `[E, n_coeff]` for `E` site pairs."""
        if z_i.shape != z_j.shape:
            raise ValueError(
                f"Expected `z_i` and `z_j` to have same shape (got: {tuple(z_i.shape)} vs {tuple(z_j.shape)})"
            )
        if r.shape != z_i.shape:
            raise ValueError(f"Expected `r` to have same shape as `z_i` (got: {tuple(r.shape)} vs {tuple(z_i.shape)})")

        coeff = self.coeff.to(dtype=r.dtype, device=r.device).view(1, -1)
        decay = torch.nn.functional.softplus(self.raw_decay.to(dtype=r.dtype, device=r.device)).view(1, -1)
        envelope = torch.exp(-decay * r.to(dtype=r.dtype, device=r.device).view(-1, 1))

        if self.element_scale is None:
            return coeff * envelope

        s_i = self.element_scale(z_i.to(device=r.device)).to(dtype=r.dtype)
        s_j = self.element_scale(z_j.to(device=r.device)).to(dtype=r.dtype)

        return coeff * (s_i * s_j) * envelope


class SKHamiltonian(Hamiltonian):
    """Slater-Koster tight-binding Hamiltonian with distance-parameterized integrals.

    This Hamiltonian builds a real symmetric tight-binding matrix from a minimal
    s/p orbital basis per element using standard Slater-Koster angular factors.

    Supported orbital shells (per site):
        - `l=0` (s)
        - `l=1` (p_x, p_y, p_z)

    Supported Slater-Koster integrals:
        - `V_ss_sigma(r)`
        - `V_sp_sigma(r)`
        - `V_pp_sigma(r)`
        - `V_pp_pi(r)`

    Orbitals are ordered per-site by the order of `orbital_shells[z]`, with each
    shell expanded to its cartesian components (s, then p_x/p_y/p_z for l=1).
    """

    def __init__(
        self,
        eigensolver: str | Eigensolver = "torch_eigh",
        orbital_shells: dict[int, list[int]] | None = None,
        sk_coefficients: nn.Module | None = None,
        energy_scale: float = 1.0,
        z_max: int = 118,
    ) -> None:
        super().__init__(eigensolver=eigensolver)

        if orbital_shells is None or not orbital_shells:
            raise ValueError("`orbital_shells` must be a non-empty dict, e.g. `{6: [0, 1]}` for carbon sp.")

        self.energy_scale = float(energy_scale)
        self.z_max = int(z_max)

        # Validate and build padded shell tensor: `[atomic_number, shell_idx] -> l`.
        shells = {int(z): list(v) for z, v in orbital_shells.items()}
        self._validate_orbital_shells(shells)

        z_dim = self.z_max + 1
        n_shells_max = max(len(v) for v in shells.values())
        shell_idxs = torch.full((z_dim, n_shells_max), -1, dtype=torch.int64)
        for z, l_list in shells.items():
            if z > self.z_max:
                raise ValueError(f"Element {z} exceeds z_max={self.z_max}.")
            for i, l in enumerate(l_list):
                shell_idxs[z, i] = int(l)

        self.register_buffer("orbital_shell_idxs", shell_idxs)

        # Per-element local-orbital tables (padded) to avoid repeated Python loops.
        max_orbitals = 0
        for l_list in shells.values():
            max_orbitals = max(max_orbitals, sum((2 * int(l) + 1) for l in l_list))

        orbital_l_map = torch.full((z_dim, max_orbitals), -1, dtype=torch.int64)
        orbital_comp_map = torch.full((z_dim, max_orbitals), -1, dtype=torch.int64)
        orbital_shell_map = torch.full((z_dim, max_orbitals), -1, dtype=torch.int64)
        n_orbitals_by_z = torch.zeros((z_dim,), dtype=torch.int64)

        for z, l_list in shells.items():
            k = 0
            for shell_idx, l in enumerate(l_list):
                l_int = int(l)
                for comp in range(2 * l_int + 1):
                    orbital_l_map[z, k] = l_int
                    orbital_comp_map[z, k] = int(comp)
                    orbital_shell_map[z, k] = int(shell_idx)
                    k += 1
            n_orbitals_by_z[z] = k

        self.register_buffer("orbital_l_map", orbital_l_map)
        self.register_buffer("orbital_comp_map", orbital_comp_map)
        self.register_buffer("orbital_shell_map", orbital_shell_map)
        self.register_buffer("n_orbitals_by_z", n_orbitals_by_z)

        # Trainable onsite energies per (element, shell instance).
        self.onsite_energies = nn.Parameter(torch.zeros((z_dim, n_shells_max), dtype=torch.float32))

        # Trainable SK integrals.
        if sk_coefficients is None:
            self.sk_coefficients = ExponentialSKCoefficients(z_max=self.z_max)
        else:
            self.sk_coefficients = sk_coefficients

    def _validate_orbital_shells(self, shells: dict[int, list[int]]) -> None:
        """Validate `orbital_shells` for this Hamiltonian.

        Subclasses can override this to support additional shells.
        """
        for z, l_list in shells.items():
            if z <= 0:
                raise ValueError(f"Expected positive atomic number in `orbital_shells`, got: {z}")
            for l in l_list:
                if int(l) not in (0, 1):
                    raise ValueError(
                        f"`SKHamiltonian` currently supports only l in {{0,1}}, got: l={l} for element {z}."
                    )

    def _orbital_counts(self, atomic_numbers: torch.IntTensor) -> tuple[torch.IntTensor, torch.IntTensor]:
        n_orbitals = self.n_orbitals_by_z[atomic_numbers.to(torch.int64)]
        if (n_orbitals <= 0).any():
            bad = torch.nonzero(n_orbitals <= 0, as_tuple=False).view(-1)
            bad_z = atomic_numbers[bad].detach().cpu().tolist()
            raise ValueError(
                "Missing orbital shells for some atomic numbers. "
                f"Got atomic_numbers={bad_z}; provide entries for these elements in `orbital_shells`."
            )
        offsets = torch.cumsum(n_orbitals, dim=0) - n_orbitals
        return n_orbitals.to(torch.int64), offsets.to(torch.int64)

    def orbital_indices(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> tuple[torch.IntTensor, torch.IntTensor]:
        n_orbitals, _offsets = self._orbital_counts(atomic_numbers)

        max_orb = int(torch.max(n_orbitals).item()) if n_orbitals.numel() > 0 else 0
        if max_orb == 0:
            return (
                torch.empty((0,), dtype=torch.int64, device=atomic_numbers.device),
                torch.empty((0,), dtype=torch.int64, device=atomic_numbers.device),
            )

        site_idxs, orbital_idxs = torch.nonzero(
            torch.arange(max_orb, device=atomic_numbers.device).view(1, -1) < n_orbitals.view(-1, 1),
            as_tuple=True,
        )
        return site_idxs.to(torch.int64), orbital_idxs.to(torch.int64)

    def orbital_names(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> list[str]:
        names: list[str] = []
        for z in atomic_numbers.tolist():
            l_values = self.orbital_shell_idxs[int(z)]
            for l in l_values.tolist():
                if l < 0:
                    continue
                if l == 0:
                    names.append("s")
                elif l == 1:
                    names.extend(["p_x", "p_y", "p_z"])
                else:
                    names.extend([f"l{l}_{i}" for i in range(2 * int(l) + 1)])
        return names

    def orbital_quantum_numbers(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> torch.Tensor:
        # Returns `(l, comp)` where `comp` enumerates components in the orbital order.
        site_idxs, orb_idxs = self.orbital_indices(atomic_numbers, positions, neighbors)
        if site_idxs.numel() == 0:
            return torch.empty((0, 2), dtype=torch.int64, device=atomic_numbers.device)

        z = atomic_numbers[site_idxs].to(torch.int64)
        ls = self.orbital_l_map[z, orb_idxs].to(torch.int64)
        comps = self.orbital_comp_map[z, orb_idxs].to(torch.int64)
        return torch.stack((ls, comps), dim=-1)

    @staticmethod
    def _undirected_site_pairs(neighbors: torch.IntTensor) -> torch.IntTensor:
        site_i, adj = torch.nonzero(neighbors >= 0, as_tuple=True)
        if site_i.numel() == 0:
            return torch.empty((0, 2), dtype=torch.int64, device=neighbors.device)
        site_j = neighbors[site_i, adj].to(torch.int64)
        a = torch.minimum(site_i.to(torch.int64), site_j)
        b = torch.maximum(site_i.to(torch.int64), site_j)
        pairs = torch.unique(torch.stack((a, b), dim=1), dim=0)
        return pairs[pairs[:, 0] != pairs[:, 1]]

    @staticmethod
    def _site_pair_images(
        site_pairs: torch.IntTensor,
        neighbors: torch.IntTensor,
        unit_cell_images: torch.IntTensor,
    ) -> torch.IntTensor:
        """Resolve images for undirected site pairs, returning src->dst images."""
        if site_pairs.ndim != 2 or site_pairs.shape[1] != 2:
            raise ValueError(f"Expected `site_pairs` to have shape [P,2], got: {tuple(site_pairs.shape)}")
        if unit_cell_images.ndim != 3 or unit_cell_images.shape[-1] != 3:
            raise ValueError(
                "Expected `unit_cell_images` to have shape [N_sites, N_neighbors, 3], "
                f"got: {tuple(unit_cell_images.shape)}"
            )
        if neighbors.shape != unit_cell_images.shape[:2]:
            raise ValueError(
                "Expected `neighbors` and `unit_cell_images` to agree in the first two dims "
                f"(got: {tuple(neighbors.shape)} vs {tuple(unit_cell_images.shape[:2])})"
            )

        src = site_pairs[:, 0].to(torch.int64)
        dst = site_pairs[:, 1].to(torch.int64)

        nbrs_src = neighbors[src]
        match_src = nbrs_src == dst.unsqueeze(1)
        has_src = match_src.any(dim=1)
        slot_src = match_src.to(torch.int64).argmax(dim=1)
        img_src = unit_cell_images[src, slot_src].to(torch.int64)

        nbrs_dst = neighbors[dst]
        match_dst = nbrs_dst == src.unsqueeze(1)
        has_dst = match_dst.any(dim=1)
        slot_dst = match_dst.to(torch.int64).argmax(dim=1)
        img_dst = (-unit_cell_images[dst, slot_dst]).to(torch.int64)

        has_any = has_src | has_dst
        if not bool(has_any.all().item()):
            bad = torch.nonzero(~has_any, as_tuple=False).view(-1)
            bad_pairs = site_pairs[bad].detach().cpu().tolist()
            raise ValueError(
                "Could not resolve `unit_cell_images` for some neighbor pairs. " f"Missing pairs (src,dst): {bad_pairs}"
            )

        if bool((has_src & has_dst).any().item()):
            both = has_src & has_dst
            if not bool((img_src[both] == img_dst[both]).all().item()):
                raise ValueError(
                    "Inconsistent `unit_cell_images` for some neighbor pairs: "
                    "src->dst image does not equal -(dst->src) image."
                )

        return torch.where(has_src.view(-1, 1), img_src, img_dst).to(torch.int64)

    @staticmethod
    def _coupling_images(
        couplings: torch.IntTensor,
        site_indices: torch.IntTensor,
        neighbors: torch.IntTensor,
        unit_cell_images: torch.IntTensor,
    ) -> torch.IntTensor:
        """Resolve per-coupling unit-cell images for a Bloch Hamiltonian.

        Assumes `couplings` are ordered as `[couplings_u; couplings_u.flip(1)]`,
        so the second half is the Hermitian counterpart and uses `-images_u`.
        """
        if couplings.ndim != 2 or couplings.shape[1] != 2:
            raise ValueError(f"Expected `couplings` to have shape [E,2], got: {tuple(couplings.shape)}")
        if couplings.shape[0] % 2 != 0:
            raise ValueError(f"Expected even number of couplings, got: E={int(couplings.shape[0])}")
        if unit_cell_images.ndim != 3 or unit_cell_images.shape[-1] != 3:
            raise ValueError(
                "Expected `unit_cell_images` to have shape [N_sites, N_neighbors, 3], "
                f"got: {tuple(unit_cell_images.shape)}"
            )
        if neighbors.shape != unit_cell_images.shape[:2]:
            raise ValueError(
                "Expected `neighbors` and `unit_cell_images` to agree in the first two dims "
                f"(got: {tuple(neighbors.shape)} vs {tuple(unit_cell_images.shape[:2])})"
            )

        device = couplings.device
        half = couplings.shape[0] // 2

        couplings_u = couplings[:half]
        site_a = site_indices[couplings_u[:, 0]].to(torch.int64)
        site_b = site_indices[couplings_u[:, 1]].to(torch.int64)

        images_u = torch.zeros((half, 3), dtype=torch.int64, device=device)
        offsite = site_a != site_b
        if offsite.any():
            pairs = torch.stack((site_a[offsite], site_b[offsite]), dim=1)
            unique_pairs, inv = torch.unique(pairs, dim=0, return_inverse=True)
            src = unique_pairs[:, 0]
            dst = unique_pairs[:, 1]

            nbrs_src = neighbors[src]
            match_src = nbrs_src == dst.unsqueeze(1)
            has_src = match_src.any(dim=1)
            slot_src = match_src.to(torch.int64).argmax(dim=1)
            img_src = unit_cell_images[src, slot_src].to(torch.int64)

            nbrs_dst = neighbors[dst]
            match_dst = nbrs_dst == src.unsqueeze(1)
            has_dst = match_dst.any(dim=1)
            slot_dst = match_dst.to(torch.int64).argmax(dim=1)
            img_dst = (-unit_cell_images[dst, slot_dst]).to(torch.int64)

            has_any = has_src | has_dst
            if not bool(has_any.all().item()):
                bad = torch.nonzero(~has_any, as_tuple=False).view(-1)
                bad_pairs = unique_pairs[bad].detach().cpu().tolist()
                raise ValueError(
                    "Could not resolve `unit_cell_images` for some neighbor pairs. "
                    f"Missing pairs (src,dst): {bad_pairs}"
                )

            if bool((has_src & has_dst).any().item()):
                both = has_src & has_dst
                if not bool((img_src[both] == img_dst[both]).all().item()):
                    raise ValueError(
                        "Inconsistent `unit_cell_images` for some neighbor pairs: "
                        "src->dst image does not equal -(dst->src) image."
                    )

            img = torch.where(has_src.view(-1, 1), img_src, img_dst)
            images_u[offsite] = img[inv]

        images = torch.cat((images_u, -images_u), dim=0)
        return images

    def hamiltonian_operator_entries(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor | None = None,
        unit_cell_images: torch.IntTensor | None = None,
        assume_bidirectional_neighbors: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.IntTensor, torch.Tensor, torch.IntTensor]:
        n_orbitals, offsets = self._orbital_counts(atomic_numbers)

        site_idxs, local_orb_idxs = self.orbital_indices(atomic_numbers, positions, neighbors)
        z_orb = atomic_numbers[site_idxs].to(torch.int64)
        shell_idx = self.orbital_shell_map[z_orb, local_orb_idxs].to(torch.int64)
        onsite_values = self.onsite_energies[z_orb, shell_idx].to(dtype=positions.dtype) * self.energy_scale

        site_i, adj = torch.nonzero(neighbors >= 0, as_tuple=True)
        if site_i.numel() == 0:
            return (
                onsite_values,
                torch.empty((0, 2), dtype=torch.int64, device=positions.device),
                torch.empty((0,), dtype=onsite_values.dtype, device=positions.device),
                torch.empty((0, 3), dtype=torch.int64, device=positions.device),
            )

        site_j = neighbors[site_i, adj].to(torch.int64)

        images_ij = torch.zeros((site_i.shape[0], 3), dtype=torch.int64, device=positions.device)
        if unit_cell_images is not None:
            if unit_cell is None:
                raise ValueError("`unit_cell_images` requires `unit_cell`.")
            images_ij = unit_cell_images.to(device=positions.device, dtype=torch.int64)[site_i, adj]

        # No deduplication: treat each directed neighbor entry as a distinct coupling.
        # This matches the repo convention where neighbor lists contain both directions
        # (i->j,image) and (j->i,-image) explicitly.
        #
        # Note: `assume_bidirectional_neighbors` is kept for backwards compatibility,
        # but is not used now that deduplication has been removed.
        _ = assume_bidirectional_neighbors

        src = site_i.to(torch.int64)
        dst = site_j.to(torch.int64)
        edge_images = images_ij.to(torch.int64)

        # Avoid double-counting the on-site diagonal if a neighbor list includes self edges with zero image.
        self_edge = (src == dst) & ~(edge_images != 0).any(dim=1)
        if bool(self_edge.any().item()):
            keep = ~self_edge
            src = src[keep]
            dst = dst[keep]
            edge_images = edge_images[keep]

        disp = positions[dst] - positions[src]
        if unit_cell is not None:
            shift = edge_images.to(dtype=positions.dtype, device=positions.device) @ unit_cell.to(
                dtype=positions.dtype, device=positions.device
            )
            disp = disp + shift

        r = torch.linalg.norm(disp, dim=-1)
        if bool((r <= 0).any().item()):
            raise ValueError("Encountered zero-distance neighbor pair while building SK Hamiltonian.")
        u = disp / r.view(-1, 1)

        z_src = atomic_numbers[src].to(torch.int64)
        z_dst = atomic_numbers[dst].to(torch.int64)

        coeffs = self.sk_coefficients(z_src, z_dst, r)  # [P, n_coeff]
        if coeffs.ndim != 2 or coeffs.shape[0] != src.shape[0]:
            raise ValueError(
                "Expected `sk_coefficients(z_i,z_j,r)` to return a tensor with shape [P, n_coeff]; "
                f"got: {tuple(coeffs.shape)} for P={int(src.shape[0])}"
            )

        # Map coefficient columns by name when available; fall back to fixed ordering.
        names = getattr(
            self.sk_coefficients,
            "COEFF_NAMES",
            ExponentialSKCoefficients.COEFF_NAMES,
        )
        name_to_idx = {str(n): i for i, n in enumerate(names)}
        required = ("ss_sigma", "sp_sigma", "pp_sigma", "pp_pi")
        if any(k not in name_to_idx for k in required):
            raise ValueError(
                "Unsupported `sk_coefficients`: expected COEFF_NAMES to include " f"{required}, got: {tuple(names)}"
            )

        v_ss = coeffs[:, name_to_idx["ss_sigma"]]
        v_sp = coeffs[:, name_to_idx["sp_sigma"]]
        v_pp_sigma = coeffs[:, name_to_idx["pp_sigma"]]
        v_pp_pi = coeffs[:, name_to_idx["pp_pi"]]

        couplings_u: list[torch.IntTensor] = []
        values_u: list[torch.Tensor] = []
        images_list_u: list[torch.IntTensor] = []

        for p in range(int(src.shape[0])):
            i = int(src[p].item())
            j = int(dst[p].item())

            n_i = int(n_orbitals[i].item())
            n_j = int(n_orbitals[j].item())
            if n_i == 0 or n_j == 0:
                continue

            idx_i = torch.arange(n_i, device=positions.device, dtype=torch.int64) + offsets[i]
            idx_j = torch.arange(n_j, device=positions.device, dtype=torch.int64) + offsets[j]

            z_i = int(atomic_numbers[i].item())
            z_j = int(atomic_numbers[j].item())
            l_i = self.orbital_l_map[z_i, :n_i].to(torch.int64)
            c_i = self.orbital_comp_map[z_i, :n_i].to(torch.int64)
            l_j = self.orbital_l_map[z_j, :n_j].to(torch.int64)
            c_j = self.orbital_comp_map[z_j, :n_j].to(torch.int64)

            if bool((l_i > 1).any().item()) or bool((l_j > 1).any().item()):
                raise ValueError("`SKHamiltonian` currently supports only s/p orbitals (l <= 1).")

            uu = u[p].to(dtype=positions.dtype)  # [3]
            ux, uy, uz = uu[0], uu[1], uu[2]
            u_xyz = torch.stack((ux, uy, uz), dim=0)

            Vss = v_ss[p].to(dtype=positions.dtype) * self.energy_scale
            Vsp = v_sp[p].to(dtype=positions.dtype) * self.energy_scale
            Vsig = v_pp_sigma[p].to(dtype=positions.dtype) * self.energy_scale
            Vpi = v_pp_pi[p].to(dtype=positions.dtype) * self.energy_scale

            block = torch.zeros((n_i, n_j), dtype=positions.dtype, device=positions.device)

            for a in range(n_i):
                la = int(l_i[a].item())
                ca = int(c_i[a].item())
                for b in range(n_j):
                    lb = int(l_j[b].item())
                    cb = int(c_j[b].item())

                    if la == 0 and lb == 0:
                        block[a, b] = Vss
                    elif la == 0 and lb == 1:
                        block[a, b] = u_xyz[cb] * Vsp
                    elif la == 1 and lb == 0:
                        block[a, b] = -u_xyz[ca] * Vsp
                    elif la == 1 and lb == 1:
                        uab = u_xyz[ca] * u_xyz[cb]
                        delta = 1.0 if ca == cb else 0.0
                        block[a, b] = uab * Vsig + (delta - uab) * Vpi
                    else:
                        raise ValueError(f"Unexpected orbital angular momentum (l_a={la}, l_b={lb}).")

            couplings_u.append(torch.cartesian_prod(idx_i, idx_j))
            values_u.append(block.reshape(-1))
            img = edge_images[p].to(dtype=torch.int64, device=positions.device).view(1, 3)
            images_list_u.append(img.expand(n_i * n_j, 3))

        if not couplings_u:
            return (
                onsite_values,
                torch.empty((0, 2), dtype=torch.int64, device=positions.device),
                torch.empty((0,), dtype=onsite_values.dtype, device=positions.device),
                torch.empty((0, 3), dtype=torch.int64, device=positions.device),
            )

        couplings_u_t = torch.cat(couplings_u, dim=0)
        values_u_t = torch.cat(values_u, dim=0)
        images_u_t = torch.cat(images_list_u, dim=0).to(torch.int64)

        return (
            onsite_values,
            couplings_u_t.to(torch.int64),
            values_u_t,
            images_u_t.to(torch.int64),
        )

    def hamiltonian_operator(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor | None = None,
        unit_cell_images: torch.IntTensor | None = None,
        **kwargs,
    ) -> TorchOperator:
        onsite_values, couplings, coupling_values, images = self.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            **kwargs,
        )

        site_indices, orbital_indices = self.orbital_indices(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

        return TorchOperator(
            onsite_values=onsite_values,
            couplings=couplings.to(torch.int64),
            coupling_values=coupling_values,
            site_indices=site_indices.to(torch.int64),
            orbital_indices=orbital_indices.to(torch.int64),
            images=images.to(torch.int64),
        )

    def hamiltonian_matrix(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor | None = None,
        unit_cell_images: torch.IntTensor | None = None,
        k_pts: torch.Tensor | None = None,
        sparse: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        onsite_values, couplings, coupling_values, images = self.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            **kwargs,
        )

        periodic = (unit_cell is not None) or (unit_cell_images is not None) or (k_pts is not None)
        if not periodic:
            return build_nonperiodic_matrix(
                onsite_values=onsite_values,
                couplings=couplings,
                coupling_values=coupling_values,
            )

        if unit_cell is None or unit_cell_images is None:
            raise ValueError("Periodic `hamiltonian_matrix` requires both `unit_cell` and `unit_cell_images`.")

        if k_pts is None:
            k_pts = torch.zeros((1, 3), device=positions.device, dtype=positions.dtype)
        elif not isinstance(k_pts, torch.Tensor):
            k_pts = torch.as_tensor(k_pts, device=positions.device, dtype=positions.dtype)
        else:
            k_pts = k_pts.to(device=positions.device, dtype=positions.dtype)

        frac_positions = positions @ torch.linalg.pinv(unit_cell.to(device=positions.device, dtype=positions.dtype))

        site_indices, _orbital_indices = self.orbital_indices(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )

        return build_matrix(
            onsite_values=onsite_values,
            couplings=couplings.to(torch.int64),
            coupling_values=coupling_values,
            site_indices=site_indices.to(torch.int64),
            images=images.to(torch.int64),
            k_pts=k_pts,
            frac_positions=frac_positions,
            sparse=sparse,
        )


class _UnusedSKCoefficients(nn.Module):
    """Placeholder SK coefficient module for `E3SKHamiltonian`.

    `E3SKHamiltonian` does not use `SKHamiltonian.sk_coefficients`, but we keep
    the attribute to avoid allocating unused parameters.
    """

    def forward(self, z_i: torch.IntTensor, z_j: torch.IntTensor, r: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise RuntimeError("`E3SKHamiltonian` does not use `sk_coefficients`.")


class E3SKHamiltonian(SKHamiltonian):
    """E(3)-equivariant Slater-Koster-style Hamiltonian for arbitrary `l`.

    This class generalizes `SKHamiltonian` by constructing direction-dependent
    inter-site coupling blocks using e3nn tensor products:

        H_{l_in -> l_out}(u, r) = TP_l.right(Y(u), w(r))

    where `Y(u)` are spherical harmonics of the bond direction and `w(r)` are
    learnable radial weights. This avoids hard-coding σ/π/δ/... angular tables
    and naturally supports higher angular momentum shells.

    Notes:
        - This module assumes neighbor lists include both directions:
          `(i->j, image)` and `(j->i, -image)`. It will build couplings for only
          a canonical half of edges and then add the symmetric transpose to
          guarantee a real symmetric Hamiltonian (Hermitian after Bloch phases).
        - Radial weights are shared across element pairs by default; element-
          specific SK parameterizations can be added later if needed.
    """

    def __init__(
        self,
        eigensolver: str | Eigensolver = "torch_eigh",
        orbital_shells: dict[int, list[int]] | None = None,
        energy_scale: float = 1.0,
        z_max: int = 118,
        l_max: int | None = None,
        r_max: float = 5.0,
        r_points: int = 32,
        edge_hidden: int = 64,
        edge_layers: int = 2,
        activation: str = "silu",
        sh_normalization: str = "integral",
    ) -> None:
        # Import lazily so `SKHamiltonian` users don't need e3nn.
        try:
            import e3nn.nn  # noqa: F401
            import e3nn.o3 as o3  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError("`E3SKHamiltonian` requires `e3nn` to be installed.") from exc

        if orbital_shells is None or not orbital_shells:
            raise ValueError("`orbital_shells` must be a non-empty dict.")

        if l_max is None:
            l_max = max(int(l) for l_list in orbital_shells.values() for l in l_list)
        self.l_max = int(l_max)
        if self.l_max < 0:
            raise ValueError("`l_max` must be non-negative.")

        super().__init__(
            eigensolver=eigensolver,
            orbital_shells=orbital_shells,
            sk_coefficients=_UnusedSKCoefficients(),
            energy_scale=energy_scale,
            z_max=z_max,
        )

        import e3nn.nn
        import e3nn.o3 as o3

        self.r_max = float(r_max)
        self.r_points = int(r_points)
        self.edge_hidden = int(edge_hidden)
        self.edge_layers = int(edge_layers)
        self.sh_normalization = str(sh_normalization)

        if self.r_max <= 0:
            raise ValueError("`r_max` must be positive.")
        if self.r_points <= 0:
            raise ValueError("`r_points` must be a positive integer.")
        if self.edge_layers <= 0:
            raise ValueError("`edge_layers` must be a positive integer.")

        self._act = get_activation_fn(activation)()

        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=self.l_max)

        # RBF distance embedding similar to `AtomwiseNNConv`.
        self.register_buffer("r_pts", torch.linspace(0.0, self.r_max, self.r_points))

        # Build per-(l_in,l_out) tensor products and radial networks.
        # We only build for l values that appear in `orbital_shells` to keep things small.
        l_values = torch.unique(self.orbital_shell_idxs[self.orbital_shell_idxs >= 0]).to(torch.int64).tolist()
        l_values = sorted({int(l) for l in l_values})
        self.l_values = tuple(l_values)

        self.tps = nn.ModuleDict()
        self.edge_nns = nn.ModuleDict()
        self.tp_l_pairs: tuple[tuple[int, int], ...]

        tp_l_pairs: list[tuple[int, int]] = []
        for l_in in self.l_values:
            for l_out in self.l_values:
                irreps_in = o3.Irreps([(1, (int(l_in), (-1) ** int(l_in)))])
                irreps_out = o3.Irreps([(1, (int(l_out), (-1) ** int(l_out)))])
                tp = o3.FullyConnectedTensorProduct(
                    irreps_in1=irreps_in,
                    irreps_in2=self.sh_irreps,
                    irreps_out=irreps_out,
                    shared_weights=False,
                    compile_right=True,
                )
                key = f"{l_in}->{l_out}"
                self.tps[key] = tp
                tp_l_pairs.append((int(l_in), int(l_out)))

                hs = [self.r_points] + [self.edge_hidden] * (self.edge_layers - 1) + [tp.weight_numel]
                self.edge_nns[key] = e3nn.nn.FullyConnectedNet(hs, act=self._act)
        self.tp_l_pairs = tuple(tp_l_pairs)

        # Precompute per-element shell orbital starts/sizes for fast slicing.
        z_dim = self.z_max + 1
        n_shells_max = self.orbital_shell_idxs.shape[1]
        shell_start = torch.full((z_dim, n_shells_max), -1, dtype=torch.int64)
        shell_size = torch.zeros((z_dim, n_shells_max), dtype=torch.int64)

        for z in range(z_dim):
            start = 0
            for s in range(n_shells_max):
                l = int(self.orbital_shell_idxs[z, s].item())
                if l < 0:
                    continue
                shell_start[z, s] = start
                size = 2 * l + 1
                shell_size[z, s] = size
                start += size

        self.register_buffer("shell_orbital_start", shell_start)
        self.register_buffer("shell_orbital_size", shell_size)

    def _validate_orbital_shells(self, shells: dict[int, list[int]]) -> None:
        for z, l_list in shells.items():
            if z <= 0:
                raise ValueError(f"Expected positive atomic number in `orbital_shells`, got: {z}")
            for l in l_list:
                l_int = int(l)
                if l_int < 0:
                    raise ValueError(f"Expected non-negative l, got: l={l_int} for element {z}.")
                if l_int > int(self.l_max):
                    raise ValueError(f"Encountered l={l_int} > l_max={self.l_max} for element {z}.")

    def _rbf(self, r: torch.Tensor) -> torch.Tensor:
        sigma = (self.r_pts[-1] - self.r_pts[0]) / float(self.r_pts.shape[0])
        r = r.to(dtype=self.r_pts.dtype, device=self.r_pts.device)
        return torch.exp(-0.5 * ((self.r_pts.view(1, -1) - r.view(-1, 1)) / sigma) ** 2)

    def orbital_names(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> list[str]:
        names: list[str] = []
        for z in atomic_numbers.tolist():
            l_values = self.orbital_shell_idxs[int(z)]
            for l in l_values.tolist():
                if l < 0:
                    continue
                l_int = int(l)
                if l_int == 0:
                    names.append("s")
                elif l_int == 1:
                    names.extend(["p_x", "p_y", "p_z"])
                else:
                    names.extend([f"l{l_int}_{i}" for i in range(2 * l_int + 1)])
        return names

    def hamiltonian_operator_entries(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor | None = None,
        unit_cell_images: torch.IntTensor | None = None,
        assume_bidirectional_neighbors: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.IntTensor, torch.Tensor, torch.IntTensor]:
        import e3nn.o3 as o3

        n_orbitals, offsets = self._orbital_counts(atomic_numbers)

        site_idxs, local_orb_idxs = self.orbital_indices(atomic_numbers, positions, neighbors)
        z_orb = atomic_numbers[site_idxs].to(torch.int64)
        shell_idx = self.orbital_shell_map[z_orb, local_orb_idxs].to(torch.int64)
        onsite_values = self.onsite_energies[z_orb, shell_idx].to(dtype=positions.dtype, device=positions.device)
        onsite_values = onsite_values * self.energy_scale

        site_i, adj = torch.nonzero(neighbors >= 0, as_tuple=True)
        if site_i.numel() == 0:
            return (
                onsite_values,
                torch.empty((0, 2), dtype=torch.int64, device=positions.device),
                torch.empty((0,), dtype=positions.dtype, device=positions.device),
                torch.empty((0, 3), dtype=torch.int64, device=positions.device),
            )

        site_j = neighbors[site_i, adj].to(torch.int64)

        images_ij = torch.zeros((site_i.shape[0], 3), dtype=torch.int64, device=positions.device)
        if unit_cell_images is not None:
            if unit_cell is None:
                raise ValueError("`unit_cell_images` requires `unit_cell`.")
            images_ij = unit_cell_images.to(device=positions.device, dtype=torch.int64)[site_i, adj]

        # Canonicalize to one direction (fast, no `unique`): requires bidirectional neighbor lists.
        if not bool(assume_bidirectional_neighbors):
            raise ValueError("`E3SKHamiltonian` currently requires `assume_bidirectional_neighbors=True`.")

        src_all = site_i.to(torch.int64)
        dst_all = site_j.to(torch.int64)
        edge_images_all = images_ij.to(torch.int64)

        keep = src_all < dst_all

        same = src_all == dst_all
        if bool(same.any().item()):
            img = edge_images_all[same]
            nonzero = (img != 0).any(dim=1)
            if bool(nonzero.any().item()):
                img_nz = img[nonzero]
                lex_pos = (
                    (img_nz[:, 0] > 0)
                    | ((img_nz[:, 0] == 0) & (img_nz[:, 1] > 0))
                    | ((img_nz[:, 0] == 0) & (img_nz[:, 1] == 0) & (img_nz[:, 2] > 0))
                )

                same_idx = torch.nonzero(same, as_tuple=False).view(-1)
                nz_idx = same_idx[torch.nonzero(nonzero, as_tuple=False).view(-1)]
                keep[nz_idx] = lex_pos

        src = src_all[keep]
        dst = dst_all[keep]
        edge_images = edge_images_all[keep]

        if src.numel() == 0:
            return (
                onsite_values,
                torch.empty((0, 2), dtype=torch.int64, device=positions.device),
                torch.empty((0,), dtype=positions.dtype, device=positions.device),
                torch.empty((0, 3), dtype=torch.int64, device=positions.device),
            )

        disp = positions[dst] - positions[src]
        if unit_cell is not None:
            shift = edge_images.to(dtype=positions.dtype, device=positions.device) @ unit_cell.to(
                dtype=positions.dtype, device=positions.device
            )
            disp = disp + shift

        r = torch.linalg.norm(disp, dim=-1)
        if bool((r <= 0).any().item()):
            raise ValueError("Encountered zero-distance neighbor pair while building E3SK Hamiltonian.")
        u = disp / r.view(-1, 1)

        y = o3.spherical_harmonics(
            self.sh_irreps,
            u,
            normalize=True,
            normalization=self.sh_normalization,
        )
        x_r = self._rbf(r.to(self.r_pts.device))

        # Precompute all per-(l_in,l_out) operator blocks for each edge.
        ops: dict[str, torch.Tensor] = {}
        for key, tp in self.tps.items():
            w = self.edge_nns[key](x_r.to(dtype=positions.dtype, device=positions.device))
            ops[key] = tp.right(y.to(dtype=positions.dtype, device=positions.device), weight=w)

        couplings_u: list[torch.IntTensor] = []
        values_u: list[torch.Tensor] = []
        images_u: list[torch.IntTensor] = []

        n_shells_max = self.orbital_shell_idxs.shape[1]

        for e in range(int(src.shape[0])):
            i = int(src[e].item())
            j = int(dst[e].item())

            n_i = int(n_orbitals[i].item())
            n_j = int(n_orbitals[j].item())
            if n_i == 0 or n_j == 0:
                continue

            z_i = int(atomic_numbers[i].item())
            z_j = int(atomic_numbers[j].item())

            idx_i = torch.arange(n_i, device=positions.device, dtype=torch.int64) + offsets[i]
            idx_j = torch.arange(n_j, device=positions.device, dtype=torch.int64) + offsets[j]

            block = torch.zeros((n_i, n_j), dtype=positions.dtype, device=positions.device)

            # Assemble block from shell-to-shell coupling operators.
            for s_i in range(n_shells_max):
                l_in = int(self.orbital_shell_idxs[z_i, s_i].item())
                if l_in < 0:
                    continue
                start_i = int(self.shell_orbital_start[z_i, s_i].item())
                size_i = int(self.shell_orbital_size[z_i, s_i].item())
                if size_i <= 0:
                    continue
                for s_j in range(n_shells_max):
                    l_out = int(self.orbital_shell_idxs[z_j, s_j].item())
                    if l_out < 0:
                        continue
                    start_j = int(self.shell_orbital_start[z_j, s_j].item())
                    size_j = int(self.shell_orbital_size[z_j, s_j].item())
                    if size_j <= 0:
                        continue

                    op = ops[f"{l_in}->{l_out}"][e]  # [dim_in, dim_out]
                    block[start_i : start_i + size_i, start_j : start_j + size_j] = op * self.energy_scale

            couplings_u.append(torch.cartesian_prod(idx_i, idx_j))
            values_u.append(block.reshape(-1))
            img = edge_images[e].to(dtype=torch.int64, device=positions.device).view(1, 3)
            images_u.append(img.expand(n_i * n_j, 3))

        if not couplings_u:
            return (
                onsite_values,
                torch.empty((0, 2), dtype=torch.int64, device=positions.device),
                torch.empty((0,), dtype=positions.dtype, device=positions.device),
                torch.empty((0, 3), dtype=torch.int64, device=positions.device),
            )

        couplings_u_t = torch.cat(couplings_u, dim=0)
        values_u_t = torch.cat(values_u, dim=0)
        images_u_t = torch.cat(images_u, dim=0)

        couplings = torch.cat((couplings_u_t, couplings_u_t.flip(1)), dim=0)
        coupling_values = torch.cat((values_u_t, values_u_t), dim=0)
        coupling_images = torch.cat((images_u_t, -images_u_t), dim=0)

        return (
            onsite_values,
            couplings.to(torch.int64),
            coupling_values,
            coupling_images.to(torch.int64),
        )


class PODHamiltonian(E3SKHamiltonian):
    """Locally truncated (per-atom) Hamiltonian via a POD/SVD basis projection.

    This module computes a *block-diagonal* change of basis `U = ⊕_i U_i`, where each
    `U_i` is obtained from a per-site POD of the site's incident coupling blocks.
    It then returns operator entries for the projected Hamiltonian:

        H' = U^T H U

    This preserves **locality** (only mixes orbitals within each atom), and can reduce the
    orbital dimension to `bands_per_atom` per site.

    Caveats:
        - This is not equivalent to global Wannierization (which uses a global/gauge-consistent
          unitary across the crystal). It is a local heuristic.
        - The POD basis depends on the predicted couplings; eigen/SVD is differentiable but
          can be numerically unstable near degeneracies. Use `detach_basis=True` to stop
          gradients through the basis.
    """

    def __init__(
        self,
        *args,
        bands_per_atom: int | None = None,
        detach_basis: bool = True,
        pod_eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if bands_per_atom is not None and bands_per_atom <= 0:
            raise ValueError("`bands_per_atom` must be a positive integer or None.")
        if pod_eps < 0:
            raise ValueError("`pod_eps` must be non-negative.")
        self.bands_per_atom: int | None = int(bands_per_atom) if bands_per_atom is not None else None
        self.detach_basis = bool(detach_basis)
        self.pod_eps = float(pod_eps)

    def set_bands_per_atom(self, bands_per_atom: int | None) -> None:
        """Update the per-atom truncation size.

        If set to None, this module behaves like `E3SKHamiltonian`.
        """
        if bands_per_atom is not None and int(bands_per_atom) <= 0:
            raise ValueError("`bands_per_atom` must be a positive integer or None.")
        self.bands_per_atom = int(bands_per_atom) if bands_per_atom is not None else None

    def hamiltonian_operator_entries(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor | None = None,
        unit_cell_images: torch.IntTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.IntTensor, torch.Tensor, torch.IntTensor]:
        if self.bands_per_atom is None:
            return super().hamiltonian_operator_entries(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images,
                **kwargs,
            )

        import e3nn.o3 as o3

        device = positions.device
        dtype = positions.dtype

        n_sites = int(atomic_numbers.shape[0])
        max_orb = int(self.orbital_l_map.shape[1])
        k = int(self.bands_per_atom)

        # Build per-orbital onsite values and a padded per-site onsite vector `e_site`.
        site_idxs, local_orb_idxs = self.orbital_indices(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
        )
        z_orb = atomic_numbers[site_idxs].to(torch.int64)
        shell_idx = self.orbital_shell_map[z_orb, local_orb_idxs].to(torch.int64)
        onsite_full = self.onsite_energies[z_orb, shell_idx].to(dtype=dtype, device=device) * self.energy_scale

        e_site = torch.zeros((n_sites, max_orb), device=device, dtype=dtype)
        e_site.index_put_(
            (site_idxs.to(torch.int64), local_orb_idxs.to(torch.int64)),
            onsite_full,
            accumulate=False,
        )

        # Canonicalize neighbor list to one directed representative per physical bond.
        site_i, adj = torch.nonzero(neighbors >= 0, as_tuple=True)
        if site_i.numel() == 0:
            onsite_red = torch.zeros((n_sites * k,), device=device, dtype=dtype)
            return (
                onsite_red,
                torch.empty((0, 2), device=device, dtype=torch.int64),
                torch.empty((0,), device=device, dtype=dtype),
                torch.empty((0, 3), device=device, dtype=torch.int64),
            )

        site_j = neighbors[site_i, adj].to(torch.int64)
        edge_images_all = torch.zeros((site_i.shape[0], 3), dtype=torch.int64, device=device)
        if unit_cell_images is not None:
            if unit_cell is None:
                raise ValueError("`unit_cell_images` requires `unit_cell`.")
            edge_images_all = unit_cell_images.to(device=device, dtype=torch.int64)[site_i, adj]

        src_all = site_i.to(torch.int64)
        dst_all = site_j.to(torch.int64)

        keep = src_all < dst_all
        same = src_all == dst_all
        if bool(same.any().item()):
            img = edge_images_all[same]
            nonzero = (img != 0).any(dim=1)
            if bool(nonzero.any().item()):
                img_nz = img[nonzero]
                lex_pos = (
                    (img_nz[:, 0] > 0)
                    | ((img_nz[:, 0] == 0) & (img_nz[:, 1] > 0))
                    | ((img_nz[:, 0] == 0) & (img_nz[:, 1] == 0) & (img_nz[:, 2] > 0))
                )
                same_idx = torch.nonzero(same, as_tuple=False).view(-1)
                nz_idx = same_idx[torch.nonzero(nonzero, as_tuple=False).view(-1)]
                keep[nz_idx] = lex_pos

        src = src_all[keep]
        dst = dst_all[keep]
        edge_images = edge_images_all[keep]

        if src.numel() == 0:
            onsite_red = torch.zeros((n_sites * k,), device=device, dtype=dtype)
            return (
                onsite_red,
                torch.empty((0, 2), device=device, dtype=torch.int64),
                torch.empty((0,), device=device, dtype=dtype),
                torch.empty((0, 3), device=device, dtype=torch.int64),
            )

        disp = positions[dst] - positions[src]
        if unit_cell is not None:
            shift = edge_images.to(dtype=dtype, device=device) @ unit_cell.to(dtype=dtype, device=device)
            disp = disp + shift

        r = torch.linalg.norm(disp, dim=-1)
        if bool((r <= 0).any().item()):
            raise ValueError("Encountered zero-distance neighbor pair while building POD Hamiltonian.")
        u = disp / r.view(-1, 1)

        y = o3.spherical_harmonics(
            self.sh_irreps,
            u,
            normalize=True,
            normalization=self.sh_normalization,
        ).to(dtype=dtype, device=device)
        x_r = self._rbf(r)
        x_r = x_r.to(dtype=dtype, device=device)

        # Compute all tensor-product "right" operators for each l-pair once.
        ops: list[torch.Tensor] = []
        for key in self.tps.keys():
            tp = self.tps[key]
            w = self.edge_nns[key](x_r)
            ops.append(tp.right(y, weight=w))

        z_src = atomic_numbers[src].to(torch.int64)
        z_dst = atomic_numbers[dst].to(torch.int64)
        l_in_all = self.orbital_shell_idxs[z_src]  # [E, S]
        l_out_all = self.orbital_shell_idxs[z_dst]  # [E, S]
        row_start_all = self.shell_orbital_start[z_src]  # [E, S]
        col_start_all = self.shell_orbital_start[z_dst]  # [E, S]

        E = int(src.shape[0])
        H_flat = torch.zeros((E, max_orb * max_orb), device=device, dtype=dtype)

        # Assemble edge blocks in padded form using scatter-add (no Python dictionaries, no edge loops).
        n_shells_max = int(self.orbital_shell_idxs.shape[1])
        op_idx = 0
        for l_in, l_out in self.tp_l_pairs:
            op = ops[op_idx]  # [E, 2l_in+1, 2l_out+1]
            op_idx += 1

            dim_in = 2 * int(l_in) + 1
            dim_out = 2 * int(l_out) + 1

            val = op.reshape(E, dim_in * dim_out)

            ar_in = torch.arange(dim_in, device=device, dtype=torch.int64).view(1, -1)
            ar_out = torch.arange(dim_out, device=device, dtype=torch.int64).view(1, -1)

            for s_in in range(n_shells_max):
                l_in_s = l_in_all[:, s_in]
                row0 = row_start_all[:, s_in]
                valid_in = (l_in_s == int(l_in)) & (row0 >= 0)
                row0 = row0.clamp_min(0).to(torch.int64).view(-1, 1)
                row_idx = row0 + ar_in  # [E, dim_in]
                row_idx_flat = row_idx.repeat_interleave(dim_out, dim=1)  # [E, dim_in*dim_out]

                for s_out in range(n_shells_max):
                    l_out_s = l_out_all[:, s_out]
                    col0 = col_start_all[:, s_out]
                    valid = valid_in & (l_out_s == int(l_out)) & (col0 >= 0)
                    col0 = col0.clamp_min(0).to(torch.int64).view(-1, 1)
                    col_idx = col0 + ar_out  # [E, dim_out]
                    col_idx_flat = col_idx.repeat(1, dim_in)  # [E, dim_in*dim_out]

                    lin = row_idx_flat * max_orb + col_idx_flat
                    H_flat.scatter_add_(1, lin, val * valid.to(dtype=dtype).view(E, 1))

        H_edge = H_flat.view(E, max_orb, max_orb)

        # Per-site covariance accumulation: C_i += H_ij H_ij^T and C_j += H_ij^T H_ij.
        cov = torch.eye(max_orb, device=device, dtype=dtype).view(1, max_orb, max_orb).expand(n_sites, -1, -1)
        cov = cov * self.pod_eps

        cov_src = H_edge @ H_edge.mT
        cov_dst = H_edge.mT @ H_edge
        cov.index_add_(0, src, cov_src)
        cov.index_add_(0, dst, cov_dst)

        # POD basis: top-k eigenvectors of C_i (batched).
        _evals, evecs = torch.linalg.eigh(cov)
        # Order columns by descending importance so band index 0 is most important.
        U = evecs[:, :, -k:].flip(-1)  # [N, max_orb, k]
        if self.detach_basis:
            U = U.detach()

        # Project onsite and include on-site off-diagonal terms.
        tmp = e_site.unsqueeze(-1) * U  # [N, max_orb, k]
        Hii = U.mT @ tmp  # [N, k, k]

        onsite_diag = torch.diagonal(Hii, dim1=-2, dim2=-1)  # [N, k]
        onsite_red = onsite_diag.reshape(n_sites * k)

        # Onsite off-diagonal couplings (upper triangle).
        if k > 1:
            tri = torch.triu_indices(k, k, offset=1, device=device)
            P = int(tri.shape[1])
            base = (torch.arange(n_sites, device=device, dtype=torch.int64) * k).view(-1, 1)
            a = base + tri[0].view(1, -1)
            b = base + tri[1].view(1, -1)
            onsite_pairs = torch.stack((a, b), dim=-1).reshape(n_sites * P, 2)
            onsite_vals = Hii[:, tri[0], tri[1]].reshape(n_sites * P).to(dtype=dtype)
            onsite_img = torch.zeros((n_sites * P, 3), device=device, dtype=torch.int64)
        else:
            onsite_pairs = torch.empty((0, 2), device=device, dtype=torch.int64)
            onsite_vals = torch.empty((0,), device=device, dtype=dtype)
            onsite_img = torch.empty((0, 3), device=device, dtype=torch.int64)

        # Project inter-site edge blocks.
        U_src = U[src]  # [E, max_orb, k]
        U_dst = U[dst]  # [E, max_orb, k]
        Htmp = H_edge @ U_dst  # [E, max_orb, k]
        Hij = (U_src.mT @ Htmp).to(dtype=dtype)  # [E, k, k]

        ii = (src * k).view(-1, 1) + torch.arange(k, device=device, dtype=torch.int64).view(1, -1)  # [E, k]
        jj = (dst * k).view(-1, 1) + torch.arange(k, device=device, dtype=torch.int64).view(1, -1)  # [E, k]

        a = ii[:, :, None].expand(E, k, k)
        b = jj[:, None, :].expand(E, k, k)
        edge_pairs = torch.stack((a, b), dim=-1).reshape(E * k * k, 2)
        edge_vals = Hij.reshape(E * k * k)
        edge_img = edge_images.to(torch.int64).view(E, 1, 1, 3).expand(E, k, k, 3).reshape(E * k * k, 3)

        # Concatenate couplings and add transpose counterparts.
        pairs_u = torch.cat((onsite_pairs, edge_pairs), dim=0)
        vals_u = torch.cat((onsite_vals, edge_vals), dim=0)
        img_u = torch.cat((onsite_img, edge_img), dim=0)

        pairs = torch.cat((pairs_u, pairs_u.flip(1)), dim=0)
        vals = torch.cat((vals_u, vals_u), dim=0)
        imgs = torch.cat((img_u, -img_u), dim=0)

        return (
            onsite_red.to(dtype=dtype),
            pairs.to(torch.int64),
            vals.to(dtype=dtype),
            imgs.to(torch.int64),
        )

    def hamiltonian_operator(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        unit_cell: torch.Tensor | None = None,
        unit_cell_images: torch.IntTensor | None = None,
        **kwargs,
    ) -> TorchOperator:
        if self.bands_per_atom is None:
            return super().hamiltonian_operator(
                atomic_numbers=atomic_numbers,
                positions=positions,
                neighbors=neighbors,
                unit_cell=unit_cell,
                unit_cell_images=unit_cell_images,
                **kwargs,
            )

        onsite_values, couplings, coupling_values, images = self.hamiltonian_operator_entries(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbors=neighbors,
            unit_cell=unit_cell,
            unit_cell_images=unit_cell_images,
            **kwargs,
        )

        n_sites = int(atomic_numbers.shape[0])
        k = int(self.bands_per_atom)
        site_indices = torch.arange(n_sites, device=positions.device, dtype=torch.int64).repeat_interleave(k)
        orbital_indices = torch.arange(k, device=positions.device, dtype=torch.int64).repeat(n_sites)

        return TorchOperator(
            onsite_values=onsite_values,
            couplings=couplings.to(torch.int64),
            coupling_values=coupling_values,
            site_indices=site_indices.to(torch.int64),
            orbital_indices=orbital_indices.to(torch.int64),
            images=images.to(torch.int64),
        )

    def orbital_indices(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> tuple[torch.IntTensor, torch.IntTensor]:
        if self.bands_per_atom is None:
            return super().orbital_indices(atomic_numbers, positions, neighbors, **kwargs)

        n_sites = int(atomic_numbers.shape[0])
        k = int(self.bands_per_atom)
        device = atomic_numbers.device

        site_indices = torch.arange(n_sites, device=device, dtype=torch.int64).repeat_interleave(k)
        orbital_indices = torch.arange(k, device=device, dtype=torch.int64).repeat(n_sites)
        return site_indices, orbital_indices

    def orbital_names(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> list[str]:
        if self.bands_per_atom is None:
            return super().orbital_names(atomic_numbers, positions, neighbors, **kwargs)

        n_sites = int(atomic_numbers.shape[0])
        k = int(self.bands_per_atom)
        return [f"pod{n}" for _site in range(n_sites) for n in range(k)]

    def orbital_quantum_numbers(
        self,
        atomic_numbers: torch.IntTensor,
        positions: torch.Tensor,
        neighbors: torch.IntTensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.bands_per_atom is None:
            return super().orbital_quantum_numbers(atomic_numbers, positions, neighbors, **kwargs)

        n_sites = int(atomic_numbers.shape[0])
        k = int(self.bands_per_atom)
        q = torch.arange(k, device=atomic_numbers.device, dtype=torch.int64).repeat(n_sites)
        return q.view(-1, 1)
