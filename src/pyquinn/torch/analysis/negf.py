import torch
from pyquinn.torch.types import TorchAtomGraph
from pyquinn.torch.models import TorchBandStructureModel


def greens_function(
        energies: torch.Tensor, 
        hamiltonian_matrix: torch.Tensor,
        overlap_matrix: torch.Tensor | None = None,
        eta: float = 1e-5) -> torch.Tensor:
    """Computes the retarded Green's function of a Hamiltonian represented 
    in a (possibly non-orthogonal) basis.

    Args:
        energies (torch.Tensor): The energies at which the Green's function
            should be evaluated (shape: [E])
        hamiltonian_matrix (torch.Tensor): The Hamiltonian matrix of 
            the system (shape: [..., N, N]).
        overlap_matrix (torch.Tensor | None, optional): The overlap matrix
            of the Hamiltonian basis if the basis is non-orthogonal 
            (shape: [..., N, N]). Defaults to None.
        eta (float, optional): Broadening parameter of the Green's function. 
            Increasing this parameter makes the Green's function 
            well-conditioned at the cost of broadening the energy 
            spectrum. Defaults to 1e-5.

    Returns:
        torch.Tensor: The Green's function of the Hamiltonian
            (shape: [..., E, N, N]).
    """

    E = energies.view(-1, 1, 1)
    H = hamiltonian_matrix.unsqueeze(-3)
    I = torch.eye(hamiltonian_matrix.shape[-1])
    S = I if overlap_matrix is None else overlap_matrix

    A_ret = (E*S + 1.j*eta*I) - H
    G_ret = torch.linalg.inv(A_ret)

    return G_ret


def perturbed_greens_function(
        energies: torch.Tensor,
        hamiltonian_matrix: torch.Tensor,
        self_energies: torch.Tensor,
        overlap_matrix: torch.Tensor | None = None,
        eta: float = 1e-5) -> torch.Tensor:
    """Computes the retarded Green's function of a Hamiltonian represented in a 
    (possibly non-orthogonal) basis with self-energy perturbations.

    Args:
        energies (torch.Tensor): The energies at which the Green's function 
            should be evaluated (shape: [E]).
        hamiltonian_matrix (torch.Tensor): The Hamiltonian matrix of 
            the system (shape: [..., N, N]).
        self_energies (torch.Tensor): A stack of `M` self-energy perturbations
            to the Hamiltonian (shape: [M, N, N]).
        overlap_matrix (torch.Tensor | None, optional): The overlap matrix
            of the Hamiltonian basis if the basis is non-orthogonal 
            (shape: [..., N, N]). Defaults to None.
        eta (float, optional): Broadening parameter of the Green's function. 
            Increasing this parameter makes the Green's function 
            well-conditioned at the cost of broadening the energy 
            spectrum. Defaults to 1e-5.

    Returns:
        torch.Tensor: The perturbed Green's function of the Hamiltonian
            (shape: [..., E, N, N]).
    """

    E = energies.view(-1, 1, 1)
    H_site = hamiltonian_matrix.unsqueeze(-3)
    I = torch.eye(hamiltonian_matrix.shape[-1])
    S = I if overlap_matrix is None else overlap_matrix

    A_ret = (E*S + 1.j*eta*I) - \
        (H_site + self_energies.sum(0))

    G_ret = torch.linalg.inv(A_ret)

    return G_ret


def sancho_rubio(
        energies: torch.Tensor,
        coupling_matrix: torch.Tensor,
        hamiltonian_matrix: torch.Tensor,
        overlap_matrix: torch.Tensor | None = None,
        eta: float = 1e-5,
        max_iter: int = 1024,
        tol: float = 1e-6) -> torch.Tensor:
    """Applies the Sancho-Rubio method that numerically solves 
    for the surface Green's function of a semi-infinite boundary.

    .. warning::
            This method does not currently support backward passes,
            due to potential numerical instabilities. If the method
            does not converge, try increasing the `eta` parameter.

    Args:
        energies (torch.Tensor): The energies at which the Green's function 
            should be evaluated (shape: [E]).
        coupling_matrix (torch.Tensor): The coupling matrix between 
            the Hamiltonian cell and the cell's nearest neighbor along the
            semi-infinite direction (shape: [..., N, N]).
        hamiltonian_matrix (torch.Tensor): The Hamiltonian matrix of 
            a unit cell of system along the semi-infinite direction 
            (shape: [..., N, N]).
        overlap_matrix (torch.Tensor | None, optional): The overlap matrix
            of the Hamiltonian basis if the basis is non-orthogonal 
            (shape: [..., N, N]). Defaults to None.
        eta (float, optional): Broadening parameter of the Green's function. 
            Increasing this parameter makes the method converge faster
            at the cost of broadening the energy spectrum. Defaults to 1e-5.
        max_iter (int, optional): Maximum number of solver iterations. 
            Defaults to 1024.
        tol (float, optional): Error tolerance of solver. Defaults to 1e-6.

    Returns:
        torch.Tensor: The surface Green's function of the semi-infinite
            Hamiltonian (shape: [..., E, N, N])
    """
    
    E = energies.view(-1, 1, 1)
    H = hamiltonian_matrix.unsqueeze(-3)
    I = torch.eye(hamiltonian_matrix.shape[-1])
    S = I if overlap_matrix is None else overlap_matrix
    V = coupling_matrix.unsqueeze(-3)

    with torch.no_grad():
        Z = (E*S + 1.j*eta*I)        
        eps_surf = H
        eps = H
        alpha = V
        beta = V.mH

        for _ in range(max_iter):

            g = torch.linalg.inv(Z - eps)
            alpha_new = alpha @ g @ alpha
            beta_new = beta @ g @ beta

            a_g_b = (alpha @ g @ beta)
            b_g_a = (beta @ g @ alpha)
            eps_new = eps + a_g_b + b_g_a
            eps_surf_new = eps_surf + a_g_b
            
            eps_err = (eps_new - eps).abs().square().mean(
                axis=(-2, -1))
            eps_surf_err = (eps_surf_new - eps_surf).abs().square().mean(
                axis=(-2, -1))
            
            max_err = torch.maximum(eps_err, eps_surf_err).max().sqrt()

            if max_err < tol:
                break

            alpha = alpha_new
            beta = beta_new
            eps = eps_new
            eps_surf = eps_surf_new

        g_surf = torch.linalg.pinv(Z - eps_surf)

    return g_surf


class TorchNEGF:

    def _validate_boundary_atomgraph(
            self,
            graph: TorchAtomGraph,
            orientation: torch.IntTensor,
            index: int | None = None):

        if torch.abs(orientation).sum().item() > 1:
            raise ValueError(f'Invalid orientation vector: {orientation}')

        if torch.any(torch.abs(orientation) > 0):

            neighbors_idx = (graph.neighbors >= 0)
            neighbor_images = graph.unit_cell_images[neighbors_idx]
            orientation_idx = torch.nonzero(torch.abs(orientation)).item()
            max_images = torch.abs(neighbor_images).max(dim=0)[0]

            max_oriented_images = \
                max_images[orientation_idx].squeeze().item()

            if max_oriented_images > 1:
                raise ValueError(
                    f'Boundary graph unit cell at index {index} exhibits '
                    'second-nearest cell neighbor coupling. (Got cell image '
                    'interaction radius of {max_oriented_images} for '
                    'orientation {orientation}. \nConsider expanding the '
                    'boundary graph to a supercell along the orientation '
                    'direction or decreasing the cutoff radius.')

    def _resolve_k_pts(
                self,
                k_pts: torch.Tensor | None, 
                template: torch.Tensor) -> torch.Tensor:

        if k_pts is None:
            k_pts = torch.zeros(
                (1, 3),
                dtype=template.dtype,
                device=template.device,
            )

        return k_pts

    def __init__(
            self,
            model: TorchBandStructureModel,
            site: TorchAtomGraph | None,
            boundaries: list[TorchAtomGraph] | None,
            orientations: list[torch.IntTensor] | None = None):
        """Creates a Non-Equilibrium Green's Function (NEGF) calculation. 
        This calculation consists of a system of interest (the `site`) 
        connected to other boundary systems. The boundary systems may be
        non-periodic, or even semi-infinite with a given orientation.

        The NEGF object provides lots of helpful methods for modeling
        the quantum tranport of electrons through the site. The
        calculation uses a pre-trained band structure model to compute 
        system Hamiltonians and other needed quantities.

        In all NEGF calculations, it is assumed that each boundary is
        coupled directly to the site and not to any other boundary.

        Args:
            model (TorchBandStructureModel): The model used in the calculation.
            site (TorchAtomGraph | None): The site of the NEGF calculation.
            boundaries (list[TorchAtomGraph] | None): A list of boundary 
                systems, if any.
            orientations (list[torch.IntTensor] | None, optional): A list of
                orientations of each boundary. Each orientation should be of 
                the form `[i, j, k]`, where i,j,k are -1, 0, or 1. Values of
                1 or -1 indicate the direction of a semi-infinite boundary 
                TorchAtomGraph with respect to the corresponding `unit_cell` 
                vector of each boundary. Values of 0 indicate either 
                periodicity or non-periodicity depending on the values 
                of the `periodic` attribute of the TorchAtomGraph. 
                Defaults to None.
        """

        if boundaries is None:
            boundaries = []

        if orientations is None:
            orientations = [
                torch.zeros(
                    3, 
                    dtype=torch.int64,
                    device=graph.positions.device)
                for graph in boundaries
            ]

        self.model = model
        self.site = site
        self.boundaries = boundaries
        self.orientations = orientations

        # validate boundary graphs and orientations
        for i, (boundary, orientation) in \
                enumerate(zip(self.boundaries, self.orientations)):
            self._validate_boundary_atomgraph(
                boundary,
                orientation,
                index=i
            )

    def boundary_greens_functions(
            self,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            eta: float = 1e-5,
            **solver_kwargs) -> list[torch.Tensor]:
        """Evaluates the Green's function of each boundary 

        Args:
            energies (torch.Tensor): The energies at which the Green's function 
                should be evaluated (shape: [E]).
            k_pts (torch.Tensor | None, optional): Reciprocal space 
                :math:`\\mathbf{k}` vectors used to sample crystal momentum
                degrees of freedom parallel to the boundary (shape: [M, 3]). 
                Defaults to None.
            eta (float, optional): Broadening parameter of the Green's 
                function. Increasing this parameter may make any surface 
                Green's function methods converge faster for semi-infinite
                boundaries at the cost of broadening the energy spectrum. 
                Defaults to 1e-5.

        Returns:
            list[torch.Tensor]: A list of boundary Green's functions, having
            shape [M, E, N, N], where `N` is the dimension of each 
            boundary Hamiltonian.
        """

        k_pts = self._resolve_k_pts(k_pts, energies)

        g_surfaces = []

        for boundary, orientation in \
                zip(self.boundaries, self.orientations):

            if (orientation != 0).any():

                # construct supercell along orientation for 
                # extracting coupling matrix
                sc_size = torch.ones_like(orientation) + \
                    2*torch.abs(orientation)
                sc_size[orientation < 0] *= -1
                sc = boundary.supercell(sc_size)
                N = sc_size.abs().prod().item()
                
                assert N == 3
                
                # extract coupling matrix elements from supercell
                H_size = self.model.orbital_indices(boundary)[0].shape[0]
                H = self.model.hamiltonian_matrix(sc, k_pts=k_pts)
                H_cell_idxs = torch.arange(
                    H_size, 2*H_size, device=orientation.device)
                H_outer_idxs = H_cell_idxs - H_size

                V = H[:, H_cell_idxs][:, :, H_outer_idxs]
                H0 = H[:, H_cell_idxs][:, :, H_cell_idxs]
                
                # compute surface Greens function with Sancho Rubio method
                g_surface = sancho_rubio(
                    energies=energies,
                    coupling_matrix=V,
                    hamiltonian_matrix=H0,
                    eta=eta,
                    **solver_kwargs
                )
            
            else:

                # if surface is not semi-infinite, use regular Greens function
                H0 = self.model.hamiltonian_matrix(boundary, k_pts=k_pts)
                g_surface = greens_function(
                    energies=energies,
                    hamiltonian_matrix=H0,
                    eta=eta
                )
            
            g_surfaces.append(g_surface)

        return g_surfaces

    def system_graph(
            self, 
            unit_cell: torch.Tensor | None = None, 
            periodic: torch.BoolTensor | None = None) -> TorchAtomGraph:
        """Builds a TorchAtomGraph that includes the site coupled to the 
        unit cell of each boundary. 

        Args:
            unit_cell (torch.Tensor | None, optional): The unit cell of 
                the returned graph. If None, the system is assigned a 
                non-periodic unit cell. Defaults to None.
            periodic (torch.BoolTensor | None, optional): The periodicity of
                the returned graph. If None, the system is 
                treated as non-periodic. Defaults to None.

        Returns:
            TorchAtomGraph: The system TorchAtomGraph
        """

        graphs = [] if self.site is None else [self.site]
        graphs += self.boundaries

        if len(graphs) == 0:
            raise ValueError(
                'The system must consist of least one AtomGraph object.')

        atomic_numbers = torch.concatenate([
            g.atomic_numbers for g in graphs])
        positions = torch.concatenate([
            g.positions for g in graphs])
        masses = torch.concatenate([
            g.masses for g in graphs])
        features = torch.concatenate([
            g.features for g in graphs])
        outputs = torch.concatenate([
            g.features for g in graphs])
        graph_outputs = torch.concatenate([
            g.features for g in graphs])
        cutoffs = torch.concatenate([
            g.cutoffs for g in graphs
        ])

        unit_cell = torch.zeros_like(graphs[0].unit_cell)
        periodic = torch.tensor(
            [False, False, False], device=graphs[0].periodic.device)

        system_graph = TorchAtomGraph(
            atomic_numbers=atomic_numbers,
            positions=positions,
            masses=masses,
            unit_cell=unit_cell,
            periodic=periodic,
            features=features,
            outputs=outputs,
            graph_outputs=graph_outputs,
            cutoffs=cutoffs,
            compute_neighbors=True)

        return system_graph

    def site_self_energies(
            self,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            eta: float = 1e-5,
            boundary_gfs: list[torch.Tensor] | None = None,
            **solver_kwargs) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns a the site Hamiltonian and a tensor of self-energy 
        perturbations to the site Hamiltonian based on the computed 
        surface Green's functions of each boundary.

        Args:
            energies (torch.Tensor): The energies at which the Green's function 
                should be evaluated (shape: [E]).
            k_pts (torch.Tensor | None, optional): Reciprocal space 
                :math:`\\mathbf{k}` vectors used to sample crystal momentum
                degrees of freedom parallel to the boundary (shape: [M, 3]). 
                Defaults to None.
            eta (float, optional): Broadening parameter of the Green's 
                function. Increasing this parameter may make any surface 
                Green's function methods converge faster for semi-infinite
                boundaries at the cost of broadening the energy spectrum. 
                Defaults to 1e-5.
            boundary_gfs (list[torch.Tensor] | None, optional): Pre-computed 
                surface Green's functions returned from a call to 
                :meth:`NEGF.surface_greens_functions`. Defaults to None.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: A tuple containing the 
            following:

            - site_hamitonian (torch.Tensor): 
                The site Hamiltonian matrix (shape: [M, N, N])
            - self_energies (torch.Tensor):
                The self-energy perturbations (shape: 
                [N_boundaries, E, M, N, N])

        """

        k_pts = self._resolve_k_pts(k_pts, energies)

        if self.site is None:
            raise ValueError(
                'The site AtomGraph must be defined '
                'to compute site self-energies.')

        H_sizes = [
            self.model.orbital_indices(g)[0].shape[0]
            for g in self.boundaries
        ]

        # determine indices of components of the system Hamiltonian
        H_site_size = self.model.orbital_indices(self.site)[0].shape[0]
        H_boundary_idxs = []
        offset = H_site_size
        for size in H_sizes:
            H_boundary_idxs.append(torch.arange(
                offset, offset + size,
                device=energies.device))
            offset += size

        # compute system graph and Hamiltonian matrix
        system = self.system_graph()
        H_system = self.model.hamiltonian_matrix(system, k_pts=k_pts)
        H_site = H_system[:, :H_site_size, :H_site_size]
        
        # compute boundary Greens functions (if not supplied):
        if boundary_gfs is None:
            boundary_gfs = self.boundary_greens_functions(
                energies=energies,
                k_pts=k_pts,
                eta=eta,
                **solver_kwargs
            )
        
        if len(boundary_gfs) != len(self.boundaries):
            raise ValueError(
                f'Expected a list of {len(self.boundaries)} '
                f'boundary Greens functions. (got list of '
                f'size: {len(boundary_gfs)}')

        # extract device coupling and compute self-energy:
        self_energies_list = []
        for idxs, gf in \
                zip(H_boundary_idxs, boundary_gfs):
            V = H_system[:, :H_site_size, idxs]
            self_energy = V @ gf @ V.mH
            self_energies_list.append(self_energy)
        
        self_energies = torch.stack(self_energies_list)

        return H_site, self_energies

    def site_greens_function(
            self,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            eta: float = 1e-5,
            boundary_gfs: list[torch.Tensor] | None = None,
            **solver_kwargs) -> torch.Tensor:
        """Returns the site Green's function with perturbations due to 
        boundary coupling.

        Args:
            energies (torch.Tensor): The energies at which the Green's function 
                should be evaluated (shape: [E]).
            k_pts (torch.Tensor | None, optional): Reciprocal space 
                :math:`\\mathbf{k}` vectors used to sample crystal momentum
                degrees of freedom parallel to the boundary (shape: [M, 3]). 
                Defaults to None.
            eta (float, optional): Broadening parameter of the Green's 
                function. Increasing this parameter may make any surface 
                Green's function methods converge faster for semi-infinite
                boundaries at the cost of broadening the energy spectrum. 
                Defaults to 1e-5.
            boundary_gfs (list[torch.Tensor] | None, optional): Pre-computed 
                surface Green's functions returned from a call to 
                :meth:`NEGF.surface_greens_functions`. Defaults to None.

        Returns:
            torch.Tensor: The site Green's function (shape: [M, E, N, N]).
        """

        k_pts = self._resolve_k_pts(k_pts, energies)

        H_site, self_energies = self.site_self_energies(
            energies=energies,
            k_pts=k_pts,
            eta=eta,
            boundary_gfs=boundary_gfs,
            **solver_kwargs)

        site_gf = perturbed_greens_function(
            energies=energies,
            hamiltonian_matrix=H_site,
            self_energies=self_energies,
            eta=eta,
        )

        return site_gf

    def site_ldos(
            self,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            eta: float = 1e-5,
            k_resolved: bool = False,
            site_gf: torch.Tensor | None = None,
            **solver_kwargs) -> torch.Tensor:
        """Evaluates the local density of states (LDOS) of the site.

        Args:
            energies (torch.Tensor): The energies at which the density of 
                states should be evaluated (shape: [E]).
            k_pts (torch.Tensor | None, optional): Reciprocal space 
                :math:`\\mathbf{k}` vectors used to sample crystal momentum
                degrees of freedom parallel to the boundary (shape: [M, 3]). 
                Defaults to None.
            eta (float, optional): Broadening parameter of the Green's 
                function.
            k_resolved (bool, optional): if True, returns the LDOS without 
                integrating over the :math:`\\mathbf{k}` degrees of freedom.
                Defaults to False.
            site_gf (torch.Tensor | None, optional): A precomputed
                site Green's function returned from a call to
                :meth:`NEGF.site_greens_function`. Defaults to None.

        Returns:
            torch.Tensor: The evaluated LDOS 
                (shape: [E, N], or [M, E, N] if `k_resolved`).
        """

        k_pts = self._resolve_k_pts(k_pts, energies)

        if site_gf is None:
            site_gf = self.site_greens_function(
                energies=energies,
                k_pts=k_pts,
                eta=eta,
                **solver_kwargs)

        ldos = (-1/torch.pi) * torch.diagonal(
            site_gf.imag, dim1=-2, dim2=-1)

        if not k_resolved:
            ldos = ldos.mean(0)

        return ldos

    def site_dos(
            self,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            eta: float = 1e-5,
            k_resolved: bool = False,
            site_gf: torch.Tensor | None = None,
            **solver_kwargs):
        """"Evaluates the density of states (DOS) of the site.

        Args:
            energies (torch.Tensor): The energies at which the density of 
                states should be evaluated (shape: [E]).
            k_pts (torch.Tensor | None, optional): Reciprocal space 
                :math:`\\mathbf{k}` vectors used to sample crystal momentum
                degrees of freedom parallel to the boundary (shape: [M, 3]). 
                Defaults to None.
            eta (float, optional): Broadening parameter of the Green's 
                function.
            k_resolved (bool, optional): if True, returns the LDOS without 
                integrating over the :math:`\\mathbf{k}` degrees of freedom.
                Defaults to False.
            site_gf (torch.Tensor | None, optional): A precomputed
                site Green's function returned from a call to
                :meth:`NEGF.site_greens_function`. Defaults to None.

        Returns:
            torch.Tensor: The evaluated LDOS 
                (shape: [E], or [M, E] if `k_resolved`).
        """
        
        k_pts = self._resolve_k_pts(k_pts, energies)

        if site_gf is None:
            site_gf = self.site_greens_function(
                energies=energies,
                k_pts=k_pts,
                eta=eta,
                **solver_kwargs
            )

        dos = (-1/torch.pi) * torch.diagonal(
            site_gf.imag, dim1=-2, dim2=-1).sum(-1)

        if not k_resolved:
            dos = dos.mean(0)

        return dos

    def _lesser_site_greens_function_data(
            self,
            energies: torch.Tensor,
            k_pts: torch.Tensor,
            eta: float,
            chemical_potentials: list[torch.Tensor] |
            list[float] | None = None,
            kt: torch.Tensor | float = 1e-5,
            boundary_gfs: list[torch.Tensor] | None = None,
            **solver_kwargs) -> list[torch.Tensor]:

        # compute self-energies
        H_site, self_energies = self.site_self_energies(
            energies=energies,
            k_pts=k_pts,
            eta=eta,
            boundary_gfs=boundary_gfs,
            **solver_kwargs)

        # resolve chemical potentials and occupation probabilities
        if chemical_potentials is None:
            chemical_potentials = [0.0]*len(self_energies)

        if len(chemical_potentials) != len(self.boundaries):
            raise ValueError(
                'Expected len(chemical_potentials) == len(self.boundaries);'
                f' got {len(chemical_potentials)} != {len(self.boundaries)}.'
            )

        occupation_probs = torch.stack([
            1.0 / (1.0 + torch.exp((energies - mu)/kt))
            for mu in chemical_potentials
        ])
        occupations = occupation_probs.view(
            occupation_probs.shape + (1, 1))

        # compute occupation-weighted broadening matrices

        gamma_weighted = 1.j*(
            (self_energies - self_energies.mH) * occupations
        ).sum(-4)

        print(gamma_weighted.shape)

        # compute retarted greens function
        g_ret = perturbed_greens_function(
            energies=energies,
            hamiltonian_matrix=H_site,
            self_energies=self_energies,
            eta=eta,
        )

        # assemble lesser greens function
        lesser_gf = g_ret @ gamma_weighted @ g_ret.mH

        return H_site, self_energies, lesser_gf

    def lesser_site_greens_function(
            self,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            eta: float = 1e-5,
            chemical_potentials: list[torch.Tensor] |
            list[float] | None = None,
            kt: torch.Tensor | float = 1e-5,
            boundary_gfs: list[torch.Tensor] | None = None,
            **solver_kwargs) -> torch.Tensor:
        """Computes the lesser Green's function of the site.

        Args:
            energies (torch.Tensor): The energies at which the density of 
                states should be evaluated (shape: [E]).
            k_pts (torch.Tensor | None, optional): Reciprocal space 
                :math:`\\mathbf{k}` vectors used to sample crystal momentum
                degrees of freedom parallel to the boundary (shape: [M, 3]). 
                Defaults to None.
            eta (float, optional): Broadening parameter of the Green's 
                function.
            chemical_potentials (list[torch.Tensor] | 
                list[float] | None, optional): 
                List of chemical potentials :math:`\\mu` of each boundary. 
                Defaults to None.
            kt (torch.Tensor | float, optional): 
                The Boltzmann constant times the temperature of the device
                (:math:`k_BT`). Defaults to 1e-5.
            boundary_gfs (list[torch.Tensor] | None, optional): Pre-computed 
                surface Green's functions returned from a call to 
                :meth:`NEGF.surface_greens_functions`. Defaults to None.

        Returns:
            torch.Tensor: The lesser site Green's function
                (shape: [M, E, N, N])
        """
        
        k_pts = self._resolve_k_pts(k_pts, energies)

        _, _, lesser_gf = self._lesser_site_greens_function_data(
            energies=energies,
            k_pts=k_pts,
            chemical_potentials=chemical_potentials,
            eta=eta,
            kt=kt,
            boundary_gfs=boundary_gfs,
            **solver_kwargs)

        return lesser_gf
            
    def site_charge_density(
            self,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            eta: float = 1e-5,
            chemical_potentials: list[torch.Tensor] |
            list[float] | None = None,
            kt: float = 1e-5,
            lesser_gf: torch.Tensor | None = None,
            **solver_kwargs) -> torch.Tensor:
        """Estimates the charge density matrix of the site by integrating over
        sampled energy values.

        For best results, energies should be sampled uniformly from the lowest 
        occupied energy level to a few multiples of `kt` above the highest
        `chemical_potential` value.

        Args:
            energies (torch.Tensor): The energies at which the density of 
                states should be evaluated and integrated over (shape: [E]).
            k_pts (torch.Tensor | None, optional): Reciprocal space 
                :math:`\\mathbf{k}` vectors used to sample crystal momentum
                degrees of freedom parallel to the boundary (shape: [M, 3]). 
                Defaults to None.
            eta (float, optional): Broadening parameter of the Green's 
                function.
            chemical_potentials (list[torch.Tensor] | 
                list[float] | None, optional): 
                List of chemical potentials :math:`\\mu` of each boundary. 
                Defaults to None.
            kt (torch.Tensor | float, optional): 
                Boltzman's constant times the temperature of the device. 
                Defaults to 1e-5.
            lesser_gf (torch.Tensor | None, optional): Pre-computed 
                lesser Green's function returned from a call to 
                :meth:`NEGF.lesser_greens_function`. Defaults to None.

        Returns:
            torch.Tensor: The charge density of the site (shape: [N, N])
        """

        if lesser_gf is None:
            lesser_gf = self.lesser_site_greens_function(
                energies=energies,
                k_pts=k_pts,
                eta=eta,
                chemical_potentials=chemical_potentials,
                kt=kt,
                **solver_kwargs
            )

        sorted_energies, sorted_idx = torch.sort(energies)
        sorted_lesser_gf = lesser_gf[..., sorted_idx, :, :]

        rho = -1.j * torch.trapezoid(
            sorted_lesser_gf, 
            sorted_energies,
            dim=-3
        ) / (2*torch.pi)

        return rho

    def site_bond_current(
            self,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None,
            eta: float = 1e-5,
            chemical_potentials: list[torch.Tensor] |
            list[float] | None = None,
            kt: torch.Tensor | float = 1e-5,
            boundary_gfs: list[torch.Tensor] | None = None,
            **solver_kwargs) -> torch.Tensor:
        """Evaluates the bond current matrix of the site
        as a function of energy.

        Args:
            energies (torch.Tensor): The energies at which the bond current
                should be evaluated (shape: [E]).
            k_pts (torch.Tensor | None, optional): Reciprocal space 
                :math:`\\mathbf{k}` vectors used to sample crystal momentum
                degrees of freedom parallel to the boundary (shape: [M, 3]). 
                Defaults to None.
            eta (float, optional): Broadening parameter of the Green's 
                function.
            chemical_potentials (list[torch.Tensor] | 
                list[float] | None, optional): 
                List of chemical potentials :math:`\\mu` of each boundary. 
                Defaults to None.
            kt (torch.Tensor | float, optional): 
                Boltzman's constant times the temperature of the device. 
                Defaults to 1e-5.
            boundary_gfs (list[torch.Tensor] | None, optional): Pre-computed 
                surface Green's functions returned from a call to 
                :meth:`NEGF.surface_greens_functions`. Defaults to None.

        Returns:
            torch.Tensor: The bond current matrix of the site 
                (shape: [M, E, N, N])
        """

        k_pts = self._resolve_k_pts(k_pts, energies)

        H_site, _, lesser_gf = \
            self._lesser_site_greens_function_data(
                energies=energies,
                k_pts=k_pts,
                eta=eta,
                chemical_potentials=chemical_potentials,
                kt=kt,
                boundary_gfs=boundary_gfs,
                **solver_kwargs)

        I_map = 2*torch.pi * \
            (H_site * lesser_gf.mT).imag

        return I_map

    def transmission(
            self,
            in_idx: torch.IntTensor | int,
            out_idx: torch.IntTensor | int,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            eta: float = 1e-5,
            k_resolved: bool = False,
            boundary_gfs: list[torch.Tensor] | None = None,
            **solver_kwargs) -> torch.Tensor:
        """Computes the transmission coefficient of the site as
        a function of energy.

        The transmission is evaluated between the index of an 
        input boundary and an output boundary for the
        given energy values.

        Args:
            in_idx (torch.IntTensor | int): The input boundary
                (shape: [B]).
            out_idx (torch.IntTensor | int): The output boundary 
                (shape: [B]).
            energies (torch.Tensor): The energies at which the bond current
                should be evaluated (shape: [E]).
            k_pts (torch.Tensor | None, optional): Reciprocal space 
                :math:`\\mathbf{k}` vectors used to sample crystal momentum
                degrees of freedom parallel to the boundary (shape: [M, 3]). 
                Defaults to None.
            eta (float, optional): Broadening parameter of the Green's 
                function.
            k_resolved (bool, optional): If True, the transmission matrix is
                returned without integrating over the sampled
                :math:`\\mathbf{k}` vectors. Defaults to False.
            boundary_gfs (list[torch.Tensor] | None, optional): Pre-computed 
                surface Green's functions returned from a call to 
                :meth:`NEGF.surface_greens_functions`. Defaults to None.

        Returns:
            torch.Tensor: The site transmission matrix 
                (shape: [B, E], or [B, M, E] if `k_resolved` is `True`)
        """

        k_pts = self._resolve_k_pts(k_pts, energies)

        if isinstance(in_idx, int) ^ isinstance(out_idx, int):
            raise ValueError(
                f'Expected in_idx and out_idx to have same type '
                f'(int or torch.IntTensor) (got: {type(in_idx)} and '
                f'{type(out_idx)}).'
            )

        if not isinstance(in_idx, int) and in_idx.ndim != 1:
            raise ValueError(
                'Expected in_idx to have ndim == 1; '
                f'(got ndim == {in_idx.ndim})')
        if not isinstance(out_idx, int) and out_idx.ndim != 1:
            raise ValueError(
                'Expected in_idx to have ndim == 1; '
                f'(got ndim == {out_idx.ndim})')

        H_site, self_energies = self.site_self_energies(
            energies=energies,
            k_pts=k_pts,
            eta=eta,
            boundary_gfs=boundary_gfs,
            **solver_kwargs)

        site_gf = perturbed_greens_function(
            energies=energies,
            hamiltonian_matrix=H_site,
            self_energies=self_energies,
            eta=eta,
        )
        
        gammas = 1.j*(self_energies - self_energies.mH)
        gammas_in = gammas[in_idx]
        gammas_out = gammas[out_idx]

        # compute T = Tr(gamma_a @ g_ret @ gamma_b @ g_adv), summing over
        # k indices (the "n" index)
        T = torch.einsum(
            '...ij, ...jk, ...kl, ...li -> ...',
            gammas_in,
            site_gf,
            gammas_out,
            site_gf.mH
        ).real

        if not k_resolved:
            T = T.mean(-2)

        # zero out any transition matrix terms where the in and out 
        # leads are the same
        same_sites = (in_idx == out_idx)
        if isinstance(same_sites, bool):
            if same_sites:
                T.zero_()
        else:
            T[same_sites] = 0

        return T

    def current(
            self,
            idx: int,
            energies: torch.Tensor,
            k_pts: torch.Tensor | None = None,
            eta: float = 1e-5,
            k_resolved: bool = False,
            chemical_potentials: 
            list[torch.Tensor] | list[float] | None = None,
            kt: torch.Tensor | float = 1e-5,
            boundary_gfs: list[torch.Tensor] | None = None,
            **solver_kwargs) -> torch.Tensor:
        """Computes the current out of a boundary into the site by integrating
        pver sampled energy values.

        For best results, energies should be sampled uniformly from a few
        multiples of `kt` below the lowest `chemical_potential` value to
        to a few multiples of `kt` above the highest `chemical_potential` 
        value.

        Args:
            idx (int): The index of the boundary.
            energies (torch.Tensor): _description_
            k_pts (torch.Tensor | None, optional): Reciprocal space 
                :math:`\\mathbf{k}` vectors used to sample crystal momentum
                degrees of freedom parallel to the boundary (shape: [M, 3]). 
                Defaults to None.
            eta (float, optional): Broadening parameter of the Green's 
                function.
            k_resolved (bool, optional): If True, the transmission matrix is
                returned without integrating over the sampled
                :math:`\\mathbf{k}` vectors. Defaults to False.
            chemical_potentials (list[torch.Tensor] | 
                list[float] | None, optional): 
                List of chemical potentials :math:`\\mu` of each boundary. 
                Defaults to None.
            kt (torch.Tensor | float, optional): 
                Boltzman's constant times the temperature of the device. 
                Defaults to 1e-5.
            boundary_gfs (list[torch.Tensor] | None, optional): Pre-computed 
                surface Green's functions returned from a call to 
                :meth:`NEGF.surface_greens_functions`. Defaults to None.

        Returns:
           torch.Tensor: The current coming out of boundary into the site
                (shape: [], or [M] if `k_resolved` is `True`)
        """

        # resolve chemical potentials and occupation probabilities
        if chemical_potentials is None:
            chemical_potentials = [0.0]*len(self.boundaries)

        if len(chemical_potentials) != len(self.boundaries):
            raise ValueError(
                'Expected len(chemical_potentials) == len(self.boundaries); '
                f'got {len(chemical_potentials)} != {len(self.boundaries)}.'
            )

        occupation_probs = torch.stack([
            1.0 / (1.0 + torch.exp((energies - mu)/kt))
            for mu in chemical_potentials
        ])

        N = len(self.boundaries)
        out_idx = torch.arange(N, device=energies.device)
        out_idx = out_idx[out_idx != idx]
        in_idx = torch.full_like(out_idx, idx)

        T = self.transmission(
            energies=energies,
            in_idx=in_idx,
            out_idx=out_idx,
            k_pts=k_pts,
            eta=eta,
            k_resolved=k_resolved,
            boundary_gfs=boundary_gfs,
            **solver_kwargs,
        )

        in_f = occupation_probs[in_idx]
        out_f = occupation_probs[out_idx]

        return torch.trapz((T*(in_f - out_f)).sum(0), dim=-1)



        
            



                        


        

        

        
            

        

            

        
        

        

        

        

        

        

        
    

    
