import pickle
from abc import ABC, abstractmethod
from typing import Any

import ase
import ase.io
import numpy as np

from pyquinn.types.backend import _load_backend_object
from pyquinn.util.data import (
    ATOMIC_NUMBERS,
    ATOMIC_SYMBOLS,
    NATURAL_CUTOFFS_DICT,
    NATURAL_MASSES_DICT,
)
from pyquinn.util.formula import get_formula
from pyquinn.util.geometry import (
    build_neighbor_list,
    monkhorst_pack_mesh,
    supercell_lattice_points,
)

SUPPORTED_BACKENDS = {"torch": "pyquinn.torch.types.TorchAtomGraph"}
""" The supported backend AtomGraph objects"""


class AtomGraphBase:
    """Base class for AtomGraph instances"""


class AtomGraph(AtomGraphBase):
    """Represents a symbolic graph of atoms in a 3D configuration with
    support for periodic boundary conditions. This graph is in the
    "uncompiled" state, meaning it is not yet associated with a backend,
    and therefore can be manipulated easily in a symbolic fashion.

    When in the "uncompiled state", the AtomGraph can be manipulated with
    symbolic elements (e.g. "X" as a stand-in for the symbol of any element).
    When the graph is compiled via a call to
    :meth:`pyquinn.atoms.AtomGraph.compile`, all symbols must be resolved by
    passing in a `symbols` dictionary (e.g. `{'X' : 'Li'}`). Furthermore, the
    cutoff radii of all elements present in the AtomGraph must be specified
    using the same dictionary format (e.g. `{'Li' : 4.0}`). If not specified,
    the default atomic covalent radii are used.

    An AtomGraph can also be enriched with additional atom-wise input features
    and the output values of models. These can be accessed through the
    following AtomGraph instance variables:

    - `features`: represents additional atom-wise features used as inputs to
        models (e.g. atomic charge).
    - `outputs`: represents atom-wise quantities that are model outputs or
        target output values (e.g. atomic energies or forces).
    - `graph_outputs`: represents global features that are model outputs
        or target values associated with the graph as a whole
        (e.g. total energy).

    Example:
        Here is an example of how an `AtomGraph` object can be constructed and
        then compiled to the PyTorch (`torch`) backend.

        .. code-block:: python

            from pyquinn.types import AtomGraph

            graph = AtomGraph(
                atoms=['Na', 'Cl'],
                positions=[
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0]
                ])

            torch_graph = graph.compile(backend='torch')

            print(torch_graph.atomic_numbers)
            print(torch_graph.positions)
            print(torch_graph.neighbors)

        Note that when an AtomGraph is compiled on a backend, a
        list of atomic neighbors of each site is computed using a
        backend-efficient method, if available. This default behavior
        can be turned off by using `graph.compile(compute_neighbors=False)`.
    """

    _backends = {}

    @staticmethod
    def register_backend(name: str):
        """Registers an :class:`~pyquinn.types.atom_graph.BackendAtomGraph` subclass.

        This should be invoked as a decorator of the subclass. For example:


        .. code-block:: python

            @AtomGraph.register_backend('my_backend_name')
            class MyBackendAtomGraph(BackendAtomGraph):
                ...

        Args:
            name (str): BackendAtomGraph class to be registered.
        """

        def decorator(subclass):

            AtomGraph._backends[name] = subclass
            return subclass

        return decorator

    @staticmethod
    def from_ase(
        atoms: ase.Atoms,
        features: list[list[float]] | None = None,
        outputs: list[list[float]] | None = None,
        graph_outputs: list[float] = None,
    ) -> "AtomGraph":
        """Constructs an AtomGraph from an ase.Atoms object.

        Args:
            atoms (ase.Atoms): An `ase.Atoms` object from which the AtomGraph
                will be constructed.
            features (list[list[float]]):
            outputs: (list[list[float]]):
            graph_outputs (list[list[float]]):

        Returns:
            AtomGraph: An AtomGraph constructed from the `ase.Atoms` object.

        Example:
            This is an example of how an AtomGraph can be constructed from an
            `ase.Atoms` object:

            .. code-block:: python

                from ase.build import bulk

                atoms = bulk('Cu') # create a bulk fcc Cu ase.Atoms object

                graph = AtomGraph.from_ase(atoms)

                print(graph.atoms)
        """

        if features is None:
            features = np.zeros((len(atoms.get_atomic_numbers()), 0))
        return AtomGraph(
            atoms=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            unit_cell=atoms.get_cell(),
            periodic=atoms.get_pbc(),
            features=features,
            outputs=outputs,
            graph_outputs=graph_outputs,
        )

    @staticmethod
    def from_file(filename: str) -> "AtomGraph":
        """Helper function to load an AtomGraph from an external file
        that is one of the ASE-supported file formats. For more
        information on supported file formats, see
        :meth:`ase.io.write`.

        Args:
            filename (str): The name of the file to load.

        Returns:
            AtomGraph: The loaded AtomGraph.
        """
        atoms = ase.io.read(filename)
        return AtomGraph.from_ase(atoms)

    @staticmethod
    def load(filename: str) -> "AtomGraph":
        """Helper function to load an AtomGraph from a pickle
        file (`*.pkl`).

        Args:
            filename (str): The name of the pickle file to load.
        """
        with open(filename, "rb") as f:
            graph = pickle.load(f)

        if not isinstance(graph, AtomGraph):
            raise OSError(
                f"Expected an instance of Atomgraph; " f"got: {type(graph)}"
            )

        return graph

    def __init__(
        self,
        atoms: list | AtomGraphBase | None = None,
        positions: list[list[float]] | None = None,
        unit_cell: list[list[float]] | None = None,
        periodic: tuple[bool] | bool = False,
        features: list[list[float]] | None = None,
        outputs: list[list[float]] | None = None,
        graph_outputs: list[float] | None = None,
        coordinates: str = "cartesian",
        origin: list[float] | None = None,
    ):
        """Constructs an AtomGraph object, which represents a symbolic graph
        of atoms in a 3D configuration with support
        for periodic boundary conditions.

        Args:
            atoms (list | AtomGraphBase | None, optional): A list of atomic
                symbols or atomic numbers (shape: [N_atoms]).
                Can also be another AtomGraphBase object. Defaults to None.
            positions (list[list[float]] | None, optional): A list of
                Cartesian positions of
                atoms in the graph (shape: [N_atoms, 3]). Defaults to None.
            unit_cell (list[list[float]] | None, optional): The 3x3 unit cell
                matrix where `unit_cell[i,j]` corresponds to the `j`th
                component of the `i`th unit cell vector. (shape: [3,3]).
                Only applies if periodic is not False. Defaults to None.
            periodic (tuple[bool] | bool, optional): The periodicity of the
                system. If True, the AtomGraph is assumed to have periodicity
                in all directions. If a tuple, the `True` elements of the
                tuple indicate periodicity with respect to the corresponding
                unit cell vector. Defaults to False.
            features (list[list[float]] | None, optional): An optional list of
                atom-wise features used as inputs for graph-based models
                (shape: [N_atoms, ...]). Defaults to None.
            outputs (list[list[float]] | None, optional): An optional list
                of atomwise outputs representing model outputs or model target
                values (shape: [N_atoms, ...]). Defaults to None.
            graph_outputs (list[float] | None, optional): An optional list of
                global graph outputs representing model outputs or target
                values corresponding to the graph as a whole
                (shape: [...]). Defaults to None.
            coordinates (str, optional): Whether cartesian or unit cell
                coordinates used (must be either  'fractional' or 'cartesian').
                Defaults to 'cartesian'.
            origin (list[float] | None, optional):
                An optional offset of the unit cell origin, which is added to
                the stored atomic positions (shape: [3]). Defaults to None.

        Raises:
            ValueError: If incompatible or invalid arguments are supplied.
        """

        if isinstance(atoms, AtomGraphBase):
            atom_graph = atoms.extract()
            atoms = atom_graph.atoms
            positions = atom_graph.positions
            unit_cell = atom_graph.unit_cell
            periodic = atom_graph.periodic
            features = atom_graph.features
            outputs = atom_graph.outputs
            graph_outputs = atom_graph.graph_outputs

        self.atoms = atoms if atoms is not None else []
        self.positions = (
            positions if positions is not None else np.zeros((0, 3))
        )

        self.unit_cell = (
            unit_cell if unit_cell is not None else np.zeros((3, 3))
        )

        self.features = (
            features
            if features is not None
            else np.zeros((self.atoms.shape[0], 0))
        )

        self.outputs = (
            outputs
            if outputs is not None
            else np.zeros((self.atoms.shape[0], 0))
        )

        self.graph_outputs = (
            graph_outputs if graph_outputs is not None else np.zeros(0)
        )

        self.periodic = periodic

        # validate features
        self._validate_features()

        # convert fractional positions to cartesian:
        if coordinates == "fractional":
            self.positions = self.positions @ self.unit_cell
        elif coordinates != "cartesian":
            raise ValueError(
                'Coordinates must be either "fractional" ' 'or "cartesian".'
            )

        # configure unit cell origin
        if origin is not None:
            origin = np.array(origin)
            if origin.shape != (3,):
                raise ValueError(
                    f"Expected origin.shape = (3,), " f"got: {origin.shape}"
                )

            self.positions += origin

    @property
    def atoms(self):
        """Returns the atomic symbols of atoms in this AtomGraph.

        Returns:
            np.ndarray[str]: The list of atomic symbols in this AtomGraph.
        """
        return self._atoms

    @atoms.setter
    def atoms(self, value):
        """Updates the atomic symbols of atoms in this AtomGraph."""

        atomic_symbols = np.array(
            [a if isinstance(a, str) else ATOMIC_SYMBOLS[int(a)] for a in value]
        )
        self._atoms = atomic_symbols

    @property
    def positions(self):
        """Returns the cartesian positions of atoms in this AtomGraph.

        Returns:
            np.ndarray: Cartesian positions of atoms in this AtomGraph
                        (shape: [n_atoms, 3]).
        """
        return self._positions

    @positions.setter
    def positions(self, value):
        """Updates the cartesian positions of atoms in this AtomGraph."""
        positions = np.array(value, dtype=np.float64)

        if positions.ndim != 2:
            raise ValueError(
                f"Expected positions.ndim = 2, got: {positions.ndim}"
            )
        if positions.shape[1] != 3:
            raise ValueError(
                f"Expected positions.shape[1] = 3, got: {positions.shape}"
            )

        self._positions = positions

    @property
    def unit_cell(self):
        """Returns the unit cell matrix of this AtomGraph.

        Returns:
            np.ndarray: Unit cell matrix of this AtomGraph
                        (shape: [3,3]).
        """
        return self._unit_cell

    @unit_cell.setter
    def unit_cell(self, value):
        """Updates the unit cell matrix of this AtomGraph."""
        unit_cell = np.array(value)

        if unit_cell.shape != (3, 3):
            raise ValueError(
                f"Expected unit_cell.shape = (3, 3), " f"got: {unit_cell.shape}"
            )

        self._unit_cell = unit_cell

    @property
    def periodic(self):
        """Returns the periodicity vector of this AtomGraph.

        Returns:
            np.ndarray[bool]: Periodicty vector
        """
        return self._periodic

    @periodic.setter
    def periodic(self, value):
        """Updates the periodicity of this AtomGraph."""
        if isinstance(value, bool):
            value = [value] * 3
        periodic = np.array(value, dtype=bool)

        if periodic.shape != (3,):
            raise ValueError(
                f"Expected periodic.shape = (3), got" f"got: {periodic.shape}"
            )

        self._periodic = periodic

    @property
    def features(self):
        """Returns the atomwise feature tensor of this AtomGraph.

        Returns:
            np.ndarray: Atomwise feature array.
        """
        return self._features

    @features.setter
    def features(self, value):
        """Updates the atomwise feature tensor of this AtomGraph.

        Args:
            value (np.ndarray): Atomwise feature tensor
        """
        features = np.array(value)
        self._features = features

    @property
    def outputs(self):
        """Returns the atomwise outputs tensor of this AtomGraph.

        Returns:
            np.ndarray: Atomwise outputs tensor
        """
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        """Updates the atomwise outputs tensor of this AtomGraph.

        Args:
            value (np.ndarray): Atomwise outputs tensor
        """
        outputs = np.array(value)
        self._outputs = outputs

    @property
    def graph_outputs(self):
        """Returns the graph outputs of this AtomGraph

        Returns:
            np.ndarray: Graph outputs array.
        """
        return self._graph_outputs

    @graph_outputs.setter
    def graph_outputs(self, value):
        """Updates the graph outputs of this AtomGraph.

        Args:
            value (np.ndarray): Graph outputs array
        """
        graph_outputs = np.array(value)
        self._graph_outputs = graph_outputs

    def chemical_formula(self) -> str:
        """Returns the chemical formula of this AtomGraph using Hill notation.

        Returns:
            str: The chemical formula.
        """

        return get_formula(self.atoms, format="hill")

    def __str__(self) -> str:
        """Returns a printable string representation of this AtomGraph"""
        visible_vars = {
            "atoms": self.chemical_formula(),
            "periodic": self.periodic.tolist(),
            "unit_cell": self.unit_cell.tolist(),
        }
        attrs = ", ".join(f"{k}={v!r}" for k, v in visible_vars.items())
        return f"AtomGraph({attrs})"

    def __repr__(self) -> str:
        """Returns a full string representation of this AtomGraph"""
        return str(self)

    def __getitem__(self, key: slice | np.ndarray) -> "AtomGraph":
        """Accesses a sub-AtomGraph at a given slice or numpy array index."""
        if isinstance(key, (int, np.integer)) or (
            isinstance(key, np.ndarray) and key.ndim <= 0
        ):
            raise KeyError(
                "Index key must be a valid numpy array index or slice with "
                "dimension greater than 0. Integer indexing is not supported "
                "for AtomGraph objects."
            )

        # create an AtomGraph with sliced views of
        _graph = AtomGraph()
        _graph._atoms = self._atoms[key]
        _graph._positions = self._positions[key]
        _graph._periodic = self._periodic
        _graph._unit_cell = self._unit_cell
        _graph._features = self._features[key]
        _graph._outputs = self._outputs[key]
        _graph._graph_outputs = self._graph_outputs

        return _graph

    def __len__(self) -> int:
        """Returns the number of atoms in this AtomGraph"""
        return self.atoms.shape[0]

    def _resolve_atomic_numbers(
        self, symbols: dict | list[str] | None = None
    ) -> np.ndarray:
        """Helper method to automatically resolve atomic numbers"""
        # convert all symbols to valid atomic symbols
        if symbols is None:
            atomic_symbols = np.array(self.atoms)
        elif isinstance(symbols, list):
            atomic_symbols = np.array(symbols)
        elif isinstance(symbols, dict):
            atomic_symbols = np.array(
                [symbols[s] if s in symbols else s for s in self.atoms]
            )

        # validate symbols
        for s in atomic_symbols:
            if len(s) == 0 or s not in ATOMIC_NUMBERS:
                raise ValueError(f'Unknown atomic symbol: "{s}"')

        # convert to atomic numbers
        atomic_numbers = np.array([ATOMIC_NUMBERS[s] for s in atomic_symbols])

        return atomic_numbers

    def _resolve_cutoffs(
        self, cutoffs: dict | list[float] | None = None
    ) -> np.ndarray:
        """Helper method to automatically resolve atomic cutoffs"""
        if cutoffs is None:
            cutoffs = np.array([NATURAL_CUTOFFS_DICT[a] for a in self.atoms])
        elif isinstance(cutoffs, list):
            cutoffs = np.array(cutoffs)
            if cutoffs.shape[0] != self.positions.shape[0]:
                raise ValueError(
                    f"Expected cutoffs.shape[0] = positions.shape[0], "
                    f"got: {cutoffs.shape[0]} != {self.positions.shape[0]}"
                )
        elif isinstance(cutoffs, dict):
            cutoffs_symbols = {
                (ATOMIC_SYMBOLS[int(k)] if not isinstance(k, str) else k): v
                for k, v in cutoffs.items()
            }

            cutoffs_dict = NATURAL_CUTOFFS_DICT | cutoffs_symbols
            cutoffs = np.array([cutoffs_dict[a] for a in self.atoms])
        else:
            raise ValueError("Expected cutoffs to be a list or dictionary.")

        return cutoffs

    def _resolve_masses(
        self, masses: dict | list[float] | None = None
    ) -> np.ndarray:
        """Helper method to automatically resolve atomic masses"""
        if masses is None:
            masses = np.array([NATURAL_MASSES_DICT[a] for a in self.atoms])
        elif isinstance(masses, list):
            masses = np.array(masses)
            if masses.shape[0] != self.positions.shape[0]:
                raise ValueError(
                    f"Expected masses.shape[0] = positions.shape[0], "
                    f"got: {masses.shape[0]} != {self.positions.shape[0]}"
                )
        elif isinstance(masses, dict):
            masses_symbols = {
                (
                    NATURAL_MASSES_DICT[int(k)] if not isinstance(k, str) else k
                ): v
                for k, v in masses.items()
            }

            masses_dict = NATURAL_MASSES_DICT | masses_symbols
            masses = np.array([masses_dict[a] for a in self.atoms])
        else:
            raise ValueError("Expected masses to be a list or dictionary.")

        return masses

    def _validate_features(self):
        """Helper method to verify the shape of features is correct."""
        if self.features.shape[0] != self.positions.shape[0]:
            raise ValueError(
                f"Expected features.shape[0] = positions.shape[0], "
                f"got: {self.features.shape[0]} != {self.positions.shape[0]}"
            )

    def _validate_outputs(self):
        """Helper method to verify the shape of outputs is correct."""
        if self.outputs.shape[0] != self.positions.shape[0]:
            raise ValueError(
                f"Expected targets.shape[0] = positions.shape[0], "
                f"got: {self.outputs.shape[0]} != {self.positions.shape[0]}"
            )

    def get_neighbor_list(
        self,
        cutoffs: dict | list[float] | None = None,
        return_images: bool = False,
    ) -> tuple[list]:
        """Returns list of neighbors and their atoms based on the supplied
        cutoff values. Two atoms are considered to be neighbors if their
        cutoff spheres intersect; that is, if their pairwise distance
        is less than the sum of their cutoff radii. The periodicity of the
        AtomGraph is also taken into account when computing neighbors.

        .. note::
            This function uses the :meth:`ase.neighborlist.NeighborList`
            class to generate a list of neighboring atoms based on the
            supplied cutoffs.

            While this method works for larger graphs (and is currently
            the fall-back method for some backends), it may be slower
            than the backend-specific methods. Ideally, this method
            should only be used for testing cutoff values or as a
            fall-back method.

        Args:
            cutoffs (dict | list[float] | None, optional): Cutoffs of
                each atom in the graph (if a list) or of each element in
                the graph (if a dict). If None or certain elements
                are not specified, the cutoff values default to a
                "safe" value that is proportional to the covalent
                bonding radius. Defaults to None.
            return_images (bool, optional): If True, returns the unit cell
                image vectors of each neighbor instead of the distance
                vectors. Defaults to False.

        Returns:
            tuple[list]: A tuple containing the following:

            - neighbor_list (list): a list of lists of indices
                representing the neighbors of each atom in the AtomGraph.
            - neighbor_vectors (list): a list of vectors corresponding to
                the distance between each atom site and its neighbors. If
                `return_images` is `True`, the returned values are integer
                vectors corresponding to images of the unit cell.
        """

        neighbors, unit_cell_images = build_neighbor_list(
            positions=self.positions,
            unit_cell=self.unit_cell,
            periodic=self.periodic,
            cutoffs=self._resolve_cutoffs(cutoffs),
        )

        neighbor_list = []
        neighbor_vectors = []
        for i, (neighbors_i, images_i) in enumerate(
            zip(neighbors, unit_cell_images)
        ):

            neighbor_list.append(neighbors_i)

            if return_images:
                neighbor_vectors.append(images_i)
            else:

                distances = (
                    []
                    if len(neighbors_i) == 0
                    else (
                        (self.positions[neighbors_i] - self.positions[i])
                        + (images_i @ self.unit_cell)
                    )
                )
                neighbor_vectors.append(distances)

        return neighbor_list, neighbor_vectors

    def substitute(self, symbols: dict):
        """Substitutes a set of symbols (elements or variables) in
        this AtomGraph.

        Args:
            symbols (dict): A substitution dictionary. Keys and values
            correspond to symbols in this AtomGraph and their substituted
            values respectively.

        """
        for i, atom_symbol in self.atoms:
            if atom_symbol in symbols:
                self.atoms[i] = symbols[atom_symbol]

    def wrap(self) -> "AtomGraph":
        """Wraps positions of atoms to lie within the unit cell."""
        images = np.floor(self.positions @ self.unit_rcell().T)
        self.positions -= images @ self.unit_cell

        return self

    def unit_rcell(self, periodicity: bool = True) -> np.ndarray:
        """Computes the reciprocal unit cell matrix.

        Returns:
            np.ndarray: The reciprocal unit cell (shape: [3, 3])
        """
        if periodicity:
            delta = self.periodic.astype(self.unit_cell.dtype)
            rcell = np.linalg.lstsq(self.unit_cell.T, np.diag(delta))[0]
        else:
            rcell = np.linalg.pinv(self.unit_cell.T)

        return rcell

    def frac_positions(self) -> np.ndarray:
        """Computes the fractional coordinates of atoms within the unit cell.

        Coordinates are set to zero for unit cell vectors of length zero.

        Returns:
            np.ndarray: Fractional positions of atoms (shape: [n_atoms, 3])
        """
        inv_unit_cell = np.linalg.pinv(self.unit_cell)
        return self.positions @ inv_unit_cell

    def band_path(
        self, path: str | None = None, n_points: int = 2048
    ) -> np.ndarray:
        """Generates a High-symmetry canonical band path based on the ASE
        implementation of `ase.Cell.bandpath()`. The variable `path` specifies
        a path in the first Brillouin zone, using the conventions specified in
        the ASE Brillouin zone data page:

        https://wiki.fysik.dtu.dk/ase/ase/dft/bztable.html


        Reciprocal cell points along the band path are returned in
        fractional coordinates.

        Args:
            path (str | None, optional): High-symmetry path in the reciprocal
                space first Brillouin zone, consisting of special point
                symbols, possibly separated by commas to indicate
                jumps in the path. If None, the canonical path is used.
                Defaults to None.
            n_points (int, optional): Number of points in the band path.
                Defaults to 1024.

        Returns:
            np.ndarray: A numpy array of fractional cell reciprocal space points
                (shape: [n_points, 3])
        """
        graph_cell = ase.Atoms(cell=self.unit_cell).cell
        frac_k_pts = graph_cell.bandpath(path=path, npoints=n_points).kpts

        return frac_k_pts

    def unit_cell_mesh(self, size: tuple[int] | int = 5) -> np.ndarray:
        """Generates a uniform Monkhorst-Pack mesh that uniformly samples
        the unit cell.

        Args:
            size (tuple[int] | int, optional): A tuple of three ints
                (or a single int) specifying the Monhorst-Pack mesh
                size. If a single int is passed, this is used as the
                size of each mesh dimension. Defaults to 5.

        Returns:
            np.ndarray: A Monkhorst-Pack reciprocal cell mesh.
        """
        mesh = monkhorst_pack_mesh(size)
        return mesh @ self.unit_cell

    def unit_rcell_mesh(self, size: tuple[int] | int = 5) -> np.ndarray:
        """Generates a uniform Monkhorst-Pack mesh that uniformly samples
        the reciprocal unit cell.

        Args:
            size (tuple[int] | int, optional): A tuple of three ints
                (or a single int) specifying the Monhorst-Pack mesh
                size. If a single int is passed, this is used as the size
                of each mesh dimension. Defaults to 5.

        Returns:
            np.ndarray: A Monkhorst-Pack reciprocal cell mesh.
        """
        mesh = monkhorst_pack_mesh(size)
        return mesh

    def supercell(self, size: list[list[float]]) -> "AtomGraph":
        """Constructs an expanded supercell of this AtomGraph
        based on `size`, which can be either a 3x3 supercell matrix,
        an array of size 3 (interpreted as a diagonal matrix), or
        a scalar (interpreted as a constant diagonal matrix).

        Args:
            size (list[list[float]]): The size matrix, list or scalar of
                the supercell.

        Returns:
            AtomGraph: The supercell AtomGraph
        """

        size = np.array(size)

        matrix, lattice_points = supercell_lattice_points(size)

        supercell = matrix @ self.unit_cell
        N = lattice_points.shape[0]
        n_atoms = N * self.positions.shape[0]
        _positions = self.positions.reshape((1, -1, 3))
        _offsets = (lattice_points @ self.unit_cell).reshape((-1, 1, 3))
        supercell_positions = (_positions + _offsets).reshape((n_atoms, 3))

        graph = AtomGraph(
            atoms=self.atoms.repeat(N, axis=0),
            positions=supercell_positions,
            unit_cell=supercell,
            periodic=self.periodic,
            features=self.features.repeat(N, axis=0),
            outputs=self.outputs.repeat(N, axis=0),
            graph_outputs=self.graph_outputs.repeat(N, axis=0),
        )

        return graph

    def compile(
        self,
        backend: str = "torch",
        cutoffs: dict | list[float] | None = None,
        masses: dict | list[float] | None = None,
        symbols: dict | list[str] | None = None,
        **backend_kwargs,
    ) -> "BackendAtomGraph":
        """Returns a compiled form of this AtomGraph that targets a specific
        backend. A compiled AtomGraph can then be manipulated using
        backend-specific methods. This is necessary, for example, if
        backend-specific autograd suppport is desired with respect to atomic
        positions or features.

        To recover an AtomGraph from a backend-specific implementation,
        use the `extract()` function.

        Args:
            backend (str): The backend to target. Defaults to 'torch'.
            cutoffs (dict | list[float] | None, optional): Cutoff radii used
                to compute neighbors of atoms in the compiled graph. If None
                or cutoffs are not supplied for certain elements,
                a "safe" default value is used for those elements. Defaults
                to None.
            masses (dict | list[float] | None, optional): Atomic masses of
                each atom or element. If None, or masses are not supplied,
                standard atomic masses are used. Defaults to None.
            symbols (dict | list[str] | None, optional): A dict of symbolic
            element substitutions made prior to compilation. Defaults to None.

        Raises:
            ValueError: If the backend is unknown, or compilation fails.

        Returns:
            BackendAtomGraph: An atom graph compiled for the backend.
        """

        # resolve cutoff and mass arrays:
        cutoffs = self._resolve_cutoffs(cutoffs)
        masses = self._resolve_masses(masses)
        atomic_numbers = self._resolve_atomic_numbers(symbols)

        if backend not in AtomGraph._backends:
            if backend in SUPPORTED_BACKENDS:
                path = SUPPORTED_BACKENDS[backend]
                AtomGraph._backends[backend] = _load_backend_object(
                    backend, path
                )
            else:
                raise ValueError(f"Unknown Backend Type: {backend}")

        return AtomGraph._backends[backend].compile(
            graph=self,
            atomic_numbers=atomic_numbers,
            cutoffs=cutoffs,
            masses=masses,
            **backend_kwargs,
        )

    def to_ase(self) -> ase.Atoms:
        """Returns an `ase.Atoms` object from an AtomGraph. Note that when
        converting to an `ase.Atoms` object, the data contained in
        the `features`, `outputs` and `graph_outputs`
        is not retained.

        Returns:
            ase.Atoms: An `ase.Atoms` object constructed from this AtomGraph.
        """
        atoms = ase.Atoms(
            numbers=self._resolve_atomic_numbers(),
            masses=self._resolve_masses(),
            positions=self.positions,
            cell=self.unit_cell,
            pbc=self.periodic,
        )

        return atoms

    def to_file(self, filename: str, format: str | None = None):
        """Helper function to write an AtomGraph to a file
        that is one of the ASE-supported file formats. For more
        information on supported file formats, see
        :meth:`ase.io.write`.

        Args:
            filename (str): The name of the file to write
            format (str, optional): The file format to use. If None,
                the format is inferred from `filename`.
        """
        atoms = self.to_ase()
        ase.io.write(filename=filename, images=atoms, format=format)

    def save(self, filename: str):
        """Helper function to save an AtomGraph to a pickle
        file (`*.pkl`).

        Args:
            filename (str): The name of the pickle file to save.
        """

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def extract(self) -> "AtomGraph":
        """Extracts an AtomGraph from its backend, returning a new AtomGraph.
        Since a regular AtomGraph has no associated backend, this method
        simply returns the calling AtomGraph object.

        Returns:
            AtomGraph: The extracted AtomGraph.
        """
        return self

    def plot(self, backend: str = "mpl", **plotter_kwargs) -> Any:
        """Plots an AtomGraph object using the given backend.

        Args:
            backend (str, optional): The backend plotter to use.
                Defaults to 'mpl'. For more information on plotter
                arguments, see :meth:`pyquinn.plot.plotter.plot_atoms()`.

        Returns:
            Any: A figure object associated with the backend plotter.
        """
        from pyquinn.plot import plotter

        return plotter.plot_atoms(atoms=self, backend=backend, **plotter_kwargs)

    def copy(self) -> "AtomGraph":
        """Returns a copy of this AtomGraph object.

        Returns:
            AtomGraph: A copy of this AtomGraph object.
        """

        return AtomGraph(self)


class BackendAtomGraph(ABC, AtomGraphBase):

    def __init__(
        self,
        atomic_numbers,
        positions,
        masses,
        unit_cell,
        periodic,
        features,
        outputs,
        graph_outputs,
        cutoffs,
        compute_neighbors: bool = True,
    ):

        self.atomic_numbers = atomic_numbers
        self.positions = positions
        self.masses = masses
        self.unit_cell = unit_cell
        self.periodic = periodic
        self.features = features
        self.outputs = outputs
        self.graph_outputs = graph_outputs

        self.cutoffs = cutoffs
        self.neighbors = None
        self.unit_cell_images = None

        if compute_neighbors:
            self.update_neighbors()

    @staticmethod
    @abstractmethod
    def compile(
        graph: AtomGraph,
        atomic_numbers: list,
        cutoffs: list,
        masses: list,
        **backend_kwargs,
    ) -> "BackendAtomGraph":
        pass

    @abstractmethod
    def update_neighbors(self):
        """Updates `self.neighbors` and
        `self.unit_cell_images` using atomic positions in
        `self.positions` and the cutoff radii in
        `self.cutoffs`.

        Ideally, this should use a backend-efficient
        implementation, but may defer to the
        less-efficient methods in
        :meth:`pyquinn.atoms.geometry` as a fall-back.
        """
        pass

    @abstractmethod
    def wrap(self):
        """Wraps positions of atoms to lie within the unit cell."""
        pass

    @abstractmethod
    def band_path(self, path: str | None = None, n_points: int = 2048) -> Any:
        """Generates a High-symmetry canonical band path using the same
        conventions as in :meth:`pyquinn.types.AtomGraph.get_band_path`,
        using backend-specific data types.

        Reciprocal cell points along the band path are returned in
        fractional coordinates.

        Args:
            path (str | None, optional): High-symmetry path in the reciprocal
                space first Brillouin zone, consisting of special point
                symbols, possibly separated by commas to indicate
                jumps in the path. If None, the canonical path is used.
                Defaults to None.
            n_points (int, optional): Number of points in the band path.
                Defaults to 1024.

        Returns:
                An array of fractional cell reciprocal space points
                (shape: [n_points, 3]).
        """
        pass

    @abstractmethod
    def unit_rcell_mesh(self, size: tuple[int] | int = 5) -> Any:
        """Generates a Monkhorst-Pack mesh that uniformly samples
        the reciprocal unit cell, using the same
        conventions as in :meth:`pyquinn.types.AtomGraph.get_rcell_mesh`.
        Differentiability with respect to unit cell shape is also maintained.

        Args:
            size (tuple[int] | int, optional): A tuple of three ints
                (or a single int) specifying the Monhorst-Pack mesh
                size. If a single int is passed, this is used as the size
                of each mesh dimension. Defaults to 5.

        Returns:
            A Monkhorst-Pack reciprocal cell mesh.
        """
        pass

    @abstractmethod
    def unit_cell_mesh(self, size: tuple[int] | int = 5) -> Any:
        """Generates a uniform Monkhorst-Pack mesh that uniformly samples
        the real-space unit cell, using the same
        conventions as in :meth:`pyquinn.types.AtomGraph.get_rcell_mesh`
        Differentiability with respect to unit cell shape is also maintained.

        Args:
            size (tuple[int] | int, optional): A tuple of three ints
                (or a single int) specifying the Monhorst-Pack mesh
                size. If a single int is passed, this is used as the
                size of each mesh dimension. Defaults to 5.

        Returns:
            A Monkhorst-Pack unit cell mesh.
        """
        pass

    @abstractmethod
    def supercell(self, size: Any) -> "BackendAtomGraph":
        """Constructs an expanded supercell of this AtomGraph
        based on `size`, which can be either a 3x3 supercell matrix,
        an array of size 3 (interpreted as a diagonal matrix), or
        a scalar (interpreted as a constant diagonal matrix).

        Differentiability with respect to all parameters
        in the original BackendAtomgraph is also maintained.

        Args:
            size (list[list[float]]): The size matrix, list or scalar of
                the supercell.

        Returns:
            BackendAtomGraph: The supercell BackendAtomGraph
        """
        pass

    @abstractmethod
    def frac_positions(self) -> np.ndarray:
        """Computes the fractional coordinates of atoms within the unit cell.
        Differentiability with respect to unit cell shape is maintained.

        Coordinates are set to zero for unit cell vectors of length zero.

        Returns:
            Fractional positions of atoms (shape: [n_atoms, 3]).
        """
        pass

    @abstractmethod
    def extract(self) -> AtomGraph:
        """Returns an instance of a concrete AtomGraph
        which is removed from any backend-specific
        implementations."""
        pass

    @staticmethod
    @abstractmethod
    def load(filename: str) -> "BackendAtomGraph":
        """Helper function to load a BackendAtomGraph to a pickle
        file. For backends without an established serialization
        format, the file extension convention of `*.<backend>.pkl`
        is recommended.

        Args:
            filename (str): The name of the pickle file to save.

        Returns:
            BackendAtomGraph: The loaded BackendAtomGraph
        """
        pass

    @abstractmethod
    def save(self, filename: str):
        """Helper function to save a BackendAtomGraph to a pickle
        file. For backends without an established serialization
        format, the file extension convention of `*.<backend>.pkl`
        is recommended.

        Args:
            filename (str): The name of the pickle file to save.
        """
        pass

    def plot(self, backend: str = "mpl", **plotter_kwargs) -> Any:
        """Plots a BackendAtomGraph object using the given backend.

        Args:
            backend (str, optional): The backend plotter to use.
                Defaults to 'mpl'. For more information on plotter
                arguments, see :meth:`pyquinn.plot.plotter.plot_Atoms`.

        Returns:
            Any: A figure object associated with the backend plotter.
        """
        from pyquinn.plot import plotter

        return plotter.plot_atoms(
            atoms=self.extract(), backend=backend, **plotter_kwargs
        )

    @abstractmethod
    def copy(self) -> "BackendAtomGraph":
        """Returns a copy of this BackendAtomGraph object

        Returns:
            BackendAtomGraph: A copy of this BackendAtomGraph.

        """
        pass
