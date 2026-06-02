import pickle
from abc import ABC, abstractmethod
from itertools import product

import numpy as np

from pyquinn.types import AtomGraph
from pyquinn.types.backend import _load_backend_object

SUPPORTED_BACKENDS = {"torch": "pyquinn.torch.types.TorchOperator"}
""" The supported backend Operator objects"""


class OperatorBase:
    """Base class for Operator instances"""


class Operator(OperatorBase):
    """Represents an Operator expanded in a basis of local orbitals at each AtomGraph site.
    Also supports periodic structures, with interactions between unit cell images.
    """

    _backends = {}

    @staticmethod
    def register_backend(name: str):
        """Registers a
            :class:`~pyquinn.types.operator.Operator`
            subclass.

        This should be invoked as a decorator of the subclass. For example:


        .. code-block:: python

            @Operator.register_backend('my_backend_name')
            class MyBackendOperator(BackendOperator):
                ...

        Args:
            name (str): BackendOperator class to be registered.
        """

        def decorator(subclass):
            Operator._backends[name] = subclass
            return subclass

        return decorator

    @staticmethod
    def from_matrix(
        matrix: list[list[float]] | np.ndarray,
        site_indices: list[float],
        orbital_indices: list[float],
    ):
        """Constructs an Operator from a matrix indexed site and orbital indices.

        Args:
            matrix (list[list[float]] | np.ndarray): A matrix representation of an operator
                (shape: [N, N, *value_shape] or [N, N] for scalar values).
            site_indices (list[float]): Indices corresponding to AtomGraph sites (shape: [N])
            orbital_indices (list[float]): Orbital indices corresponding to each site's orbitals,
                starting at 0 for each site. (shape: N)

        Returns:
            Operator: The created Operator
        """

        matrix = np.asarray(matrix)
        onsite_values = np.diagonal(matrix, axis1=0, axis2=1)
        if onsite_values.ndim > 1:
            onsite_values = np.moveaxis(onsite_values, -1, 0)

        if matrix.ndim == 2:
            i, j = np.triu(matrix, 1).nonzero()
        else:
            # determine nonzero entries based on max-abs over value dims
            value_axes = tuple(range(2, matrix.ndim))
            tri_mask = np.triu(np.ones(matrix.shape[:2], dtype=bool), 1)
            max_abs = np.max(np.abs(matrix), axis=value_axes)
            mask = tri_mask & (max_abs != 0)
            i, j = np.nonzero(mask)

        coupling_values = matrix[i, j]
        couplings = np.column_stack((i, j))

        return Operator(
            onsite_values=onsite_values,
            couplings=couplings,
            coupling_values=coupling_values,
            site_indices=site_indices,
            orbital_indices=orbital_indices,
        )

    def load(filename: str) -> "Operator":
        """Helper function to load an Operator
        from a pickle file (`*.pkl`).

        Args:
            filename (str): The name of the pickle file to load.

        Returns:
            Operator: The loaded Operator.
        """
        with open(filename, "rb") as f:
            op = pickle.load(f)

        if not isinstance(op, Operator):
            raise OSError(
                f"Expected an instance of Operator; " f"got: {type(op)}"
            )
        return op

    def __init__(
        self,
        onsite_values: list[float],
        couplings: list[tuple[int, int]],
        coupling_values: list[complex],
        site_indices: list[float],
        orbital_indices: list[float],
        images: list[tuple[int, int, int]] | None = None,
    ):

        self.onsite_values = np.array(onsite_values)
        self.couplings = np.array(couplings, dtype=int)
        self.coupling_values = np.array(coupling_values)
        self.site_indices = np.array(site_indices, dtype=int)
        self.orbital_indices = np.array(orbital_indices, dtype=int)

        if images is not None:
            images = np.array(images, dtype=int)

        self.images = images

        if self.couplings.shape[0] != self.coupling_values.shape[0]:
            raise ValueError(
                "Shape of first dimension of `couplings` must equal that of `coupling_values` "
                "(got: {self.couplings.shape[0]} != {self.coupling_values.shape[0]}"
            )
        if self.onsite_values.shape[0] != self.site_indices.shape[0]:
            raise ValueError(
                "Shape of first dimension of `onsite_values` must equal that of `site_indices` "
                f"(got: {self.onsite_values.shape[0]} != {self.site_indices.shape[0]})"
            )
        if self.onsite_values.ndim > 1 and self.coupling_values.ndim > 1:
            if self.onsite_values.shape[1:] != self.coupling_values.shape[1:]:
                raise ValueError(
                    "Trailing dimensions of `onsite_values` must match those of `coupling_values` "
                    f"(got: {self.onsite_values.shape[1:]} != {self.coupling_values.shape[1:]})"
                )

    def _get_couplings_dict(self, with_images=True):

        couplings_dict = {}

        for i, edge in enumerate(self.couplings):
            edge_key = tuple(edge.tolist())
            value = self.coupling_values[i]

            if with_images:
                image_key = (
                    (0, 0, 0)
                    if self.images is None
                    else tuple(self.images[i].tolist())
                )
                if edge_key not in couplings_dict:
                    couplings_dict[edge_key] = {}
                image_dict = couplings_dict[edge_key]

                if image_key not in image_dict:
                    image_dict[image_key] = np.zeros(
                        value.shape, dtype=self.coupling_values.dtype
                    )
                image_dict[image_key] += value
            else:
                if edge_key not in couplings_dict:
                    couplings_dict[edge_key] = np.zeros(
                        value.shape, dtype=self.coupling_values.dtype
                    )
                couplings_dict[edge_key] += value

        return couplings_dict

    def _sort_couplings_idx(self):
        i, j = self.couplings.T
        if self.images is None:
            sort_idx = np.lexsort((j, i))
        else:
            l, m, n = self.images.T
            sort_idx = np.lexsort((n, m, l, j, i))

        return sort_idx

    def sort_couplings(self):
        sort_idx = self._sort_couplings_idx()

        self.couplings = self.couplings[sort_idx]
        self.coupling_values = self.coupling_values[sort_idx]

        if self.images is not None:
            self.images = self.images[sort_idx]

        return self

    def sort_indices(self):
        sort_idx = np.lexsort((self.orbital_indices, self.site_indices))
        self.site_indices = self.site_indices[sort_idx]
        self.orbital_indices = self.orbital_indices[sort_idx]
        self.couplings = sort_idx[self.couplings]

        return self

    def sort(self):
        self.sort_indices()
        self.sort_couplings()

        return self

    def trim(self, tol=1e-10):

        if self.coupling_values.ndim == 1:
            trim_idx = np.abs(self.coupling_values) > tol
        else:
            value_axes = tuple(range(1, self.coupling_values.ndim))
            trim_idx = (
                np.linalg.norm(self.coupling_values, axis=value_axes) > tol
            )

        self.couplings = self.couplings[trim_idx]
        self.coupling_values = self.coupling_values[trim_idx]

        if self.images is not None:
            self.images = self.images[trim_idx]

        return self

    def _get_site_indices_dict(self):

        # generate map of sites-> orbitals
        site_indices_dict = {}

        for i, site_idx in enumerate(self.site_indices):
            if site_idx not in site_indices_dict:
                site_indices_dict[site_idx] = set()
            site_indices_dict[site_idx].add(i)

        for site_idx, indices in site_indices_dict.items():
            site_indices_dict[site_idx] = sorted(indices)

        return site_indices_dict

    def set_graph_couplings(
        self,
        graph: AtomGraph,
        cutoffs: dict | list[float] | None = None,
        sort: bool = True,
    ):

        neighbors, images = graph.extract().get_neighbor_list(
            cutoffs=cutoffs, return_images=True
        )

        # get old couplings dictionary
        couplings_dict = self._get_couplings_dict(with_images=True)
        site_indices_dict = self._get_site_indices_dict()

        # generate new list of couplings and values to match graph
        graph_couplings = []
        graph_coupling_values = []
        graph_images = []
        value_shape = (
            self.coupling_values.shape[1:]
            if self.coupling_values.ndim > 1
            else ()
        )

        for n_i, (nj_list, image_list) in enumerate(zip(neighbors, images)):
            nj_list = [int(n_i)] + list(nj_list)
            image_list = [(0, 0, 0)] + list(image_list)
            for n_j, im in zip(nj_list, image_list):

                # iterate over all pairwise orbitals
                for i, j in product(
                    site_indices_dict[n_i], site_indices_dict[n_j]
                ):

                    edge_key = (int(i), int(j))
                    image_key = tuple(im)
                    value = np.zeros(
                        value_shape, dtype=self.coupling_values.dtype
                    )

                    # check if edge already exists in this operator (if so, keep it)
                    if (
                        edge_key in couplings_dict
                        and image_key in couplings_dict[edge_key]
                    ):
                        # print(couplings_dict[edge_key][image_key])
                        value += couplings_dict[edge_key][image_key]

                    graph_couplings.append(edge_key)
                    graph_coupling_values.append(value)
                    graph_images.append(image_key)

        self.couplings = np.array(graph_couplings)
        if len(graph_coupling_values) == 0:
            self.coupling_values = np.zeros(
                (0,) + value_shape, dtype=self.coupling_values.dtype
            )
        else:
            self.coupling_values = np.stack(graph_coupling_values, axis=0)
        self.images = np.array(graph_images)

        if sort:
            self.sort()

        return self

    def matrix(
        self,
        k_pts: list[list[float]] | None = None,
        frac_positions: list[list[float]] | None = None,
        graph: AtomGraph | None = None,
    ):

        np_k_pts = np.zeros((1, 3)) if k_pts is None else np.array(k_pts)

        N = self.onsite_values.shape[0]
        N_k = np_k_pts.shape[0]

        # determine the shape/dtype of the matrix required
        value_shape = (
            self.onsite_values.shape[1:] if self.onsite_values.ndim > 1 else ()
        )
        value_dtype = np.result_type(
            self.onsite_values.dtype, self.coupling_values.dtype
        )
        needs_complex = np.iscomplexobj(self.onsite_values) or np.iscomplexobj(
            self.coupling_values
        )
        needs_complex = needs_complex or (self.couplings.size > 0)
        if needs_complex and value_dtype.kind != "c":
            value_dtype = np.result_type(value_dtype, np.complex64)

        # populate matrix
        H_k = np.zeros((N_k, N, N) + value_shape, dtype=value_dtype)
        diag_idx = np.arange(N)
        H_k[:, diag_idx, diag_idx] += self.onsite_values

        np_images = np.zeros((1, 3)) if self.images is None else self.images

        np_frac_distances = 0

        if graph is not None:
            frac_positions = graph.positions @ np.linalg.pinv(graph.unit_cell)

        if frac_positions is not None:
            frac_positions = np.array(frac_positions)
            np_frac_distances = (
                frac_positions[self.site_indices[self.couplings[:, 1]]]
                - frac_positions[self.site_indices[self.couplings[:, 0]]]
            )

        for i, k in enumerate(np_k_pts):
            phase = np.exp(2.0j * np.pi * ((np_frac_distances + np_images) @ k))
            z_k = self.coupling_values * phase.reshape(
                (-1,) + (1,) * len(value_shape)
            )
            np.add.at(H_k[i], (self.couplings[:, 0], self.couplings[:, 1]), z_k)

        return H_k

    def save(self, filename: str):
        """Helper function to save an Operator to a pickle
        file (`*.pkl`).

        Args:
            filename (str): The name of the pickle file to save.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def compile(
        self, backend: str = "torch", **backend_kwargs
    ) -> "BackendOperator":

        if backend not in Operator._backends:
            if backend in SUPPORTED_BACKENDS:
                path = SUPPORTED_BACKENDS[backend]
                Operator._backends[backend] = _load_backend_object(
                    backend, path
                )
            else:
                raise ValueError(f"Unknown Backend Type: {backend}")

        return Operator._backends[backend].compile(
            operator=self, **backend_kwargs
        )

    def extract(self) -> "Operator":
        """Extracts an Operator from its backend, returning a new Operator.
        Since a regular Operator has no associated backend, this method
        simply returns the calling Operator object.

        Returns:
            Operator: The extracted Operator.
        """
        return self


class BackendOperator(ABC, OperatorBase):

    def __init__(
        self,
        onsite_values,
        couplings,
        coupling_values,
        site_indices,
        orbital_indices,
        images,
    ):

        self.onsite_values = onsite_values
        self.couplings = couplings
        self.coupling_values = coupling_values
        self.site_indices = site_indices
        self.orbital_indices = orbital_indices
        self.images = images

    @staticmethod
    @abstractmethod
    def compile(operator: Operator, **backend_kwargs) -> "BackendOperator":
        pass

    @abstractmethod
    def extract(self) -> Operator:
        pass
