import unittest
import numpy as np

from pyquinn.atoms.atom_data import ATOMIC_SYMBOLS
from pyquinn.atoms.atom_graph import AtomGraph, SUPPORTED_BACKENDS


class AtomGraphTest(unittest.TestCase):

    def test_atom_graph(self):
        
        n_elements = 118

        atomic_numbers = list(range(1, n_elements+1))
        atoms = [ATOMIC_SYMBOLS[z] for z in atomic_numbers]

        positions = np.linspace(0, 200, n_elements*3)\
            .reshape(n_elements, 3).tolist()
        unit_cell = np.linspace(0, 10, 9).reshape(3, 3)
        features = np.zeros((n_elements, 1, 5))
        periodic = (True, False, True)

        # build atom graph
        graph = AtomGraph(
            atoms=atomic_numbers, 
            positions=positions,
            unit_cell=unit_cell, 
            periodic=periodic, 
            features=features)

        # test all properties of atom graph
        self.assertTrue(np.all(graph.atoms == atoms))
        self.assertTrue(np.allclose(graph.positions, positions))
        self.assertTrue(np.allclose(graph.features, features))
        self.assertTrue(np.allclose(graph.periodic, periodic))

        # test setting the atoms with atomic numbers:
        self.atoms = atomic_numbers
        self.assertTrue(np.all(graph.atoms == atoms))

    def test_atom_graph_fractional(self):

        atoms = ['K', 'Pt', 'U']
        positions = np.diag(np.arange(1, 4)).tolist()
        unit_cell = np.arange(9).reshape((3, 3)).tolist()
        origin = np.arange(3).tolist()

        graph = AtomGraph(
            atoms=atoms,
            positions=positions,
            unit_cell=unit_cell,
            periodic=True,
            origin=origin,
            coordinates='fractional',
        )

        true_positions = np.array([
            [0, 2, 4],
            [6, 9, 12],
            [18, 22, 26]
        ])

        # ensure atoms are translated to atomic numbers:
        self.assertTrue(np.all(graph.atoms == np.array(atoms)))

        # ensure fractional positions are computed correctly
        self.assertTrue(tuple(graph.periodic), (True, True, True))
        self.assertTrue(np.allclose(graph.positions, true_positions))

    def test_compute_neighbor_list(self):

        graphene_atoms = ['C', 'C']
        graphene_a = 0.1 #2.8
        graphene_positions = np.array([
            [1.      , 0.        , 0.           ], # noqa
            [0.5     , 1/np.sqrt(12), 0.        ]  # noqa 
        ])*graphene_a

        graphene_unit_cell = np.array([
            [ 1,   0.,            0. ], # noqa
            [-0.5, np.sqrt(3/4),  0. ], # noqa
            [ 0.,  0.,            0. ]  # noqa
        ])*graphene_a

        graphene_periodic = [True, True, False]

        # generate graphene lattice:
        graphene = AtomGraph(
            atoms=graphene_atoms,
            positions=graphene_positions,
            unit_cell=graphene_unit_cell,
            periodic=graphene_periodic
        )

        neighbors_1, vectors = graphene.get_neighbor_list(
            {'C': graphene_a*1.01/2})
        neighbors_2, images = graphene.get_neighbor_list(
            {'C': graphene_a*1.01/2}, return_images=True)

        distances = np.array(
            [graphene_a/np.sqrt(3)]*3 +
            [graphene_a]*6
        )
        self.assertTrue(np.allclose(
            np.sort(np.sqrt((vectors[0]**2).sum(-1))), distances))
        self.assertTrue(np.allclose(
            np.sort(np.sqrt((vectors[1]**2).sum(-1))), distances))

        si_atoms = ['Si', 'Si']
        si_a = 3.0
        si_positions = np.array([
            [0.,   0.,   0.    ], # noqa
            [1/4,  1/4,  1/4   ]  # noqa
        ])*si_a
        si_unit_cell = np.array([
            [0. ,   1/2,   1/2  ], # noqa
            [1/2,   0. ,   1/2  ], # noqa
            [1/2,   1/2,   0.   ]  # noqa
        ])*si_a

        # generate silicon diamond lattice:
        silicon = AtomGraph(
            atoms=si_atoms,
            positions=si_positions,
            unit_cell=si_unit_cell,
            periodic=True
        )

        si_neighbors_1, si_vectors = silicon.get_neighbor_list(
            {'Si': si_a*1.01/np.sqrt(8)})
        si_neighbors_2, si_images = silicon.get_neighbor_list(
            {'Si': si_a*1.01/np.sqrt(8)}, return_images=True)

        si_distances = np.array(
            [np.sqrt(3)/4*si_a]*4 + 
            [1/np.sqrt(2)*si_a]*12
        )

        self.assertTrue(np.allclose(
            np.sort(np.sqrt((si_vectors[0]**2).sum(-1))), si_distances))
        self.assertTrue(np.allclose(
            np.sort(np.sqrt((si_vectors[1]**2).sum(-1))), si_distances))

    def test_compile(self):

        n_elements = 118

        atomic_numbers = list(range(1, n_elements+1))
        atoms = [ATOMIC_SYMBOLS[z] for z in atomic_numbers]

        positions = np.linspace(0, 200, n_elements*3)\
            .reshape(n_elements, 3).tolist()
        unit_cell = np.eye(3)
        features = np.zeros((n_elements, 1, 5))
        outputs = np.ones((n_elements))
        graph_outputs = np.arange(5)

        periodic = (True, False, True)

        # build atom graph
        graph = AtomGraph(
            atoms=atoms, 
            positions=positions,
            unit_cell=unit_cell, 
            periodic=periodic, 
            features=features,
            outputs=outputs,
            graph_outputs=graph_outputs)

        self.assertTrue(SUPPORTED_BACKENDS)

        backend_graphs = [
            graph.compile(backend=backend)
            for backend in SUPPORTED_BACKENDS
        ]

        self.assertTrue(len(backend_graphs) > 0)

        self.assertRaises(
            ValueError, 
            lambda: graph.compile(backend='unknown_backend')
        )

        extracted_graphs = [
            bg.extract()
            for bg in backend_graphs
        ]

        for eg in extracted_graphs:
            self.assertTrue(np.all(eg.atoms == graph.atoms))
            self.assertTrue(np.allclose(eg.positions, graph.positions))
            self.assertTrue(np.allclose(eg.unit_cell, graph.unit_cell))
            self.assertTrue(np.allclose(eg.periodic, graph.periodic))
            self.assertTrue(np.allclose(eg.features, graph.features))
            self.assertTrue(np.allclose(eg.outputs, graph.outputs))
            self.assertTrue(np.allclose(eg.graph_outputs, graph.graph_outputs))
