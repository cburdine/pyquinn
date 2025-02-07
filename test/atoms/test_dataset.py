import unittest
from tempfile import TemporaryDirectory

from pyquinn.atoms.dataset import AtomGraphDataset, SUPPORTED_DATASET_BACKENDS
from pyquinn.atoms.atom_graph import AtomGraph
from pyquinn.atoms.data import ATOMIC_SYMBOLS

import numpy as np


class AtomGraphDatasetTest(unittest.TestCase):

    def test_atom_graph_dataset(self):

        si_atoms = ['Si', 'Si']
        si_frac_positions = np.array([
            [0.,   0.,   0.    ], # noqa
            [1/4,  1/4,  1/4   ]  # noqa
        ])
        si_frac_unit_cell = np.array([
            [0. ,   1/2,   1/2  ], # noqa
            [1/2,   0. ,   1/2  ], # noqa
            [1/2,   1/2,   0.   ]  # noqa
        ])

        n_graphs = 5
        si_graphs = [
            AtomGraph(
                atoms=si_atoms, 
                positions=si_frac_positions*alat,
                unit_cell=si_frac_unit_cell*alat,
                periodic=True)
            for alat in np.linspace(1.0, 2.0, n_graphs)
        ]

        dataset = AtomGraphDataset(graphs=si_graphs)

        # check dataset size
        self.assertEqual(len(dataset), n_graphs)

        # check dataset atom graphs
        for i, (name, graph) in enumerate(dataset.items()):
            self.assertTrue(np.all(graph.atoms == si_graphs[i].atoms))
            self.assertTrue(np.all(graph.positions == si_graphs[i].positions))

    def test_atom_graph_dataset_compile(self):

        si_atoms = ['Si', 'Si']
        si_frac_positions = np.array([
            [0.,   0.,   0.    ], # noqa
            [1/4,  1/4,  1/4   ]  # noqa
        ])
        si_frac_unit_cell = np.array([
            [0. ,   1/2,   1/2  ], # noqa
            [1/2,   0. ,   1/2  ], # noqa
            [1/2,   1/2,   0.   ]  # noqa
        ])

        n_graphs = 5
        si_graphs = [
            AtomGraph(
                atoms=si_atoms, 
                positions=si_frac_positions*alat,
                unit_cell=si_frac_unit_cell*alat,
                periodic=True)
            for alat in np.linspace(1.0, 2.0, n_graphs)
        ]

        dataset = AtomGraphDataset(graphs=si_graphs)

        for backend in SUPPORTED_DATASET_BACKENDS:
            with TemporaryDirectory() as temp_dir:

                compiled_dataset = dataset.compile(
                    backend=backend, 
                    cache_dir=temp_dir)


                self.assertEqual(
                    len(compiled_dataset.names), len(dataset.names))
                self.assertEqual(
                    len(compiled_dataset.graphs), len(dataset.graphs))

                for i, (name, graph) in enumerate(zip(
                        compiled_dataset.names, 
                        compiled_dataset.graphs)):

                    # check that names agree:
                    dataset_graph_name = dataset.names[i]
                    self.assertEqual(name, dataset_graph_name)

                    loaded_graph = AtomGraph._backends[backend].load(graph)
                    extracted_graph = loaded_graph.extract()
                    dataset_graph = dataset[i]
                    self.assertTrue(np.all(
                        extracted_graph.atoms == dataset_graph.atoms))
                    self.assertTrue(np.all(
                        extracted_graph.positions == dataset_graph.positions))
