import unittest
from tempfile import TemporaryDirectory

import numpy as np

from pyquinn.types.atom_graph import AtomGraph
from pyquinn.types.dataset import SUPPORTED_DATASET_BACKENDS, AtomGraphDataset


class AtomGraphDatasetTest(unittest.TestCase):
    def test_atom_graph_dataset(self):

        si_atoms = ["Si", "Si"]
        si_frac_positions = np.array([[0.0, 0.0, 0.0], [1 / 4, 1 / 4, 1 / 4]])  # noqa  # noqa
        si_frac_unit_cell = np.array(
            [
                [0.0, 1 / 2, 1 / 2],  # noqa
                [1 / 2, 0.0, 1 / 2],  # noqa
                [1 / 2, 1 / 2, 0.0],  # noqa
            ]
        )

        n_graphs = 5
        si_graphs = [
            AtomGraph(
                atoms=si_atoms,
                positions=si_frac_positions * alat,
                unit_cell=si_frac_unit_cell * alat,
                periodic=True,
            )
            for alat in np.linspace(1.0, 2.0, n_graphs)
        ]

        dataset = AtomGraphDataset(graphs=si_graphs)

        # check dataset size
        self.assertEqual(len(dataset), n_graphs)

        # check dataset atom graphs
        for i, (_name, graph) in enumerate(dataset.items()):
            self.assertTrue(np.all(graph.atoms == si_graphs[i].atoms))
            self.assertTrue(np.all(graph.positions == si_graphs[i].positions))

    def test_atom_graph_dataset_compile(self):

        si_atoms = ["Si", "Si"]
        si_frac_positions = np.array([[0.0, 0.0, 0.0], [1 / 4, 1 / 4, 1 / 4]])  # noqa  # noqa
        si_frac_unit_cell = np.array(
            [
                [0.0, 1 / 2, 1 / 2],  # noqa
                [1 / 2, 0.0, 1 / 2],  # noqa
                [1 / 2, 1 / 2, 0.0],  # noqa
            ]
        )

        n_graphs = 5
        si_graphs = [
            AtomGraph(
                atoms=si_atoms,
                positions=si_frac_positions * alat,
                unit_cell=si_frac_unit_cell * alat,
                periodic=True,
            )
            for alat in np.linspace(1.0, 2.0, n_graphs)
        ]

        dataset = AtomGraphDataset(graphs=si_graphs)

        for backend in SUPPORTED_DATASET_BACKENDS:
            with TemporaryDirectory() as temp_dir:
                for cache_dir in (temp_dir, None):
                    compiled_dataset = dataset.compile(backend=backend, cache_dir=cache_dir, verbosity=0)

                    self.assertEqual(len(compiled_dataset.names), len(dataset.names))
                    self.assertEqual(len(compiled_dataset.graphs), len(dataset.graphs))

                    for i, (name, graph) in enumerate(
                        zip(compiled_dataset.names, compiled_dataset.graphs, strict=False)
                    ):
                        # check that names agree:
                        dataset_graph_name = dataset.names[i]
                        self.assertEqual(name, dataset_graph_name)

                        if isinstance(graph, str):
                            graph = AtomGraph._backends[backend].load(graph)
                        extracted_graph = graph.extract()
                        dataset_graph = dataset[i]

                        self.assertTrue(np.all(extracted_graph.atoms == dataset_graph.atoms))
                        self.assertTrue(
                            np.allclose(
                                extracted_graph.positions,
                                dataset_graph.positions,
                            )
                        )
