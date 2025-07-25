import unittest
from tempfile import TemporaryDirectory

from pyquinn.types.bandstructure_dataset import (
    BandStructureDataset,
    SUPPORTED_BANDSTRUCTURE_DATASET_BACKENDS
)
from pyquinn.types.atom_graph import AtomGraph
from pyquinn.types.bandstructure import BandStructure

import numpy as np


class AtomGraphDatasetTest(unittest.TestCase):

    def test_bandstructure_dataset(self):
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

        n_graphs = 10
        si_graphs = [
            AtomGraph(
                atoms=si_atoms, 
                positions=si_frac_positions*alat,
                unit_cell=si_frac_unit_cell*alat,
                periodic=True)
            for alat in np.linspace(1.0, 2.0, n_graphs)
        ]

        si_band_structures = [
            BandStructure(
                energies=np.full((100, 10), E),
                k_pts=np.linspace(0, 0.5, 300).reshape(100, 3),
                projections=np.full((100, 10), 0.1),
                chemical_potential=E)
            for E in np.linspace(10.0, 40.0, n_graphs)
        ]

        dataset = BandStructureDataset(
            graphs=si_graphs,
            band_structures=si_band_structures
        )

        # check dataset size
        self.assertEqual(len(dataset), n_graphs)

        # check dataset items
        for i, (name, (graph, bands)) in enumerate(dataset.items()):
            self.assertTrue(np.all(graph.atoms == si_graphs[i].atoms))
            self.assertTrue(np.all(graph.positions == si_graphs[i].positions))
            self.assertTrue(np.all(bands.energies == si_band_structures[i].energies)) # noqa
            self.assertTrue(np.all(bands.k_pts == si_band_structures[i].k_pts)) # noqa
            self.assertTrue(
                np.all(bands.projections == si_band_structures[i].projections)) # noqa
            self.assertTrue(
                np.all(bands.chemical_potential == si_band_structures[i].chemical_potential)) # noqa

    def test_bandstructure_dataset_compile(self):

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

        si_bands = BandStructure(
            energies=np.ones((100, 10)),
            k_pts=np.linspace(0, 0.5, 300).reshape(100, 3))

        si_band_structures = [si_bands]*len(si_graphs)

        dataset = BandStructureDataset(
            graphs=si_graphs,
            band_structures=si_band_structures
        )

        for backend in SUPPORTED_BANDSTRUCTURE_DATASET_BACKENDS:
            with TemporaryDirectory() as temp_dir:
                for cache_dir in (temp_dir, None):
                    compiled_dataset = dataset.compile(
                        backend=backend, 
                        cache_dir=cache_dir)

                    self.assertEqual(
                        len(compiled_dataset.names), len(dataset.names))
                    self.assertEqual(
                        len(compiled_dataset.graphs), len(dataset.graphs))
                    self.assertEqual(
                        len(compiled_dataset.band_structures), 
                        len(dataset.band_structures))

                    for i, (name, graph, bands) in enumerate(zip(
                            compiled_dataset.names, 
                            compiled_dataset.graphs,
                            compiled_dataset.band_structures)):

                        # check that names agree:
                        dataset_graph_name = dataset.names[i]
                        self.assertEqual(name, dataset_graph_name)

                        dataset_graph, dataset_bands = dataset[i]

                        # check that the AtomGraphs agree
                        if isinstance(graph, str):
                            graph = AtomGraph._backends[backend].load(graph)

                        extracted_graph = graph.extract()
                        self.assertTrue(np.all(
                            extracted_graph.atoms == dataset_graph.atoms))
                        self.assertTrue(np.allclose(
                            extracted_graph.positions, dataset_graph.positions)) # noqa

                        # check that the BandStructures agree
                        if isinstance(bands, str):
                            bands = BandStructure._backends[backend].load(bands) # noqa
                        extracted_bands = bands.extract()
                        self.assertTrue(np.allclose(
                            extracted_bands.energies, dataset_bands.energies))

                        self.assertTrue(np.allclose(
                            extracted_bands.k_pts, dataset_bands.k_pts))

                        self.assertTrue(np.allclose(
                            extracted_bands.projections,dataset_bands.projections)) # noqa

