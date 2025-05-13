import os
import tempfile
from contextlib import ExitStack
import numpy as np

from pyquinn.io.data_source import Database
from pyquinn.atoms import AtomGraph, BandStructure


class MaterialsProject(Database):
    """A database connector for the Materials Project database.

        The Materials Project is an open database of material structures
        and properties.

        :ref:`https://materialsproject.org/`

    .. info::
        To access the Materials Project, you will need to obtain an API
        key, which you can get by signing up for an account at
        :ref:`https://next-gen.materialsproject.org/api`.

        You will also need to have the Materials Project Python
        client installed, which you can do by running the command

        ```
        $ pip install mp-api
        ```

    Below is an example of basic usage:

    .. code-block:: python

            from pyquinn.io.databases import MaterialsProject

            # retrieve the crystal structure of bulk silicon
            with MaterialsProject(mp_api_key='<your_api_key_here>') as mp:
                graph = mp.get_atomgraph_by_id('mp-149')

            print(graph.atoms)
            print(graph.positions)
            print(graph.unit_cell)
    """

    _thermodynamic_fields = [
        'energy_above_hull',
        'formation_energy_per_atom',
        'energy_per_atom'
    ]

    def __init__(self, mp_api_key: str | None = None):
        """Constructs a connector to the Materials Project Database

        Args:
            mp_api_key (str | None, optional): Materials Project API key. 
                If not supplied manually, the string stored in the 
                environment variable `$MP_API_KEY` is
                used. Defaults to None.

        Raises:
            ImportError: If the `mp_api` client is not installed.
        """

        super().__init__()

        try:
            from mp_api.client import MPRester
        except ImportError:
            raise ImportError(
                'The "mp_api" client package must be installed to access the '
                'Materials Project database. Try running: pip install mp-api')

        if not mp_api_key:
            if len(os.environ['MP_API_KEY']):
                mp_api_key = os.environ['MP_API_KEY']
            else:
                raise Exception('Materials Project API key must be given or '
                                'set in the environment variable $MP_API_KEY')

        self._cache_dir = tempfile.TemporaryDirectory()
        self.mp_connector = MPRester(mp_api_key)
        self._stack = ExitStack()

    def __enter__(self):
        self._stack.__enter__()
        self._stack.enter_context(self._cache_dir)
        self._stack.enter_context(self.mp_connector)
        return self

    def __exit__(self, type, value, traceback):
        self._stack.__exit__(type, value, traceback)

    def get_atomgraph_by_id(self, id: str) -> AtomGraph | None:
        """Retrieves an AtomGraph object associated with a database ID string.

            For the Materials Project database, the ID strings 
            are of the form "mp-XXXXXXX"

        Args:
            id (str): Database ID string

        Returns:
            AtomGraph | None: The AtomGraph associated with the ID, if found.
        """
        
        id_file = os.path.join(self._cache_dir.name, f'{id}.cif')

        # check if file is cached, if not, fetch data from database:
        if not os.path.isfile(id_file):
            structure = self.mp_connector.get_structure_by_material_id(id)
            structure.to(id_file, fmt='cif')

        # return structure as atoms:
        graph = AtomGraph.from_file(id_file)
        return graph

    def get_atomgraphs(
            self, 
            formula: str, 
            max_ehull: float | None = None) -> dict[str, AtomGraph]:
        """Retrieves a list of all AtomGraphs associated with the 
           given chemical formula string.

        Args:
            formula (str): A chemical formula string.

        Returns:
            dict[str, AtomGraph]: A dictionary consisting of (ID, AtomGraph) pairs
                matching the given formula.
        """

        docs = self.mp_connector.summary.search(
            formula=formula, 
            energy_above_hull=(0.0, max_ehull),
            fields=['material_id', 'structure'])

        atomgraphs = {}
        for doc in docs:
            # cache material structure:
            id_file = os.path.join(
                self._cache_dir.name, f'{doc.material_id}.cif')
            doc.structure.to(id_file, fmt='cif')

            # parse structure as atoms:
            graph = AtomGraph.from_file(id_file)
            atomgraphs[doc.material_id] = graph

        return atomgraphs

    def get_atomgraph_ids(
            self, formula: str, 
            max_ehull: float | None = None) -> list[str]:
        """Retrieves the IDs associated with all AtomGraphs matching
        the given chemical formula string.

        Args:
            formula (str): A chemical formula string.
        
        Returns:
            list[str]: A list of Materials Project IDs.
        """
        
        docs = self.mp_connector.summary.search(
            formula=formula, 
            energy_above_hull=(0.0, max_ehull),
            fields=['material_id'])

        return [str(doc.material_id) for doc in docs]

    def get_atomgraphs_with_elements(
            self, 
            elements: list[str] | str,
            max_ehull: float | None = None,
            contains_all: bool = True) -> dict[str, AtomGraph]:
        """Retrieves a list of all AtomGraphs containing either some or
        all of the supplied list of elements.

        .. warning::
            Results returned from this query can be quite large;
            it is recommended to instead use 
            `get_atomgraph_ids_with_elements()`
            to retrieve a list of IDs first.

        Args:
            elements (list[str] | str): A list of element symbols
                (either in list form or in a string separated by commas)
            max_ehull (float, optional): The maximum allowed energy above hull.
                If None, no limit is imposed. Defaults to None.
            contains_all (bool, optional): If true, requires matches to 
                contain all specified elements. Defaults to True.

        Returns:
            dict[str, AtomGraph]: A dictionary consisting of (ID, AtomGraph) 
                pairs containing the given elements.
        """
        
        # parse list of elements
        elements = self._parse_element_list(elements)

        # search for materials containing elements
        docs = self.mp_connector.summary.search(
            elements=elements,
            energy_above_hull=(0.0, max_ehull),
            num_elements=len(elements) if contains_all else None,
            fields=['material_id', 'structure'])

        # cache and parse structures
        atomgraphs = {}
        for doc in docs:
            id_file = os.path.join(
                self._cache_dir.name, f'{doc.material_id}.cif')
            doc.structure.to(id_file, fmt='cif')
            graph = AtomGraph.from_file(id_file)
            atomgraphs[str(doc.material_id)] = graph

        return atomgraphs

    def get_atomgraph_ids_with_elements(
            self, 
            elements: list[str],
            max_ehull: float | None = None,
            contains_all: bool = True) -> list[str]:
        """Retrieves the IDs associated with all AtomGraphs containing 
        either some or all of the supplied list of elements.

        Args:
            elements (list[str] | str): A list of element symbols
                (either in list form or in a string separated by commas)
            max_ehull (float, optional): The maximum allowed energy above hull.
                If None, no limit is imposed. Defaults to None.
            contains_all (bool, optional): If true, requires matches to 
                contain all specified elements. Defaults to True.

        Returns:
            list[str]: A list of Materials Project IDs associated with
                Atomgraphs containing the given elements.
        """
        
        # parse list of elements
        elements = self._parse_element_list(elements)

        # search for materials containing elements
        docs = self.mp_connector.summary.search(
            elements=elements, 
            energy_above_hull=(0.0, max_ehull),
            num_elements=len(elements) if contains_all else None,
            fields=['material_id'])

        return [str(doc.material_id) for doc in docs]

    def search_by_id(self, id: str) -> dict | None:
        """Retrieves a Materials Project data entry associated 
        with a given database ID. The database entry is in a Python
        dictionary format, consisting of key-value pairs.

        Args:
            id (str): A Materials Project ID string

        Returns:
            dict | None: A Materials Project data entry associated with the
                given ID, if found.
        """
        docs = self.mp_connector.summary.search(material_ids=[id])
        return dict(docs[0]) if docs else None

    def search(self, params: dict) -> dict[str, dict]:
        """Performs a general Materials Project query with the given
        parameters. The format of the database query parameters corresponds 
        with the keyword arguments of the :meth:`MPRester.summary,search()`
        function.
        
        Args:
            params (dict[str, str]): A dictionary of query parameters.

        Returns:
            dict[str, dict]: A dictionary of (ID, data entry) pairs, where
                each data entry is formatted as a dictionary. 
        """

        # ensure mp ids are retured in query:
        params = params.copy()
        if 'fields' in params and ('material_id' not in params['fields']):
            params['fields'] = list(params['fields'])
            params['fields'].append('material_id')
        
        # perform query and return raw results:
        docs = self.mp_connector.summary.search(**params)
        results = {
            doc.material_id: dict(doc) for doc in docs
            for doc in docs
        }

        for doc in docs:
            # cache material structures (if present):
            if 'atomgraph' in doc:
                id_file = os.path.join(
                    self._cache_dir.name, f'{doc.material_id}.cif')
                doc.structure.to(id_file, fmt='cif')

                # parse structure as atoms:
                atoms = AtomGraph.from_file(id_file)
                results[doc.material_id]['atomgraph'] = atoms
        
        return results

    def get_thermodynamic_data_by_id(self, id: str) -> dict:
        """Retrieves the thermodynamic data associated with a 
        given database ID string.

        Args:
            id (str): A Materials Project ID string.

        Returns:
            dict: A dictionary of key-value pairs with the 
                following fields:
                - `energy_above_hull`: The energy above hull of this material.
                - `formation_energy_per_atom`,: The formation energy per atom.
                - `energy_per_atom`: The total energy per atom.
        """

        docs = self.mp_connector.summary.search(
            material_ids=id,
            fields=['material_id']+self._thermodynamic_fields
        )

        doc_dict = dict(docs[0])
        result = {
            field: doc_dict[field] for field in self._thermodynamic_fields
        }

        return result

    def get_thermodynamic_data_with_elements(
            self, 
            elements: list[str], 
            max_ehull: float | None = None,
            contains_all: bool = True) -> dict:
        """Retrieves the thermodynamic data associated with all materials
        containing either some or all of the supplied list of elements.

        Args:
            id (str): A Materials Project ID string.

        Returns:
            dict: A dictionary of key-value pairs with the 
                following fields:
                - `energy_above_hull`: The energy above hull of this material.
                - `formation_energy_per_atom`,: The formation energy per atom.
                - `energy_per_atom`: The total energy per atom.
        """
        
        elements = self._parse_element_list(elements)

        docs = self.mp_connector.summary.search(
            elements=elements, 
            energy_above_hull=(0.0, max_ehull),
            num_elements=len(elements) if contains_all else None,
            fields=['material_id'] + self._thermodynamic_fields)

        results = {}
        for doc in docs:
            doc_dict = dict(doc)
            results[str(doc.material_id)] = {
                field: doc_dict[field] for field in self._thermodynamic_fields
            }

        return results

    def get_bandstructure_by_id(self, id: str) -> BandStructure | None:
        """Returns a precomputed BandStructure object associated with 
        the given database ID, if found.


        Args:
            id (str): A Materials Project ID string.

        Returns:
            BandStructure | None: A BandStructure object.
        """

        bandstructure = self.mp_connector.get_bandstructure_by_material_id(id)

        # compute clipped bands and projections:
        bands_list = []
        projs_list = []

        k_pts = np.array([
            x.cart_coords for x in bandstructure.kpoints
        ])
    
        for k in bandstructure.bands:
            bands = np.array(bandstructure.bands[k])

            bands_list.append(
                np.moveaxis(bands, (0, 1), (1, 0)))

            if k not in bandstructure.projections:
                projs_list = None
            elif projs_list is not None:
                projs = np.array(bandstructure.projections[k])
                projs_list.append(
                    np.moveaxis(projs, (0, 1, 2, 3), (1, 0, 3, 2)))
        
        energies = np.stack(bands_list, axis=-1).squeeze(-1)
        
        projections = np.stack(projs_list, axis=-1).squeeze(-1) \
            if projs_list is not None else None

        bandstructure = BandStructure(
            energies=energies,
            k_pts=k_pts,
            projections=projections,
            chemical_potential=float(bandstructure.efermi)
        )

        return bandstructure
