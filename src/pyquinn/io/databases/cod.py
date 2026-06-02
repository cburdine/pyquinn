"""
Connects to the Crystallography Open Database (COD) API.

.. seealso::

    For more information on the Crystallography Open Database API, see the
    `Crystallography Open Database API page`_.

.. _Crystallography Open Database API page:
    https://wiki.crystallography.net/RESTful_API/

"""

import os
import tempfile
from contextlib import ExitStack

from requests import HTTPError, get

from pyquinn.io.data_source import Database
from pyquinn.types import AtomGraph


class COD(Database):
    """A database connector for the Crystallography Open Database (COD).

        The COD an open database of material structures
        and properties.

        https://www.crystallography.net/cod/

    .. note::
        To access the Materials Project, you will need to obtain an API
        key, which you can get by signing up for an account at
        https://next-gen.materialsproject.org/api.

        You will also need to have the Materials Project Python
        client installed, which you can do by running the command

        .. code-block:: console
            pip install mp-api


    Below is an example of basic usage:

    .. code-block:: python

            from pyquinn.io.databases import COD

            # retrieve the crystal structure of bulk silicon
            with COD() as cod:
                graph = cod.get_atomgraph_by_id('1526655')

            print(graph.atoms)
            print(graph.positions)
            print(graph.unit_cell)
    """

    _COD_SEARCH_ENDPOINT: str = "http://www.crystallography.net/cod/result"
    _COD_DATA_ENDPOINT: str = "http://www.crystallography.net/cod/"

    def __init__(self):
        """Constructs a connector to the Crystallography Open Database."""
        super().__init__()

        self._cache_dir = tempfile.TemporaryDirectory()
        self._stack = ExitStack()

    def __enter__(self):
        self._stack.__enter__()
        self._stack.enter_context(self._cache_dir)
        return self

    def __exit__(self, type, value, traceback):
        self._stack.__exit__(type, value, traceback)

    def get_atomgraph_by_id(self, id: str) -> AtomGraph | None:
        """Retrieves an AtomGraph object associated with a database ID string.

            For the COD, the ID strings are just numbers.

        Args:
            id (str): Database ID string

        Returns:
            AtomGraph | None: The AtomGraph associated with the ID, if found.
        """
        # check if file is cached, if not, fetch data from database:
        id_file = os.path.join(self._cache_dir.name, f"{id}.cif")
        if not os.path.isfile(id_file):
            # make HTTP request:
            response = get(COD._COD_DATA_ENDPOINT + f"{id}.cif")
            if response.status_code == 404:
                return None
            elif response.status_code != 200:
                raise HTTPError(
                    f"HTTP Request failed with status code " f"{response.status_code}.",
                    response=response,
                )
            else:
                self._write_file(id_file, response.text)

        # return structure as atoms:
        graph = AtomGraph.from_file(id_file)
        return graph

    def get_atomgraphs(self, formula: str) -> dict[str, AtomGraph]:
        """Retrieves a list of all AtomGraphs associated with the
           given chemical formula string.

        Args:
            formula (str): A chemical formula string.

        Returns:
            dict[str, AtomGraph]: A dictionary consisting of
                (ID, AtomGraph) pairs matching the given formula.
        """

        # retrieve structure ids:
        cod_ids = self.get_atomgraph_ids(formula)

        # retrieve all structures:
        atomgraphs = {}
        for cod_id in cod_ids:
            try:
                graph = self.get_atomgraph_by_id(cod_id)
                atomgraphs[cod_id] = graph
            except HTTPError:
                continue

        return atomgraphs

    def get_atomgraph_ids(self, formula: str) -> list[str]:
        """Retrieves the IDs associated with all AtomGraphs matching
        the given chemical formula string.

        Args:
            formula (str): A chemical formula string.

        Returns:
            list[str]: A list of COD IDs.
        """

        cod_query_params = {"formula": formula}
        results = self.search(cod_query_params).keys()
        return [str(result) for result in results]

    def get_atomgraphs_with_elements(self, elements: list[str] | str) -> dict[str, AtomGraph]:
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

        Returns:
            dict[str, AtomGraph]: A dictionary consisting of (ID, AtomGraph)
                pairs containing the given elements.
        """

        # retrieve structure ids
        cod_ids = self.get_atomgraph_ids_with_elements(elements)

        # retrieve all structures
        atomgraphs = {}
        for cod_id in cod_ids:
            try:
                graph = self.get_atomgraph_by_id(cod_id)
                atomgraphs[cod_id] = graph
            except HTTPError:
                continue

        return atomgraphs

    def get_atomgraph_ids_with_elements(self, elements: list[str] | str) -> list[str]:
        """Retrieves the IDs associated with all AtomGraphs containing
        either some or all of the supplied list of elements.

        Args:
            elements (list[str] | str): A list of element symbols
                (either in list form or in a string separated by commas)

        Returns:
            list[str]: A list of COD IDs associated with
                Atomgraphs containing the given elements.
        """

        # parse list of elements
        elements = self._parse_element_list(elements)

        # validate and trim element list (COD only supports 8 elements):
        if not elements:
            raise Exception("Elements list cannot be empty")
        elif len(elements) > 8:
            raise Exception("COD only supports queries with up to 8 elements " f"(got: {len(elements)}).")

        params = {f"el{i + 1}": sym for i, sym in enumerate(elements)}
        params["strictmax"] = str(len(elements))

        # perform search
        results = self.search(params)

        # return results
        return [str(cod_id) for cod_id in results.keys()]

    def search_by_id(self, id: str) -> dict | None:
        """Retrieves a COD data entry associatedRuff
        with a given database ID. The database entry is in a Python
        dictionary format, consisting of key-value pairs.

        Args:
            id (str): A COD ID string

        Returns:
            dict | None: A COD data entry associated with the
                given ID, if found.
        """
        id_params = {"id": id}
        results = self.search(id_params).values()
        return results[0] if results else None

    def search(self, params: dict[str, str]) -> dict[str, dict]:
        """Performs a general COD query with the given
        parameters. The format of the database query parameters corresponds
        with the keyword arguments of the :meth:`MPRester.summary,search()`
        function.

        .. seealso::

            For more information on the COD API, see the
            `Crystallography Open Database API page`_.

        .. _Crystallography Open Database API page:
            https://wiki.crystallography.net/RESTful_API/

        Args:
            params (dict[str, str]): A dictionary of query parameters.

        Returns:
            dict[str, dict]: A dictionary of (ID, data entry) pairs, where
                each data entry is formatted as a dictionary.
        """
        if not params:
            raise Exception("Parameter dict cannot be empty.")

        fmt_params = {"format": "json"}
        response = get(COD._COD_SEARCH_ENDPOINT, params=(params | fmt_params))
        if response.status_code == 404:
            return dict()
        elif response.status_code != 200:
            raise Exception(f"HTTP Request failed with status code {response.status_code}.")
        else:
            results = response.json()
            return {res["file"]: res for res in results}
