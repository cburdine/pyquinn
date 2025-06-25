import os
from requests import get, HTTPError
from contextlib import ExitStack
import tempfile

from pyquinn.io.data_source import Database
from pyquinn.types import AtomGraph


class COD(Database):

    _COD_SEARCH_ENDPOINT: str = 'http://www.crystallography.net/cod/result'
    _COD_DATA_ENDPOINT: str = 'http://www.crystallography.net/cod/'

    def __init__(self, ):
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

        # check if file is cached, if not, fetch data from database:
        id_file = os.path.join(self._cache_dir.name, f'{id}.cif')
        if not os.path.isfile(id_file):
            
            # make HTTP request:
            response = get(COD._COD_DATA_ENDPOINT+f'{id}.cif')
            if response.status_code == 404:
                return None
            elif response.status_code != 200:
                raise HTTPError(
                    f'HTTP Request failed with status code '
                    f'{response.status_code}.',
                    response=response)
            else:
                self._write_file(id_file, response.text)
        
        # return structure as atoms:
        graph = AtomGraph.from_file(id_file)
        return graph

    def get_atomgraphs(self, formula: str) -> dict[str, AtomGraph]:
        
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
        cod_query_params = {'formula': formula}
        results = self.search(cod_query_params).keys()
        return [str(result) for result in results]        

    def get_atomgraphs_with_elements(
            self, 
            elements: list[str] | str) -> dict[str, AtomGraph]:
        
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

    def get_atomgraph_ids_with_elements(
            self, 
            elements: list[str] | str) -> list[str]:

        # parse list of elements
        elements = self._parse_element_list(elements)

        # validate and trim element list (COD only supports 8 elements):
        if not elements:
            raise Exception('Elements list cannot be empty')
        elif len(elements) > 8:
            raise Exception('COD only supports queries with up to 8 elements '
                            f'(got: {len(elements)}).')
        
        params = {f'el{i+1}': sym for i, sym in enumerate(elements)}
        params['strictmax'] = str(len(elements))

        # perform search
        results = self.search(params)

        # return results
        return [str(cod_id) for cod_id in results.keys()]

    def search_by_id(self, id: str) -> dict | None:
        id_params = {'id': id}
        results = self.search(id_params).values()
        return results[0] if results else None

    def search(self, params: dict[str, str]) -> dict[str, dict]:
        
        if not params:
            raise Exception('Parameter dict cannot be empty.')

        fmt_params = {'format': 'json'}
        response = get(COD._COD_SEARCH_ENDPOINT, params=(params | fmt_params))
        if response.status_code == 404:
            return dict()
        elif response.status_code != 200:
            raise Exception(
                f'HTTP Request failed with status code '
                f'{response.status_code}.')
        else:
            results = response.json()
            return {res['file']: res for res in results}
