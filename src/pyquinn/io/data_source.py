"""Provides the abstract interface for data sources."""

import re
from abc import ABC, abstractmethod
from contextlib import ExitStack

from pyquinn.types import AtomGraph
from pyquinn.util.data import ATOMIC_NUMBERS


class DataSource(ABC):
    """Abstract base class for a data input and output source."""

    def __init__(self):
        self._exit_stack = ExitStack()

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __exit__(self):
        raise NotImplementedError()


class Database(DataSource, ABC):
    """Abstract base class for an atomic structure database connnector.

    All database connectors that extend this base class are
    intended to be used via Python's enter/exit context managers.
    This means that they should be created using the following syntax
    pattern:

    .. code-block:: python

            from pyquinn.io.databases import MyCustomDatabase

            # enter database context
            with MyCustomDatabase(..) as db:

                # retrieve data and interact with database inside context
                graph = db.get_atomgraph_by_id('<atomgraph_id>')


            # once the context is exited, the database connection is closed
            print(graph.atoms)

    """

    def _parse_element_list(self, elements: str | list[str]) -> list[str]:
        """Parses and validates a list of element symbols"""
        if isinstance(elements, str):
            elements = re.split("[ ;,-]", elements)

        for elem in elements:
            if not isinstance(elem, str):
                raise ValueError(
                    f"Expected element symbol of type str (got {type(elem)})."
                )
            if len(elem) <= 0 or elem not in ATOMIC_NUMBERS:
                raise ValueError(
                    f"Expected valid element symbol (got: {elem})."
                )

        return elements

    def _write_file(self, filename: str, text: str):
        """Writes a string to a file."""
        with open(filename, "w") as f:
            f.write(text)

    @abstractmethod
    def get_atomgraph_by_id(self, id: str) -> AtomGraph | None:
        """Retrieves an AtomGraph object associated with a database ID string.

            e.g., for the Materials Project database, the ID strings
            are of the form "mp-XXXXXXX"

        Args:
            id (str): Database ID string

        Returns:
            AtomGraph | None: The AtomGraph associated with the ID, if found.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_atomgraphs(self, formula: str) -> dict[str, AtomGraph]:
        """Retrieves a list of all AtomGraphs associated with the
           given chemical formula string.

        Args:
            formula (str): The chemical formula string
                (formatted to database specifications)

        Returns:
            dict[str, AtomGraph]: A dictionary consisting of (ID, AtomGraph) pairs
                matching the given formula.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_atomgraph_ids(self, formula: str) -> list[str]:
        """Retrieves the IDs associated with all AtomGraphs matching
        the given chemical formula string.

        Args:
            formula (str): The chemical formula string
                (formatted to database specifications)

        Returns:
            list[str]: A list of database IDs.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_atomgraphs_with_elements(
        self, elements: list[str] | str, contains_all: bool = True
    ) -> dict[str, AtomGraph]:
        """Retrieves a list of all atomgraphs containing either some or
        all of the supplied list of elements.

        .. warning::
            Results returned from this query can be quite large;
            it is recommended to instead use
            `get_atomgraph_ids_with_elements()`
            to retrieve a list of IDs first.

        Args:
            elements (list[str] | str): A list of element symbols
                (either in list form or in a string separated by commas)
            contains_all (bool, optional): If true, requires matches to
                contain all specified elements. Defaults to True.

        Returns:
            dict[str, AtomGraph]: A dictionary consisting of (ID, AtomGraph) pairs
                containing the given elements.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_atomgraph_ids_with_elements(
        self, elements: list[str] | str, contains_all: bool = True
    ) -> list[str]:
        """Retrieves the IDs associated with all AtomGraphs containing
        either some or all of the supplied list of elements.

        Args:
            elements (list[str] | str): A list of element symbols
                (either in list form or in a string separated by commas)
            contains_all (bool, optional): If true, requires matches to
                contain all specified elements. Defaults to True.

        Returns:
            list[str]: A list of IDs associated with Atomgraphs
                containing the given elements.
        """
        raise NotImplementedError()

    @abstractmethod
    def search_by_id(self, id: str) -> dict | None:
        """Retrieves a database data entry associated with a given
        database ID.

        Args:
            id (str): A database ID string

        Returns:
            dict | None: A database data entry associated with the
                given ID, if found.
        """
        raise NotImplementedError()

    @abstractmethod
    def search(self, params: dict[str, str]) -> dict[str, dict]:
        """Performs a general database query with the given parameters.
        The format of database query parameters is specific to each database's
        API specification.

        Args:
            params (dict[str, str]): A dictionary of query parameters.

        Returns:
            dict[str, dict]: A dictionary of (ID, data entry) pairs, where
                each data entry is formatted as a dictionary.
        """
        raise NotImplementedError()

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __exit__(self, type, value, traceback):
        raise NotImplementedError()
