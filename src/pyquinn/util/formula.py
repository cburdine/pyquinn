from collections import Counter

_FORMULA_FORMATTERS: dict


def get_formula(atoms: list[str], format: str = "hill") -> str:
    """Returns a chemical formula from a list of atoms using the given formula

    Args:
        atoms (list[str]): List of atomic symbols.
        format (str, optional): Format for the chemical formula.
            Defaults to 'hill'.

    Raises:
        ValueError: If format is invalid.

    Returns:
        str: The chemical formula string
    """

    if format not in FORMULA_FORMATS:
        raise ValueError(
            'Invalid format "{format}" ; ' f"expected one of: {FORMULA_FORMATS}"
        )

    return _FORMULA_FORMATTERS[format](atoms)


def get_hill_formula(atoms: list[str]) -> str:
    """Returns a chemical formula from a list of atoms using the Hill format.

    Args:
        atoms (list[str]): List of atomic symbols.

    Returns:
        str: The chemical formula string.
    """

    counts = Counter(atoms)

    # Determine ordering
    elements = []
    if "C" in counts:
        elements.append("C")
        if "H" in counts:
            elements.append("H")
        others = sorted(e for e in counts if e not in ("C", "H"))
        elements.extend(others)
    else:
        elements = sorted(counts.keys())

    # Construct formula
    formula = ""
    for elem in elements:
        count = counts[elem]
        formula += f"{elem}{count if count > 1 else ''}"
    return formula


_FORMULA_FORMATTERS = {"hill": get_hill_formula}

FORMULA_FORMATS = list(_FORMULA_FORMATTERS.keys())
"""Supported chemical formula formats"""
