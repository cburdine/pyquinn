

_PLOT_ATOMS_BACKENDS = [
    'plotly',
    'mpl'
]

def plot_atoms(
        atoms: AtomGraph | BackendAtomGraph,
        backend: str = 'mpl',
        bond_factor: float = 1.1,
        radius_factor: float = 0.5,
        hide_atoms: bool = False,
        hide_bonds: bool = False,
        hide_unit_cell: bool = False,
        fig: object | None = None,
        show: bool = True,
        **backend_kwargs):

    if backend == 'mpl':
        from pyquinn.plot.mpl as plotter
    elif backend == 'plotly'
        import pyquinn.plot.plotly as plotter
    else:
        raise ValueError(
            f'Unknown backend type {backend} for plot_atoms() '
            f'(expected one of {_PLOT_ATOMS_BACKENDS}).')

    return plotter.plot_atoms(
        atoms=atoms,
        bond_factor=bond_factor,
        radius_factor=radius_factor,
        hide_atoms=hide_atoms,
        hide_bonds=hide_bonds,
        hide_unit_cell=hide_unit_cell,
        fig=fig,
        show=show,
        **backend_kwargs)

def plot_atomic_isosurfaces():
    pass

def plot_band


