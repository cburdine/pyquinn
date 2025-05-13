from pyquinn.types import AtomGraph
from pyquinn.types.atom_graph import BackendAtomGraph

from pyquinn.util.data import (
    ATOM_COLORS, ATOMIC_NUMBERS, ATOMIC_COVALENT_RADII,
    ATOMIC_SYMBOLS
)
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go


def _make_sphere(xyz, r, resolution=10):
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = r * np.cos(u)*np.sin(v) + xyz[0]
    Y = r * np.sin(u)*np.sin(v) + xyz[1]
    Z = r * np.cos(v) + xyz[2]
    return (X, Y, Z)


def _make_unitcell(
        unit_cell, 
        periodic=False, 
        origin=np.zeros(3)):

    # compute unit cell vertices:
    v1, v2, v3, o = \
        unit_cell[0], unit_cell[1], unit_cell[2], origin

    vertices = np.array([
        o, o + v1, o + v2, o + v3,
        o + v1 + v2, o + v1 + v3, o + v2 + v3, o + v1 + v2 + v3
    ])

    edges = [
        (0, 1), (0, 2), (0, 3),  # Edges from origin
        (1, 4), (1, 5),  # Edges from v1
        (2, 4), (2, 6),  # Edges from v2
        (3, 5), (3, 6),  # Edges from v3
        (4, 7), (5, 7), (6, 7)  # Edges connecting the top face
    ]

    # Create a list of x, y, z coordinates for edges
    x_lines, y_lines, z_lines = [], [], []

    for edge in edges:
        for vertex in edge:
            x_lines.append(vertices[vertex][0])
            y_lines.append(vertices[vertex][1])
            z_lines.append(vertices[vertex][2])
        
        # Add None to break the line (important for disconnected segments)
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    return x_lines, y_lines, z_lines


def _make_bonds(positions, nvecs):
    
    x_lines, y_lines, z_lines = [], [], []
    for x0, dxs in zip(positions, nvecs):
        for dx in dxs:
            x_lines.extend([x0[0], x0[0] + dx[0], None])
            y_lines.extend([x0[1], x0[1] + dx[1], None])
            z_lines.extend([x0[2], x0[2] + dx[2], None])

    return x_lines, y_lines, z_lines


def _get_isosurface_mesh(mesh_xyz, mesh_values, iso_val, mesh_data):

    try:
        from skimage import measure
    except ImportError:
        raise ImportError(
            'Package "scikit-image" (skimage) must be installed to '
            'visualize complex isosurfaces. '
            'Please install with "pip install scikit-image".')

    mesh_shape, mesh_spacing = _infer_shape_and_spacing(mesh_xyz)
    V = mesh_values.reshape(mesh_shape)

    surface_xyz, surface_ijk, _, _ = \
        measure.marching_cubes(V, level=iso_val, spacing=mesh_spacing)
    
    surface_xyz += np.min(mesh_xyz, axis=0, keepdims=True)

    surface_data = griddata(
        points=mesh_xyz,
        values=mesh_data,
        xi=surface_xyz,
        method='nearest')

    return surface_xyz, surface_ijk, surface_data


def _infer_shape_and_spacing(mesh_xyz):
    x_unique, x_counts = np.unique(mesh_xyz[:, 0], return_counts=True)
    y_unique, y_counts = np.unique(mesh_xyz[:, 0], return_counts=True)
    z_unique, z_counts = np.unique(mesh_xyz[:, 0], return_counts=True)

    x_spacing = np.unique(np.diff(x_unique))
    y_spacing = np.unique(np.diff(x_unique))
    z_spacing = np.unique(np.diff(z_unique))

    axis_data = [
        ('x', x_counts, x_spacing), 
        ('y', y_counts, y_spacing), 
        ('z', z_counts, z_spacing)
    ]

    for axis, counts, spacing in axis_data:
        if len(np.unique(counts)) > 1:
            raise ValueError(
                'Argument "mesh_xyz" must be a valid cubic mesh along the '
                f'"{axis}" direction.\n'
                f'Found different grid sizes along this axis: {counts}')

        if not np.allclose(spacing, spacing[0]):
            raise ValueError(
                'Argument "mesh_xyz" must be a uniform cubic '
                f'mesh along the "{axis}" direction.\n'
                f'Found different step sizes along this axis: {spacing}')

    shape = (x_unique.shape[0], y_unique.shape[0], z_unique.shape[0])
    spacing = (x_spacing[0], y_spacing[0], z_spacing[0])

    return shape, spacing
    

def plot_atoms(
        atoms: AtomGraph | BackendAtomGraph, 
        bond_factor: float = 1.1,
        radius_factor: float = 0.5,
        hide_atoms: bool = False,
        hide_bonds: bool = False,
        hide_unit_cell: bool = False,
        fig: go.Figure | None = None,
        show: bool = True):

    atoms = atoms.extract()
    data = []
 
    if not hide_atoms:
        # render atoms:
        for i, symbol in enumerate(atoms.atoms):
            z = ATOMIC_NUMBERS[symbol] \
                if symbol in ATOMIC_NUMBERS else 0

            # get color and radius of atom
            rgb = ATOM_COLORS[z]
            color = f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'
            radius = ATOMIC_COVALENT_RADII[z]*radius_factor

            # make sphere
            xyz = atoms.positions[i]
            (X, Y, Z) = _make_sphere(xyz, radius)

            # asdd sphere surface
            data.append(go.Surface(
                x=X, y=Y, z=Z, 
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=f'{ATOMIC_SYMBOLS[z]} ({i})'))

    if not hide_bonds:
        bond_cutoffs = [
            bond_factor*ATOMIC_COVALENT_RADII[
                ATOMIC_NUMBERS[sym]
            ] if sym in ATOMIC_NUMBERS else 0.0
            for sym in atoms.atoms
        ]

        _, nvecs = atoms.get_neighbor_list(cutoffs=bond_cutoffs)
        lines = _make_bonds(atoms.positions, nvecs)
        
        data.append(go.Scatter3d(
            x=lines[0], y=lines[1], z=lines[2],
            mode='lines+markers',
            line=dict(color='grey', width=10),
            name='Bond'
        ))
    
    # render unit cell:
    if not hide_unit_cell:
        lx, ly, lz = _make_unitcell(atoms.unit_cell, atoms.periodic)
        data.append(go.Scatter3d(
            x=lx, y=ly, z=lz, mode='lines', 
            line=dict(color='black', width=3),
            name='Unit Cell'
        ))

    # plot figure
    if fig is None:
        fig = go.Figure()

    fig.add_traces(data=data)
    fig.update_traces(showlegend=False)
    fig.update_layout(scene=dict(aspectmode='data'))
    
    if show:
        fig.show()
    
    return fig


ATOMIC_ISOSURFACE_MODES = [
    'real',
    'signed',
    'complex',
]


def plot_atomic_isosurface(
        mesh_xyz: np.ndarray,
        mesh_values: np.ndarray,
        mode: str = 'real',
        atoms: AtomGraph | None = None,
        iso_value: float | None = None,
        color: str | None = None,
        alpha: float = 1.0,
        fig: go.Figure | None = None,
        show: bool = True,
        **plot_atoms_kwargs):

    if mode not in ATOMIC_ISOSURFACE_MODES:
        raise ValueError(
            f'Expected mode to be one of {ATOMIC_ISOSURFACE_MODES}; '
            f'got: "{mode}".')

    # plot figure
    if fig is None:
        fig = go.Figure()

    # generate isosurface data
    data = []

    if mode == 'signed':

        pos_mesh_values = np.maximum(0.0, np.real(mesh_values))
        neg_mesh_values = np.minimum(0.0, np.real(mesh_values))

        pos_iso_value = np.mean(pos_mesh_values) \
            if iso_value is None else np.abs(iso_value)
        neg_iso_value = np.mean(neg_mesh_values) \
            if iso_value is None else -np.abs(iso_value)
        comp_iso_value = 0.0 if iso_value is None else np.abs(iso_value)

        if np.min(neg_mesh_values) < -comp_iso_value:
            data.append(go.Isosurface(
                x=mesh_xyz[:, 0], 
                y=mesh_xyz[:, 1], 
                z=mesh_xyz[:, 2],
                value=neg_mesh_values,
                isomin=neg_iso_value,
                isomax=neg_iso_value,
                colorscale=[[0, 'red'], [1, 'red']],
                opacity=alpha,
                showscale=False
            ))
        if np.max(pos_mesh_values) > comp_iso_value:
            data.append(go.Isosurface(
                x=mesh_xyz[:, 0], 
                y=mesh_xyz[:, 1], 
                z=mesh_xyz[:, 2],
                value=pos_mesh_values,
                isomin=pos_iso_value,
                isomax=pos_iso_value,
                colorscale=[[0, 'blue'], [1, 'blue']],
                opacity=alpha,
                showscale=False
            ))
    elif mode == 'complex':
        
        iso_value = None if iso_value is None else np.abs(iso_value)
        mesh_phases = mesh_values / np.abs(mesh_values)
        surf_xyz, surf_ijk, surf_data = _get_isosurface_mesh(
            mesh_xyz=mesh_xyz,
            mesh_values=np.abs(mesh_values),
            iso_val=iso_value,
            mesh_data=mesh_phases)
        
        data.append(
            go.Mesh3d(
                x=surf_xyz[:, 0], y=surf_xyz[:, 1], z=surf_xyz[:, 2],
                i=surf_ijk[:, 0], j=surf_ijk[:, 1], k=surf_ijk[:, 2],
                intensity=(np.angle(surf_data) + np.pi)/(2*np.pi),
                colorscale='phase',
                cmin=0, cmax=1,
                opacity=alpha,
                showscale=True,
                colorbar_title="Phase (θ/2π)",
                hovertemplate="Phase: 2π * %{intensity:.2f}"
            ))
    else:
        colorscale = None if color is None \
            else [[0, color], [1, color]]

        data.append(
            go.Isosurface(
                x=mesh_xyz[:, 0], 
                y=mesh_xyz[:, 1], 
                z=mesh_xyz[:, 2],
                value=mesh_values,
                isomin=iso_value,
                isomax=iso_value,
                colorscale=colorscale,
                opacity=alpha,
                showscale=False,
            ))
        
    fig.add_traces(data=data)

    if atoms is not None:
        plot_atoms(
            atoms, fig=fig, show=show,
            **plot_atoms_kwargs)

    fig.update_layout(scene=dict(aspectmode='data'))

    return fig


