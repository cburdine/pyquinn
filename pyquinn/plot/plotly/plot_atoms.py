from pyquinn.atoms.atom_graph import AtomGraph, BackendAtomGraph
from pyquinn.atoms.atom_data import ATOM_COLORS, ATOMIC_NUMBERS, ATOMIC_COVALENT_RADII
import numpy as np

import plotly.express as px
import plotly.graph_objects as go


def _make_sphere(xyz, r, resolution=20):
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = r * np.cos(u)*np.sin(v) + xyz[0]
    Y = r * np.sin(u)*np.sin(v) + xyz[1]
    Z = r * np.cos(v) + xyz[2]
    return (X, Y, Z)


def _make_unitcell(unit_cell, periodic=False, origin=np.zeros(3)):

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


def plot_atoms(atoms: AtomGraph | BackendAtomGraph, show: bool = True):

    atoms = atoms.extract()
    data = []
    
    # render atoms:
    for i, symbol in enumerate(atoms.atoms):
        z = ATOMIC_NUMBERS[symbol] \
            if symbol in ATOMIC_NUMBERS else 0

        # get color and radius of atom
        rgb = ATOM_COLORS[z]
        color = f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'
        radius = ATOMIC_COVALENT_RADII[z]*0.6

        # make sphere
        xyz = atoms.positions[i]
        (X, Y, Z) = _make_sphere(xyz, radius)

        # asdd sphere surface
        data.append(go.Surface(
            x=X, y=Y, z=Z, 
            colorscale=[[0, color], [1, color]]))

    # render unit cell:
    lx, ly, lz = _make_unitcell(atoms.unit_cell, atoms.periodic)

    layout = go.Layout(
        scene=dict(aspectmode='data')
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(
        lighting=dict(
            ambient=0.8,
            diffuse=0.9,
            roughness=0.0,
            fresnel=4.5),
        showscale=False)

    fig.add_trace(go.Scatter3d(
        x=lx, y=ly, z=lz, mode='lines', 
        line=dict(color='black', width=3)
    ))
    
    if show:
        fig.show()
    
    return fig
