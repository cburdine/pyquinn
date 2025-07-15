"""Contains useful chemistry constants."""
import numpy as np

ATOMIC_SYMBOLS = np.array([
    # 0
    '',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
])
"""Numpy array mapping quantum numbers to atomic symbols"""

ATOMIC_NUMBERS = {
    symbol: i+1 for i, symbol in enumerate(ATOMIC_SYMBOLS[1:])
}
"""Dictionary mapping atomic symbols to quantum numbers"""


# IUPAC version dated 28 November 2016
ATOMIC_NAMES = np.array([
    '', 'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron',
    'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon', 'Sodium',
    'Magnesium', 'Aluminium', 'Silicon', 'Phosphorus', 'Sulfur',
    'Chlorine', 'Argon', 'Potassium', 'Calcium', 'Scandium',
    'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron',
    'Cobalt', 'Nickel', 'Copper', 'Zinc', 'Gallium', 'Germanium',
    'Arsenic', 'Selenium', 'Bromine', 'Krypton', 'Rubidium',
    'Strontium', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum',
    'Technetium', 'Ruthenium', 'Rhodium', 'Palladium', 'Silver',
    'Cadmium', 'Indium', 'Tin', 'Antimony', 'Tellurium',
    'Iodine', 'Xenon', 'Caesium', 'Barium', 'Lanthanum',
    'Cerium', 'Praseodymium', 'Neodymium', 'Promethium',
    'Samarium', 'Europium', 'Gadolinium', 'Terbium',
    'Dysprosium', 'Holmium', 'Erbium', 'Thulium', 'Ytterbium',
    'Lutetium', 'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium',
    'Osmium', 'Iridium', 'Platinum', 'Gold', 'Mercury',
    'Thallium', 'Lead', 'Bismuth', 'Polonium', 'Astatine',
    'Radon', 'Francium', 'Radium', 'Actinium', 'Thorium',
    'Protactinium', 'Uranium', 'Neptunium', 'Plutonium',
    'Americium', 'Curium', 'Berkelium', 'Californium',
    'Einsteinium', 'Fermium', 'Mendelevium', 'Nobelium',
    'Lawrencium', 'Rutherfordium', 'Dubnium', 'Seaborgium',
    'Bohrium', 'Hassium', 'Meitnerium', 'Darmastadtium',
    'Roentgenium', 'Copernicium', 'Nihonium', 'Flerovium',
    'Moscovium', 'Livermorium', 'Tennessine', 'Oganesson'])

# Atomic masses are based on:
#
#   Meija, J., Coplen, T., Berglund, M., et al. (2016). Atomic weights of
#   the elements 2013 (IUPAC Technical Report). Pure and Applied Chemistry,
#   88(3), pp. 265-291. Retrieved 30 Nov. 2016,
#   from doi:10.1515/pac-2015-0305
#
# Standard atomic weights are taken from Table 1: "Standard atomic weights
# 2013", with the uncertainties ignored.
# For hydrogen, helium, boron, carbon, nitrogen, oxygen, magnesium, silicon,
# sulfur, chlorine, bromine and thallium, where the weights are given as a
# range the "conventional" weights are taken from Table 3 and the ranges are
# given in the comments.
# The mass of the most stable isotope (in Table 4) is used for elements
# where there the element has no stable isotopes (to avoid NaNs): Tc, Pm,
# Po, At, Rn, Fr, Ra, Ac, everything after Np
#
ATOMIC_MASSES = np.array([
    np.nan,  # None
    1.008,  # H [1.00784, 1.00811]
    4.002602,  # He
    6.94,  # Li [6.938, 6.997]
    9.0121831,  # Be
    10.81,  # B [10.806, 10.821]
    12.011,  # C [12.0096, 12.0116]
    14.007,  # N [14.00643, 14.00728]
    15.999,  # O [15.99903, 15.99977]
    18.998403163,  # F
    20.1797,  # Ne
    22.98976928,  # Na
    24.305,  # Mg [24.304, 24.307]
    26.9815385,  # Al
    28.085,  # Si [28.084, 28.086]
    30.973761998,  # P
    32.06,  # S [32.059, 32.076]
    35.45,  # Cl [35.446, 35.457]
    39.948,  # Ar
    39.0983,  # K
    40.078,  # Ca
    44.955908,  # Sc
    47.867,  # Ti
    50.9415,  # V
    51.9961,  # Cr
    54.938044,  # Mn
    55.845,  # Fe
    58.933194,  # Co
    58.6934,  # Ni
    63.546,  # Cu
    65.38,  # Zn
    69.723,  # Ga
    72.630,  # Ge
    74.921595,  # As
    78.971,  # Se
    79.904,  # Br [79.901, 79.907]
    83.798,  # Kr
    85.4678,  # Rb
    87.62,  # Sr
    88.90584,  # Y
    91.224,  # Zr
    92.90637,  # Nb
    95.95,  # Mo
    97.90721,  # 98Tc
    101.07,  # Ru
    102.90550,  # Rh
    106.42,  # Pd
    107.8682,  # Ag
    112.414,  # Cd
    114.818,  # In
    118.710,  # Sn
    121.760,  # Sb
    127.60,  # Te
    126.90447,  # I
    131.293,  # Xe
    132.90545196,  # Cs
    137.327,  # Ba
    138.90547,  # La
    140.116,  # Ce
    140.90766,  # Pr
    144.242,  # Nd
    144.91276,  # 145Pm
    150.36,  # Sm
    151.964,  # Eu
    157.25,  # Gd
    158.92535,  # Tb
    162.500,  # Dy
    164.93033,  # Ho
    167.259,  # Er
    168.93422,  # Tm
    173.054,  # Yb
    174.9668,  # Lu
    178.49,  # Hf
    180.94788,  # Ta
    183.84,  # W
    186.207,  # Re
    190.23,  # Os
    192.217,  # Ir
    195.084,  # Pt
    196.966569,  # Au
    200.592,  # Hg
    204.38,  # Tl [204.382, 204.385]
    207.2,  # Pb
    208.98040,  # Bi
    208.98243,  # 209Po
    209.98715,  # 210At
    222.01758,  # 222Rn
    223.01974,  # 223Fr
    226.02541,  # 226Ra
    227.02775,  # 227Ac
    232.0377,  # Th
    231.03588,  # Pa
    238.02891,  # U
    237.04817,  # 237Np
    244.06421,  # 244Pu
    243.06138,  # 243Am
    247.07035,  # 247Cm
    247.07031,  # 247Bk
    251.07959,  # 251Cf
    252.0830,  # 252Es
    257.09511,  # 257Fm
    258.09843,  # 258Md
    259.1010,  # 259No
    262.110,  # 262Lr
    267.122,  # 267Rf
    268.126,  # 268Db
    271.134,  # 271Sg
    270.133,  # 270Bh
    269.1338,  # 269Hs
    278.156,  # 278Mt
    281.165,  # 281Ds
    281.166,  # 281Rg
    285.177,  # 285Cn
    286.182,  # 286Nh
    289.190,  # 289Fl
    289.194,  # 289Mc
    293.204,  # 293Lv
    293.208,  # 293Ts
    294.214,  # 294Og
])
"""Contains the IUPAC atomic masses"""

ATOM_COLORS = np.array([
    (255, 0, 0),  # 
    (255, 255, 255),  # H
    (255, 192, 203),  # He
    (178, 34, 34),  # Li
    (255, 20, 147),  # Be
    (0, 255, 0),  # B
    (200, 200, 200),  # C
    (143, 143, 255),  # N
    (240, 0, 0),  # O
    (218, 165, 32),  # F
    (255, 20, 147),  # Ne
    (0, 0, 255),  # Na
    (34, 139, 34),  # Mg
    (128, 128, 144),  # Al
    (218, 165, 32),  # Si
    (255, 165, 0),  # P
    (255, 200, 50),  # S
    (0, 255, 0),  # Cl
    (255, 20, 147),  # Ar
    (255, 20, 147),  # K
    (128, 128, 144),  # Ca
    (255, 20, 147),  # Sc
    (128, 128, 144),  # Ti
    (255, 20, 147),  # V
    (128, 128, 144),  # Cr
    (128, 128, 144),  # Mn
    (255, 165, 0),  # Fe
    (255, 20, 147),  # Co
    (165, 42, 42),  # Ni
    (165, 42, 42),  # Cu
    (165, 42, 42),  # Zn
    (255, 20, 147),  # Ga
    (255, 20, 147),  # Ge
    (255, 20, 147),  # As
    (255, 20, 147),  # Se
    (165, 42, 42),  # Br
    (255, 20, 147),  # Kr
    (255, 20, 147),  # Rb
    (255, 20, 147),  # Sr
    (255, 20, 147),  # Y
    (255, 20, 147),  # Zr
    (255, 20, 147),  # Nb
    (255, 20, 147),  # Mo
    (255, 20, 147),  # Tc
    (255, 20, 147),  # Ru
    (255, 20, 147),  # Rh
    (255, 20, 147),  # Pd
    (128, 128, 144),  # Ag
    (255, 20, 147),  # Cd
    (255, 20, 147),  # In
    (255, 20, 147),  # Sn
    (255, 20, 147),  # Sb
    (255, 20, 147),  # Te
    (160, 32, 240),  # I
    (255, 20, 147),  # Xe
    (255, 20, 147),  # Cs
    (255, 165, 0),  # Ba
    (255, 20, 147),  # La
    (255, 20, 147),  # Ce
    (255, 20, 147),  # Pr
    (255, 20, 147),  # Nd
    (255, 20, 147),  # Pm
    (255, 20, 147),  # Sm
    (255, 20, 147),  # Eu
    (255, 20, 147),  # Gd
    (255, 20, 147),  # Tb
    (255, 20, 147),  # Dy
    (255, 20, 147),  # Ho
    (255, 20, 147),  # Er
    (255, 20, 147),  # Tm
    (255, 20, 147),  # Yb
    (255, 20, 147),  # Lu
    (255, 20, 147),  # Hf
    (255, 20, 147),  # Ta
    (255, 20, 147),  # W
    (255, 20, 147),  # Re
    (255, 20, 147),  # Os
    (255, 20, 147),  # Ir
    (255, 20, 147),  # Pt
    (218, 165, 32),  # Au
    (255, 20, 147),  # Hg
    (255, 20, 147),  # Tl
    (255, 20, 147),  # Pb
    (255, 20, 147),  # Bi
    (255, 20, 147),  # Po
    (255, 20, 147),  # At
    (255, 255, 255),  # Rn
    (255, 255, 255),  # Fr
    (255, 255, 255),  # Ra
    (255, 255, 255),  # Ac
    (255, 20, 147),  # Th
    (255, 255, 255),  # Pa
    (255, 20, 147),  # U
    (255, 255, 255),  # Np
    (255, 255, 255),  # Pu
    (255, 255, 255),  # Am
    (255, 255, 255),  # Cm
    (255, 255, 255),  # Bk
    (255, 255, 255),  # Cf
    (255, 255, 255),  # Es
    (255, 255, 255),  # Fm
    (255, 255, 255),  # Md
    (255, 255, 255),  # No
    (255, 255, 255),  # Lr
    (255, 255, 255),  # Rf
    (255, 255, 255),  # Db
    (255, 255, 255),  # Sg
    (255, 255, 255),  # Bh
    (255, 255, 255),  # Hs
    (255, 255, 255),  # Mt
    (255, 255, 255),  # Ds
    (255, 255, 255),  # Rg
    (255, 255, 255),  # Cn
    (255, 255, 255),  # Nh
    (255, 255, 255),  # Fl
    (255, 255, 255),  # Mc
    (255, 255, 255),  # Lv
    (255, 255, 255),  # Ts
    (255, 255, 255),  # Og
])
"""Contains the RGB colors of each element on the scale of 0-255."""


# Covalent radii from:
#
#  Covalent radii revisited,
#  Beatriz Cordero, Verónica Gómez, Ana E. Platero-Prats, Marc Revés,
#  Jorge Echeverría, Eduard Cremades, Flavia Barragán and Santiago Alvarez,
#  Dalton Trans., 2008, 2832-2838 DOI:10.1039/B801115J
#
_NO_COVALENT_RADIUS = 0.2
"""Default covalent radius of unknown or unlisted elements in Angstroms."""

ATOMIC_COVALENT_RADII = np.array([
    _NO_COVALENT_RADIUS,
    0.31,  # H
    0.28,  # He
    1.28,  # Li
    0.96,  # Be
    0.84,  # B
    0.76,  # C
    0.71,  # N
    0.66,  # O
    0.57,  # F
    0.58,  # Ne
    1.66,  # Na
    1.41,  # Mg
    1.21,  # Al
    1.11,  # Si
    1.07,  # P
    1.05,  # S
    1.02,  # Cl
    1.06,  # Ar
    2.03,  # K
    1.76,  # Ca
    1.70,  # Sc
    1.60,  # Ti
    1.53,  # V
    1.39,  # Cr
    1.39,  # Mn
    1.32,  # Fe
    1.26,  # Co
    1.24,  # Ni
    1.32,  # Cu
    1.22,  # Zn
    1.22,  # Ga
    1.20,  # Ge
    1.19,  # As
    1.20,  # Se
    1.20,  # Br
    1.16,  # Kr
    2.20,  # Rb
    1.95,  # Sr
    1.90,  # Y
    1.75,  # Zr
    1.64,  # Nb
    1.54,  # Mo
    1.47,  # Tc
    1.46,  # Ru
    1.42,  # Rh
    1.39,  # Pd
    1.45,  # Ag
    1.44,  # Cd
    1.42,  # In
    1.39,  # Sn
    1.39,  # Sb
    1.38,  # Te
    1.39,  # I
    1.40,  # Xe
    2.44,  # Cs
    2.15,  # Ba
    2.07,  # La
    2.04,  # Ce
    2.03,  # Pr
    2.01,  # Nd
    1.99,  # Pm
    1.98,  # Sm
    1.98,  # Eu
    1.96,  # Gd
    1.94,  # Tb
    1.92,  # Dy
    1.92,  # Ho
    1.89,  # Er
    1.90,  # Tm
    1.87,  # Yb
    1.87,  # Lu
    1.75,  # Hf
    1.70,  # Ta
    1.62,  # W
    1.51,  # Re
    1.44,  # Os
    1.41,  # Ir
    1.36,  # Pt
    1.36,  # Au
    1.32,  # Hg
    1.45,  # Tl
    1.46,  # Pb
    1.48,  # Bi
    1.40,  # Po
    1.50,  # At
    1.50,  # Rn
    2.60,  # Fr
    2.21,  # Ra
    2.15,  # Ac
    2.06,  # Th
    2.00,  # Pa
    1.96,  # U
    1.90,  # Np
    1.87,  # Pu
    1.80,  # Am
    1.69,  # Cm
    _NO_COVALENT_RADIUS,  # Bk
    _NO_COVALENT_RADIUS,  # Cf
    _NO_COVALENT_RADIUS,  # Es
    _NO_COVALENT_RADIUS,  # Fm
    _NO_COVALENT_RADIUS,  # Md
    _NO_COVALENT_RADIUS,  # No
    _NO_COVALENT_RADIUS,  # Lr
    _NO_COVALENT_RADIUS,  # Rf
    _NO_COVALENT_RADIUS,  # Db
    _NO_COVALENT_RADIUS,  # Sg
    _NO_COVALENT_RADIUS,  # Bh
    _NO_COVALENT_RADIUS,  # Hs
    _NO_COVALENT_RADIUS,  # Mt
    _NO_COVALENT_RADIUS,  # Ds
    _NO_COVALENT_RADIUS,  # Rg
    _NO_COVALENT_RADIUS,  # Cn
    _NO_COVALENT_RADIUS,  # Nh
    _NO_COVALENT_RADIUS,  # Fl
    _NO_COVALENT_RADIUS,  # Mc
    _NO_COVALENT_RADIUS,  # Lv
    _NO_COVALENT_RADIUS,  # Ts
    _NO_COVALENT_RADIUS,  # Og
])
"""Contains the standard atomic covalent radii of each element in Angstroms."""

NATURAL_CUTOFFS_DICT = {
    ATOMIC_SYMBOLS[z]: ATOMIC_COVALENT_RADII[z]
    for z in range(1, 118+1)
}
"""Dictionary mapping atomic symbols to standard atomic covalent radii."""

NATURAL_MASSES_DICT = {
    ATOMIC_SYMBOLS[z]: ATOMIC_MASSES[z]
    for z in range(1, 118+1)
}
"""Dictionary mapping atomic symbols to standard atomic masses."""
