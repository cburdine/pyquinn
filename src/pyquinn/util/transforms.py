import numpy as np
from scipy.linalg import expm


def voigt_strain_to_transform(strain_voigt, finite_strain=True):

    strain_voigt = np.array(strain_voigt)

    eps = np.array(
        [
            [strain_voigt[0], strain_voigt[5], strain_voigt[4]],
            [strain_voigt[5], strain_voigt[1], strain_voigt[3]],
            [strain_voigt[4], strain_voigt[3], strain_voigt[2]],
        ]
    )

    F = expm(eps) if finite_strain else (np.eye(3) + eps)

    return F


def random_voigt_strain(sigma_voigt):

    voigt_strain = np.random.normal(6) * sigma_voigt

    return voigt_strain
