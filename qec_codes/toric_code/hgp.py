import numpy as np
from scipy.linalg import block_diag


def repetition_code(d: int) -> np.ndarray:
    """Check matrix for the repetition code.

    Parameters
    ----------
    d: int
        Distance of the code

    Returns
    -------
    The check matrix
    """
    Hr = np.zeros((d, d), dtype=np.uint8)

    for i in range(d):
        for j in (i, (i + 1) % d):
            Hr[i][j] = 1
    return Hr


def toric_code_x_stabilisers(d: int) -> np.ndarray:
    """Check matrix for the X stabilizers of the toric code.

    Parameters
    ----------
    d: int
        Distance of the code

    Returns
    -------
    The check matrix
    """
    Hr = repetition_code(d)
    H = np.hstack(
        [
            np.kron(Hr, np.eye(Hr.shape[1], dtype=np.uint8)),
            np.kron(np.eye(Hr.shape[0], dtype=np.uint8), Hr.T),
        ],
        dtype=np.uint8,
    )
    return H % 2


def toric_code_x_logicals(d: int) -> np.ndarray:
    """Logical X operators of the toric code.

    Parameters
    ----------
    d: int
        Distance of the code

    Returns
    -------
    A matrix containing the logical operators per row and error mechanisms
    per column. Each entry is 1, if the mechanism flips the logical qubit,
    0 otherwise.
    """
    H1 = np.zeros((1, d), dtype=np.uint8)
    H1[0][0] = 1
    H0 = np.ones((1, d), dtype=np.uint8)
    return block_diag(np.kron(H1, H0), np.kron(H0, H1)) % 2


def toric_code_stabolizers_and_logicals(d: int) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to return parity check matrix and logical observables for the toric code.

    Parameters
    ----------
    d: int
        Distance of the code

    Returns
    -------
    The parity check matrix and the logical observable for the XX observable.
    """
    return toric_code_x_stabilisers(d), toric_code_x_logicals(d)
