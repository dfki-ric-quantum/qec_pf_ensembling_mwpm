import numpy as np
import stim

from qec_codes import generate


def dem_to_check_matrix(
    dem: stim.DetectorErrorModel, filter_inactive_detectors: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a stim detector error model to parity check matrix.

    Parameters
    ----------
    dem: stim.DetectorErrorModel
        The detector error model
    filter_inactive_detectors: bool = True
        If True, filter out detectors that have no error that triggers them

    Returns
    -------
    Parity check matrix, logical observable check matrix and weights for each error.
    """
    H = np.zeros((dem.num_detectors, dem.num_errors), dtype=np.uint8)
    p = np.empty((dem.num_errors,), dtype=np.float32)
    L = np.zeros((dem.num_observables, dem.num_errors), dtype=np.uint8)

    error_idx = 0

    for instruction in dem:
        if instruction.type == "error":
            for t in instruction.targets_copy():
                v = str(t)
                if v.startswith("D"):
                    H[int(v[1:]), error_idx] = 1
                elif v.startswith("L"):
                    L[int(v[1:]), error_idx] = 1

            perr = float(instruction.args_copy()[0])
            p[error_idx] = np.log((1 - perr) / perr)
            error_idx += 1

    if filter_inactive_detectors:
        H = H[~np.all(H == 0, axis=1)]

    return H, L, p


def stabilizers_and_logical_from_circuit(
    code: str, d: int, p: float, *, observable: str
) -> tuple[np.ndarray, np.ndarray]:
    """Get stabilizer check matrix and logical observable for a code.

    Parameters
    ----------
    code: str
        The code to generate the objects for, "toric", "planar" or "rotated"
    d: int
        distance of the code
    p: float
        Data qubit flip probability
    observable: str
        Observable to consider 'x' or 'z'

    Returns
    -------
    The check matrix and the logical operator for the observable
    """

    _codes = {
        "toric": "toric_code:memory_{}",
        "planar": "surface_code:memory_{}",
        "rotated": "rotated_surface_code:memory_{}",
    }

    circuit = generate(
        _codes[code].format(observable),
        distance=d,
        rounds=1,
        before_round_data_bitflip=p,
    )

    H, L, _ = dem_to_check_matrix(circuit.detector_error_model(decompose_errors=True))
    return H, L
