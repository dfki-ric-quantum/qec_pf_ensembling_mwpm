import stim

from qec_codes.toric_code import generate_toric_code
from qec_codes.rotated_surface_code import generate_rotated_surface_code
from qec_codes.surface_code import generate_surface_code

CODE_TASKS: list[str] = [
    "toric_code:memory_xx",
    "toric_code:memory_zz",
    "surface_code:memory_x",
    "surface_code:memory_z",
    "rotated_surface_code:memory_x",
    "rotated_surface_code:memory_z",
]


def _validate_code_task(code_task: str) -> None:
    if code_task not in CODE_TASKS:
        raise ValueError(
            f"No such code task: {code_task}. Supported: {', '.join(CODE_TASKS)}"
        )


def generate(
    code_task: str,
    *,
    distance: int,
    rounds: int,
    after_clifford_depolarization: float = 0.0,
    before_round_data_bitflip: float = 0.0,
    before_round_data_depolarization: float = 0.0,
    before_measure_flip_probability: float = 0.0,
    after_reset_flip_probability: float = 0.0,
    include_end_detectors: bool = False,
) -> stim.Circuit:
    """Generate stim circuit for a code task.

    Parameters
    ----------
    code_task: str
        Code task to perform, currently supported 'toric_code:memory_xx', 'toric_code:memory_zz',
        'surface_code:memory_x', 'surface_code:memory_z', 'rotated_surface_code:memory_x',
        and 'rotated_surface_code:memory_z'
    distance: int
        Distance of the toric code
    rounds: int
        Rounds of syndrome measurements
    after_clifford_depolarization: float
        Defaults to 0. The probability (p) of `DEPOLARIZE1(p)` operations
        to add after every single-qubit Clifford operation and `DEPOLARIZE2(p)`
        operations to add after every two-qubit Clifford operation. The after-Clifford
        depolarizing operations are only included if this probability is not 0.
    before_round_data_bitflip: float
        Defaults to 0. The probability (p) of `X_ERROR(p)` operations to apply to
        every data qubit at the start of a round of stabilizer measurements. X basis measurements
        us 'Z_ERROR(p)' instead.
    before_round_data_depolarization: float
        Defaults to 0. The probability (p) of `DEPOLARIZE1(p)` operations to apply
        to every data qubit at the start of a round of stabilizer measurements.
        The start-of-round depolarizing operations are only included if this probability
        is not 0.
    before_measure_flip_probability: float
        Defaults to 0. The probability (p) of `X_ERROR(p)` operations applied to qubits
        before each measurement (X basis measurements use `Z_ERROR(p)` instead). The
        before-measurement flips are only included if this probability is not 0.
    after_reset_flip_probability: float
        Defaults to 0. The probability (p) of `X_ERROR(p)` operations applied to qubits
        after each reset (X basis resets use `Z_ERROR(p)` instead). The after-reset flips are
        only included if this probability is not 0.
    include_end_detectors: bool
        Include detectors for parity between data qubits and their ancillas before observable
        measurement.

    Returns
    -------
    The generated circuit.
    """
    _validate_code_task(code_task)

    code = code_task.split(":")[0]
    observable = code_task.split("_")[-1]

    generators = {
        "toric_code": generate_toric_code,
        "rotated_surface_code": generate_rotated_surface_code,
        "surface_code": generate_surface_code
    }

    try:
        return generators[code](
            observable,
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=after_clifford_depolarization,
            before_round_data_bitflip=before_round_data_bitflip,
            before_round_data_depolarization=before_round_data_depolarization,
            before_measure_flip_probability=before_measure_flip_probability,
            after_reset_flip_probability=after_reset_flip_probability,
            include_end_detectors=include_end_detectors,
        )
    except KeyError:
        raise ValueError(f"Code task {code_task} not implemented")
