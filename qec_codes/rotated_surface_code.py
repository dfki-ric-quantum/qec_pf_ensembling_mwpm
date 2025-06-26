from collections.abc import Iterable

import stim

from qec_codes.util import sorted_complex

X2D = [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]
Z2D = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]


def generate_syndrome_cycle(
    observable: str,
    q2i: dict[complex, int],
    data_qubits: Iterable[complex],
    x_ancillas: Iterable[complex],
    z_ancillas: Iterable[complex],
    distance: int,
    detectors: bool = False,
    after_clifford_depolarization: float = 0.0,
    before_round_data_bitflip: float = 0.0,
    before_round_data_depolarization: float = 0.0,
    before_measure_flip_probability: float = 0.0,
    after_reset_flip_probability: float = 0.0,
) -> stim.Circuit:
    """Generate a syndrome cycle for the rotated surface code code.

    Parameters
    ----------
    observable: str
        Either 'x' or 'z', the observable measured on the logic qubit at the end
        of the experiments.
    q2i: dict[complex, int]
        Mapping from complex qubit coordinates to their index
    data_qubits: Iterable[complex]
        Coordinates of the data qubits
    x_ancillas: Iterable[complex]
        Coordinates of the X-stabilizer ancillas
    z_ancillas: Iterable[complex]
        Coordinates of the Z-stabilizer ancillas
    distance: int
        Distance of the toric code
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

    Returns
    -------
    The generated circuit for the syndrome cycle
    """
    x_syndrome_idxs = [q2i[xa] for xa in x_ancillas]
    z_syndrome_idxs = [q2i[za] for za in z_ancillas]

    circuit = stim.Circuit()

    circuit.append_operation("TICK")

    if before_round_data_depolarization > 0.0:
        circuit.append_operation(
            "DEPOLARIZE1",
            [q2i[dq] for dq in data_qubits],
            before_round_data_depolarization,
        )

    if before_round_data_bitflip > 0.0:
        bf_error = "X_ERROR" if observable == "z" else "Z_ERROR"
        circuit.append_operation(
            bf_error,
            [q2i[dq] for dq in data_qubits],
            before_round_data_bitflip,
        )

    circuit.append_operation("H", x_syndrome_idxs)

    if after_clifford_depolarization > 0.0:
        circuit.append_operation(
            "DEPOLARIZE1", x_syndrome_idxs, after_clifford_depolarization
        )

    circuit.append_operation("TICK")

    #####
    # TODO: CNOTs
    ####
    for direction in range(4):
        x_syndrome_cnot: list[int] = []
        z_syndrome_cnot: list[int] = []

        for xa in x_ancillas:
            dq_coord = xa + X2D[direction]

            if dq_coord in data_qubits:
                control = q2i[xa]
                target = q2i[dq_coord]
                x_syndrome_cnot += [control, target]

        for za in z_ancillas:
            dq_coord = za + Z2D[direction]

            if dq_coord in data_qubits:
                control = q2i[dq_coord]
                target = q2i[za]
                z_syndrome_cnot += [control, target]

        circuit.append_operation("CNOT", x_syndrome_cnot)

        if after_clifford_depolarization > 0.0:
            circuit.append_operation(
                "DEPOLARIZE2", x_syndrome_cnot, after_clifford_depolarization
            )

        circuit.append_operation("CNOT", z_syndrome_cnot)

        if after_clifford_depolarization > 0.0:
            circuit.append_operation(
                "DEPOLARIZE2", z_syndrome_cnot, after_clifford_depolarization
            )

        circuit.append_operation("TICK")

    circuit.append_operation("H", x_syndrome_idxs)

    if after_clifford_depolarization > 0.0:
        circuit.append_operation(
            "DEPOLARIZE1", x_syndrome_idxs, after_clifford_depolarization
        )

    circuit.append_operation("TICK")

    if before_measure_flip_probability > 0.0:
        circuit.append_operation(
            "X_ERROR",
            x_syndrome_idxs + z_syndrome_idxs,
            before_measure_flip_probability,
        )

    circuit.append_operation("MR", x_syndrome_idxs)
    circuit.append_operation("MR", z_syndrome_idxs)

    if after_reset_flip_probability > 0.0:
        circuit.append_operation(
            "X_ERROR",
            x_syndrome_idxs + z_syndrome_idxs,
            after_reset_flip_probability,
        )

    if detectors:
        offset = len(x_ancillas) + len(z_ancillas)
        target_index = -1

        for za in reversed(z_ancillas):
            circuit.append_operation(
                "DETECTOR",
                [stim.target_rec(target_index), stim.target_rec(target_index - offset)],
                [za.real, za.imag, 0],
            )
            target_index -= 1

        for xa in reversed(x_ancillas):
            circuit.append_operation(
                "DETECTOR",
                [stim.target_rec(target_index), stim.target_rec(target_index - offset)],
                [xa.real, xa.imag, 0],
            )
            target_index -= 1

        # circuit.append_operation("SHIFT_COORDS", [], [0, 0, 1])

    return circuit


def build_observable(
    observable: str, *, data_qubits: Iterable[complex]
) -> stim.Circuit:
    """Build observable for the rotated surface code.

    Parameters
    ----------
    observable: str
        Either 'x' or 'z', the observable measured on the logical qubit at the end
        of the experiments.
    data_qubits: Iterable[complex]
        Coordinates of data qubits

    Returns
    -------
    A circuit containing the observable for both logical qubits
    """
    circuit = stim.Circuit()

    obs_rec: list[stim.GateTarget] = []

    for target, dq in enumerate(reversed(data_qubits), start=1):
        if int(dq.real) == 1 and observable == "x":
            obs_rec.append(stim.target_rec(-target))
        elif int(dq.imag) == 1 and observable == "z":
            obs_rec.append(stim.target_rec(-target))

    circuit.append_operation("OBSERVABLE_INCLUDE", obs_rec, 0)

    return circuit


def generate_rotated_surface_code(
    observable: str,
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
    """Generate the rotated surface code

    Parameters
    ----------
    observable: str
        Either 'x' or 'z', the observable measured on the logic qubit at the end
        of the experiments.
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
    The generated circuit for the syndrome cycles
    """

    data_qubits = set()
    x_ancillas = set()
    z_ancillas = set()

    for row in range(1, 2 * distance, 2):
        for col in range(1, 2 * distance, 2):
            data_coord = col + row * 1j
            data_qubits.add(data_coord)

    for ridx, row in enumerate(range(0, 2 * distance + 1, 2)):
        for cidx, col in enumerate(range(2, 2 * distance, 2)):
            even = ridx % 2 == 0 and cidx % 2 == 0
            odd = ridx % 2 == 1 and cidx % 2 == 1

            if even or odd:
                x_coord = col + row * 1j
                x_ancillas.add(x_coord)

    for ridx, row in enumerate(range(2, 2 * distance, 2)):
        for cidx, col in enumerate(range(0, 2 * distance + 1, 2)):
            inner = ridx % 2 == 0 and cidx % 2 == 1
            outer = ridx % 2 == 1 and cidx % 2 == 0

            if inner or outer:
                z_coord = col + row * 1j
                z_ancillas.add(z_coord)

    x_ancillas = sorted_complex(x_ancillas)
    z_ancillas = sorted_complex(z_ancillas)
    data_qubits = sorted_complex(data_qubits)

    q2i: dict[complex, int] = {
        q: i
        for i, q in enumerate(sorted_complex([*x_ancillas, *z_ancillas, *data_qubits]))
    }

    data_qubit_idxs = [q2i[dq] for dq in data_qubits]
    ancilla_idxs = [q2i[ancilla] for ancilla in [*x_ancillas, *z_ancillas]]

    circuit = stim.Circuit()

    for q, i in q2i.items():
        circuit.append_operation("QUBIT_COORDS", [i], [q.real, q.imag])

    circuit.append_operation("R", ancilla_idxs)

    if after_reset_flip_probability > 0.0:
        circuit.append_operation("X_ERROR", ancilla_idxs, after_reset_flip_probability)

    if observable == "z":
        circuit.append_operation("R", data_qubit_idxs)

        if after_reset_flip_probability > 0.0:
            circuit.append_operation(
                "X_ERROR", data_qubit_idxs, after_reset_flip_probability
            )
    else:
        circuit.append_operation("RX", data_qubit_idxs)

        if after_reset_flip_probability > 0.0:
            circuit.append_operation(
                "Z_ERROR", data_qubit_idxs, after_reset_flip_probability
            )

    circuit += generate_syndrome_cycle(
        observable=observable,
        q2i=q2i,
        data_qubits=data_qubits,
        x_ancillas=x_ancillas,
        z_ancillas=z_ancillas,
        detectors=False,
        distance=distance,
    )

    for _ in range(rounds):
        circuit += generate_syndrome_cycle(
            observable=observable,
            q2i=q2i,
            data_qubits=data_qubits,
            x_ancillas=x_ancillas,
            z_ancillas=z_ancillas,
            detectors=True,
            distance=distance,
            after_clifford_depolarization=after_clifford_depolarization,
            before_round_data_bitflip=before_round_data_bitflip,
            before_round_data_depolarization=before_round_data_depolarization,
            before_measure_flip_probability=before_measure_flip_probability,
            after_reset_flip_probability=after_reset_flip_probability,
        )

    circuit.append_operation("TICK")

    if observable == "z":
        if before_measure_flip_probability > 0.0:
            circuit.append("X_ERROR", data_qubit_idxs, before_measure_flip_probability)

        circuit.append_operation("M", data_qubit_idxs)
    else:
        if before_measure_flip_probability > 0.0:
            circuit.append("Z_ERROR", data_qubit_idxs, before_measure_flip_probability)

        circuit.append_operation("MX", data_qubit_idxs)

    if include_end_detectors:
        raise NotImplementedError("End detectors not yet implemented")
    #     circuit += build_end_detectors(
    #         observable,
    #         distance=distance,
    #         data_qubits=data_qubits,
    #         x_ancillas=x_ancillas,
    #         z_ancillas=z_ancillas,
    #     )

    circuit += build_observable(observable, data_qubits=data_qubits)

    return circuit
