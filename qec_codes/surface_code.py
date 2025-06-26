from collections.abc import Iterable

import stim

from qec_codes.util import sorted_complex

P2D: list[complex] = [-1j, -1, 1, 1j]
P2V: list[complex] = [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j]
V2D: list[complex] = [-1j, -1, 1, 1j]


def in_bounds(coord: complex, distance: int) -> bool:
    """Check if a qubit coordinate is within bounds for the planar code.

    Parameters
    ----------
    coords: complex
        The qubit coordinates as complex number
    distance: int
        Distance of the code

    Returns
    -------
    True if the coordinates are within bounds False otherwise
    """
    in_left_upper = 0 <= coord.real <= 2 * (distance - 1)
    in_right_lower = 0 <= coord.imag <= 2 * (distance - 1)

    return in_left_upper and in_right_lower


def generate_syndrome_cycle(
    observable: str,
    q2i: dict[complex, int],
    data_qubits: Iterable[complex],
    plaq_ancillas: Iterable[complex],
    vertex_ancillas: Iterable[complex],
    distance: int,
    detectors: bool = False,
    after_clifford_depolarization: float = 0.0,
    before_round_data_bitflip: float = 0.0,
    before_round_data_depolarization: float = 0.0,
    before_measure_flip_probability: float = 0.0,
    after_reset_flip_probability: float = 0.0,
) -> stim.Circuit:
    """Generate a syndrome cycle for the planar Surface Code (unrotated).

    Parameters
    ----------
    observable: str
        Either 'x' or 'z', the observable measured on the logical qubit at the end
        of the experiments.
    q2i: dict[complex, int]
        Mapping from complex qubit coordinates to their index
    data_qubits: Iterable[complex]
        Coordinates of the data qubits
    plaq_ancillas: Iterable[complex]
        Coordinates of the ancillas on the plaquettes
    vertex_ancillas: Iterable[complex]
        Coordinates of the ancillas on the vertices
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

    x_syndrome_ancillas = [q2i[va] for va in vertex_ancillas]
    z_syndrome_ancillas = [q2i[pa] for pa in plaq_ancillas]

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

    circuit.append_operation("H", x_syndrome_ancillas)

    if after_clifford_depolarization > 0.0:
        circuit.append_operation(
            "DEPOLARIZE1", x_syndrome_ancillas, after_clifford_depolarization
        )

    circuit.append_operation("TICK")

    for direction in range(4):
        x_syndrome_cnot: list[int] = []
        z_syndrome_cnot: list[int] = []

        for va in vertex_ancillas:
            control = q2i[va]

            target = va + V2D[direction]
            if in_bounds(target, distance):
                x_syndrome_cnot += [control, q2i[target]]

        for pa in plaq_ancillas:
            target = q2i[pa]

            control = pa + P2D[direction]
            if in_bounds(control, distance):
                z_syndrome_cnot += [q2i[control], target]

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

    circuit.append_operation("H", x_syndrome_ancillas)

    if after_clifford_depolarization > 0.0:
        circuit.append_operation(
            "DEPOLARIZE1", x_syndrome_ancillas, after_clifford_depolarization
        )

    circuit.append_operation("TICK")

    if before_measure_flip_probability > 0.0:
        circuit.append_operation(
            "X_ERROR",
            x_syndrome_ancillas + z_syndrome_ancillas,
            before_measure_flip_probability,
        )

    circuit.append_operation("MR", x_syndrome_ancillas)
    circuit.append_operation("MR", z_syndrome_ancillas)

    if after_reset_flip_probability > 0.0:
        circuit.append_operation(
            "X_ERROR",
            x_syndrome_ancillas + z_syndrome_ancillas,
            after_reset_flip_probability,
        )

    if detectors:
        offset = len(plaq_ancillas) + len(vertex_ancillas)
        target_index = -1

        for pa in reversed(plaq_ancillas):
            circuit.append_operation(
                "DETECTOR",
                [stim.target_rec(target_index), stim.target_rec(target_index - offset)],
                [pa.real, pa.imag, 0],
            )
            target_index -= 1

        for va in reversed(vertex_ancillas):
            circuit.append_operation(
                "DETECTOR",
                [stim.target_rec(target_index), stim.target_rec(target_index - offset)],
                [va.real, va.imag, 0],
            )
            target_index -= 1

        # circuit.append_operation("SHIFT_COORDS", [], [0, 0, 1])

    return circuit


def build_observable(
    observable: str, *, data_qubits: Iterable[complex]
) -> stim.Circuit:
    """Build observable for the planar surface code.

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
        if int(dq.real) == 0 and observable == 'z':
            obs_rec.append(stim.target_rec(-target))
        elif int(dq.imag) == 0 and observable == 'x':
            obs_rec.append(stim.target_rec(-target))

    circuit.append_operation("OBSERVABLE_INCLUDE", obs_rec, 0)

    return circuit


def build_end_detectors(
    observable: str,
    *,
    distance: int,
    data_qubits: Iterable[complex],
    plaq_ancillas: Iterable[complex],
    vertex_ancillas: Iterable[complex],
) -> stim.Circuit:
    """Build detectors for the end of the memory experiment.

    The final detectors check for parity of the data qubits with their four surrounding ancillas,
    in case of the ZZ observable, the plaquette ancillas, in case of the XX observable the link
    ancillas. These detectors find errors in the final measurement of the observable.

    Parameters
    ----------
    observable: str
        Either 'x' or 'z', the observable measured on both logic qubits at the end
        of the experiments.
    distance: int
        Distance of the toric code
    data_qubits: Iterable[complex]
        Coordinates of data qubits
    plaq_ancillas: set[complex]
        Coordinates of the ancillas on the plaquettes
    vertex_ancillas: set[complex]
        Coordinates of the ancillas on the vertices
    """
    circuit = stim.Circuit()

    all_qubits = [*vertex_ancillas, *plaq_ancillas, *data_qubits]
    meas_order = dict(zip(reversed(all_qubits), range(-1, -(len(all_qubits) + 1), -1)))

    detector_targets: list[list[stim.stim.GateTarget]] = []
    detector_coords: list[complex] = []

    if observable == "z":
        for pa in plaq_ancillas:
            target_records = [stim.target_rec(meas_order[pa])]

            target_records += [
                stim.target_rec(meas_order[pa + shift])
                for shift in P2D
                if in_bounds(pa + shift, distance)
            ]
            detector_targets.append(target_records)
            detector_coords.append(pa)
    else:
        for va in vertex_ancillas:
            target_records = [stim.target_rec(meas_order[va])]
            target_records += [
                stim.target_rec(meas_order[va + shift])
                for shift in V2D
                if in_bounds(va + shift, distance)
            ]
            detector_targets.append(target_records)
            detector_coords.append(va)

    for detector_target, detector_coord in zip(detector_targets, detector_coords):
        circuit.append_operation(
            "DETECTOR",
            detector_target,
            [detector_coord.real, detector_coord.imag, 1],
        )

    return circuit


def generate_surface_code(
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
    """Generate the (unrotated) surface code

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

    plaq_ancillas = set()
    data_qubits = set()
    vertex_ancillas = set()

    for row in range(distance):
        for col in range(distance):
            center = 2 * col + 1 + 2j * row

            if in_bounds(center, distance):
                plaq_ancillas.add(center)

            for shift in P2D:
                coord = center + shift
                if in_bounds(coord, distance):
                    data_qubits.add(coord)

            for shift in P2V:
                coord = center + shift
                if in_bounds(coord, distance):
                    vertex_ancillas.add(coord)

    plaq_ancillas = sorted_complex(plaq_ancillas)
    vertex_ancillas = sorted_complex(vertex_ancillas)
    data_qubits = sorted_complex(data_qubits)

    q2i: dict[complex, int] = {
        q: i
        for i, q in enumerate(
            sorted_complex([*plaq_ancillas, *vertex_ancillas, *data_qubits])
        )
    }

    data_qubit_idxs = [q2i[dq] for dq in data_qubits]
    ancilla_idxs = [q2i[ancilla] for ancilla in [*plaq_ancillas, *vertex_ancillas]]

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
        plaq_ancillas=plaq_ancillas,
        vertex_ancillas=vertex_ancillas,
        detectors=False,
        distance=distance,
    )

    for _ in range(rounds):
        circuit += generate_syndrome_cycle(
            observable=observable,
            q2i=q2i,
            data_qubits=data_qubits,
            plaq_ancillas=plaq_ancillas,
            vertex_ancillas=vertex_ancillas,
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
        circuit += build_end_detectors(
            observable,
            distance=distance,
            data_qubits=data_qubits,
            plaq_ancillas=plaq_ancillas,
            vertex_ancillas=vertex_ancillas,
        )

    circuit += build_observable(observable, data_qubits=data_qubits)

    return circuit
