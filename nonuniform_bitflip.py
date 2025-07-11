import csv
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import product
from multiprocessing.pool import Pool

import numpy as np
from absl import app, flags, logging
from pymatching import Matching

from qec_codes import generate
from qec_codes.conversion import dem_to_check_matrix


@dataclass
class EquiClass:
    """Dataclass to store equivalent correction operator data.

    Parameters
    ----------
    count: int
        How often this class was encountered in decoding.
    operators: list[str]
        Bit strings of the unique operators that were decoded for this class.
    """

    count: int
    operators: list[str]


def get_error_model(d: int, p: Iterable[float]) -> tuple[np.ndarray]:
    """Get parity check matrix and logical observable for a code.

    Parameters
    ----------
    d: int
        Distance of the toric code
    p: Iterable[float]
        Physical error/flip probabilities

    Returns
    -------
    The parity check matrix for the X stabilizers, the corresponding logical
    observable and the propagated error probabities
    """
    circuit = generate(
        "toric_code:memory_xx",
        distance=d,
        rounds=1,
        before_round_data_bitflip=p,
    )

    return dem_to_check_matrix(circuit.detector_error_model(decompose_errors=True))


def base_decoding(
    H: np.ndarray,
    L: np.ndarray,
    p: float,
    syndromes: np.ndarray,
    actual_obs: np.ndarray,
) -> int:
    """Run vanilla MWPM decoder on syndromes.

    Parameters
    ----------
    H: np.ndarray
        The parity check matrix of the code
    L: np.ndarray
        Logical observables
    p: Iterable[float]
        The propagated error probabilities
    syndromes: np.ndarray
        syndromes, shape (n_shots, n_detectors)
    actual_obs: np.ndarray
        the actual observables flipped by the noise that created the syndromes, shape
        (n_shots, n_observables)

    Returns
    -------
    The number of decoding errors
    """

    matching = Matching.from_check_matrix(
        H, weights=np.log((1 - p) / p), faults_matrix=L
    )
    predicted_obs = matching.decode_batch(syndromes)

    return np.sum(np.any(predicted_obs != actual_obs, axis=1))


def compute_logical_errors(
    d: int,
    p: float,
    std: float,
    scale: float,
    *,
    n_vars: int,
    n_shots: int,
) -> tuple[int, float, int, int, int]:
    """Compute logical errors with pymatching, vanilla and with detector graph perturbation.

    Parameters
    ----------
    d: int
        Distance of the toric code
    p: float
        Physical error/flip probability mean
    std: float
        Physical error/flip probability std
    scale: float
        Perturbation std
    n_vars: int
        Number of perturbations to try
    n_shots: int
        Number of shots to decode

    Returns
    -------
    Distance, flip mean, flip std, perturbation scale, number of shots, number of
    logical errors perturbated, number of errors for vanilla pymatching
    """

    logging.info("Started: %d, %f, %f", d, p, std)


    rng = np.random.default_rng()
    bfp = rng.normal(loc=p, scale=std, size=2 * d**2)
    bfp = np.clip(bfp, a_min=0.00001, a_max=0.49999)

    H, L, _ = get_error_model(d, bfp)

    n_errors_pym = 0
    n_errors_per = 0

    noise = (rng.random((n_shots, H.shape[1])) < bfp).astype(np.uint8)
    syndromes = (noise @ H.T) % 2
    actual_obs = (noise @ L.T) % 2

    for shot in range(n_shots):
        equi_classes: dict[tuple[int, int], EquiClass] = dict()

        for _ in range(n_vars):
            weights = rng.normal(loc=bfp, scale=scale)
            weights = np.clip(weights, a_min=0.00001, a_max=0.49999)
            weights = np.log((1 - weights) / weights)

            matching = Matching.from_check_matrix(H, weights=weights)
            pred = matching.decode(syndromes[shot])

            logical_flip = tuple(L @ pred % 2)
            pred_str = "".join([str(pr) for pr in pred])

            if logical_flip in equi_classes:
                equi_classes[logical_flip].count += 1
                if pred_str not in equi_classes[logical_flip].operators:
                    equi_classes[logical_flip].operators.append(pred_str)
            else:
                equi_classes[logical_flip] = EquiClass(count=1, operators=[pred_str])

        _, equi_class = max(equi_classes.items(), key=lambda x: x[1].count)
        corr_str = equi_class.operators[0]
        corr = np.array([int(cs) for cs in corr_str], dtype=np.uint8)

        if not np.array_equal(actual_obs[shot], L @ corr % 2):
            n_errors_per += 1

    n_errors_pym = base_decoding(H, L, bfp, syndromes, actual_obs)

    logging.info("Finished: %d, %f, %f", d, p, std)

    return d, p, std, scale, n_shots, n_errors_pym, n_errors_per


DISTANCES: list[int] = [3, 4, 7, 8, 15, 16]
P: list[float] = [
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
]

STD: list[float] = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.12, 0.15]
SCALE = 0.025

RES_DIR = "results/"

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "num_workers", 8, short_name="n", help="number of worker processes"
)
flags.DEFINE_integer(
    "num_shots", 1000, short_name="s", help="number of shots for each trial"
)
flags.DEFINE_integer(
    "num_variations",
    100,
    short_name="p",
    help="number of perturbations for each trial",
)


def main(_):
    logging.info("Starting experiment with %d workers", FLAGS.num_workers)

    work_set = product(DISTANCES, P, STD)

    task = partial(
        compute_logical_errors,
        scale=SCALE,
        n_vars=FLAGS.num_variations,
        n_shots=FLAGS.num_shots,
    )
    with Pool(processes=FLAGS.num_workers) as pool:
        results = pool.starmap(task, work_set)

    os.makedirs(RES_DIR, exist_ok=True)
    time = datetime.now().strftime("%d-%m-%Y_%H-%M")

    fname = "{}/toric_code_nonuniform_pert_{}_{}.csv".format(
        RES_DIR,
        FLAGS.num_variations,
        time,
    )

    with open(fname, "w", newline="") as outfile:
        csv_writer = csv.writer(outfile, delimiter=",")
        csv_writer.writerow(
            ["d", "p", "std", "scale", "n_shots", "errors_base", "errors_pert"]
        )
        csv_writer.writerows(results)

    logging.info("Done, results written to %s", fname)


if __name__ == "__main__":
    app.run(main)
