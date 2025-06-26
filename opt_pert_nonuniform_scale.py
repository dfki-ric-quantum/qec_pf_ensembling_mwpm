import csv
import os
from dataclasses import dataclass
from functools import partial
from itertools import product
from multiprocessing.pool import Pool

import numpy as np
from absl import app, flags, logging
from pymatching import Matching
from scipy.optimize import direct

from qec_codes.toric_code.hgp import toric_code_stabolizers_and_logicals
from qec_codes.util import jeffreys_interval

D = [8, 16]
P = [0.05, 0.07, 0.1, 0.11]
STD = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15]
N_SHOTS = 10000
N_VARS = 50
SMALL_SCALE = [0.001, 0.01, 0.02, 0.03, 0.04]


@dataclass
class EquiClass:
    """Equivalence class of correction operators with equal action on the codespace."""

    count: int
    operators: list[str]


class Objective:
    """Optimization objective."""

    def __init__(self, d: int, p: float, std: float, n_shots: int, n_vars: int) -> None:
        """The constructor.

        Parameters
        ----------
        d: int
            Toric code distance
        p: float
            Mean error rate
        std: float
            Error rate standard deviation
        n_shots: int
            Number of shots per experiment
        n_vars: int
            Number of perturbations per experiment
        """
        self.d = d
        self.pmean = p
        self.std = std
        self.n_shots = n_shots
        self.n_vars = n_vars

        self.rng = np.random.default_rng()

        self.bfp = self.rng.normal(loc=p, scale=std, size=2 * d**2)
        self.bfp = np.clip(self.bfp, a_min=0.00001, a_max=0.49999)

        self.H, self.L = toric_code_stabolizers_and_logicals(d)
        self.noise = (self.rng.random((n_shots, self.H.shape[1])) < self.bfp).astype(
            np.uint8
        )
        self.syndromes = (self.noise @ self.H.T) % 2
        self.actual_obs = (self.noise @ self.L.T) % 2
        self.trials = []

    def __call__(self, scale: float) -> float:
        """Call the optimization objective. Executes minimum weight perfect matching decoding
        for the toric code with the given parameters and returns the logical error rate
        under perturbed MWPM.

        Parameters
        ----------
        scale: float
            The standard deviation for the perturbation of the matching graph

        Returns
        -------
        Logical error rate under perturbation
        """
        n_errors = 0

        for shot in range(self.n_shots):
            equi_classes: dict[tuple[int, int], EquiClass] = dict()

            for _ in range(self.n_vars):
                weights = self.rng.normal(loc=self.bfp, scale=scale)
                weights = np.clip(weights, a_min=0.00001, a_max=0.49999)
                weights = np.log((1 - weights) / weights)

                matching = Matching.from_check_matrix(self.H, weights=weights)
                pred = matching.decode(self.syndromes[shot])

                logical_flip = tuple(self.L @ pred % 2)
                pred_str = "".join([str(pr) for pr in pred])

                if logical_flip in equi_classes:
                    equi_classes[logical_flip].count += 1
                    if pred_str not in equi_classes[logical_flip].operators:
                        equi_classes[logical_flip].operators.append(pred_str)
                else:
                    equi_classes[logical_flip] = EquiClass(
                        count=1, operators=[pred_str]
                    )

            _, equi_class = max(equi_classes.items(), key=lambda x: x[1].count)
            corr_str = equi_class.operators[0]
            corr = np.array([int(cs) for cs in corr_str], dtype=np.uint8)

            if not np.array_equal(self.actual_obs[shot], self.L @ corr % 2):
                n_errors += 1

        res = n_errors / self.n_shots
        self.trials.append((scale, res))
        return res


def optimize(d: int, p: float, std: float) -> dict:
    """Optimize the objective.

    Parameters
    ----------
    d: int
        Distance of the toric code
    p: float
        Mean bitflip error rate
    std: float
        Standard deviation of the error rate

    Returns
    -------
    A dictonary containing `d`, `p`, `std` the optimization result as well as a list with all
    perturbation standard deviations and their logical error rates that were tested.
    """
    logging.info("Starting %d, %f, %f...", d, p, std)
    objective = Objective(d=d, p=p, std=std, n_shots=N_SHOTS, n_vars=N_VARS)

    opt_res = direct(objective, bounds=[(0.0, 0.5)], eps=1e-3, maxfun=1)

    logging.info("Finished %d, %f, %f.", d, p, std)

    return {
        "d": d,
        "p": p,
        "std": std,
        "opt_res": opt_res,
        "trials": objective.trials,
    }


def fixed_vals(d: int, p: float, std: float, vals: list[float]) -> dict:
    """Execute the optimization objective with a list of fixed values.

    Parameters
    ----------
    d: int
        Distance of the toric code
    p: float
        Mean bitflip error
    std: float
        Standard deviation of the error rate
    vals: list[float]
        List of perturbation standard deviations to try

    Returns
    -------
    A dictonary containing `d`, `p`, `std`, and perturbation standard deviations and
    their logical error rates that were tested.

    """
    logging.info("Starting %d, %f, %f...", d, p, std)
    objective = Objective(d=d, p=p, std=std, n_shots=N_SHOTS, n_vars=N_VARS)

    res = []
    for scale in vals:
        res.append(objective(scale))

    logging.info("Finished %d, %f, %f.", d, p, std)

    return {
        "d": d,
        "p": p,
        "std": std,
        "res": res,
    }


def group_data(opt: list[dict], small: list[dict]) -> dict:
    """Group results into suitable format for csv export.

    Parameters
    ----------
    opt: list[dict]
        The optimization result
    small: list[dict]
        Results for fixed small values

    Returns
    -------
    A dictonary reflecting the hierarchy of the csv export
    """
    grouped = {}

    for optr in opt:
        d = optr["d"]
        p = optr["p"]
        std = optr["std"]
        opt = optr["opt_res"].x[0]

        if d not in grouped:
            grouped[d] = {}

        if p not in grouped[d]:
            grouped[d][p] = {}

        if std not in grouped[d][p]:
            grouped[d][p][std] = {
                "scale": [],
                "pl": [],
                "lower": [],
                "upper": [],
                "opt": opt,
            }

        for s, pl in optr["trials"]:
            if s < 1e-6:
                continue
            grouped[d][p][std]["scale"].append(s[0])
            grouped[d][p][std]["pl"].append(pl)
            lower, upper = jeffreys_interval(pl * N_SHOTS, N_SHOTS)
            grouped[d][p][std]["lower"].append(np.abs(pl - lower))
            grouped[d][p][std]["upper"].append(np.abs(upper - pl))

    for dr in small:
        d = dr["d"]
        p = dr["p"]
        std = dr["std"]

        for pl, s in zip(dr["res"], SMALL_SCALE):
            grouped[d][p][std]["scale"].append(s)
            grouped[d][p][std]["pl"].append(pl)
            lower, upper = jeffreys_interval(pl * N_SHOTS, N_SHOTS)
            grouped[d][p][std]["lower"].append(np.abs(pl - lower))
            grouped[d][p][std]["upper"].append(np.abs(upper - pl))

    return grouped


def write_results(grouped: dict, fname: str) -> None:
    """Write results to csv file.

    Parameters
    ----------
    grouped: dict
        The grouped results as returned by `group_data`
    fname: str
        Filename to write to, without path
    """
    os.makedirs("results/opt", exist_ok=True)

    with open(f"results/opt/{fname}", "w") as outfile:
        fieldnames = ["d", "p", "std", "opt", "scale", "pl", "lower", "upper"]

        writer = csv.writer(outfile)
        writer.writerow(fieldnames)

        for d, ddata in grouped.items():
            for p, pdata in ddata.items():
                for std, sdata in pdata.items():
                    opt = sdata["opt"]
                    for s, pl, lower, upper in zip(
                        sdata["scale"], sdata["pl"], sdata["lower"], sdata["upper"]
                    ):
                        writer.writerow([d, p, std, opt, s, pl, lower, upper])


FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="out", short_name="o", default="opt_nonunifcsv", help="Output file"
)
flags.DEFINE_integer(name="procs", short_name="p", default=8, help="No. of processes")


def main(_):
    ws = product(D, P, STD)

    objectives = {"opt": optimize, "small": partial(fixed_vals, vals=SMALL_SCALE)}
    res = dict()

    for key, objective in objectives.items():
        with Pool(processes=FLAGS.procs) as pool:
            res[key] = pool.starmap(objective, ws)

    grouped_res = group_data(**res)
    write_results(grouped_res, FLAGS.out)


if __name__ == "__main__":
    app.run(main)
