from collections.abc import Iterable

from scipy import stats


def sorted_complex(xs: Iterable[complex]) -> list[complex]:
    """Sorts a collection of complex numbers, by real then imaginary part."""
    return sorted(xs, key=lambda x: (x.real, x.imag))


def jeffreys_interval(
    n_failures: int,
    n_shots: int,
    prior: tuple[float, float] = [0.5, 0.5],
    conf: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval with Jeffrey's prior.

    Parameters
    ----------
    n_failures: int
        Number of failures
    n_shots: int
        Number of shots
    prior: tuple[float, float] = [0.5, 0.5]
        Prior distribution
    conf: float = 0.95
        Confidence interval

    Returns
    -------
    The lower and upper bound of the confidence interval
    """

    a, b = prior
    n_successes = n_shots - n_failures
    a += n_successes
    b += n_failures

    lower_bound = stats.beta.ppf((1 - conf) / 2, a, b)
    upper_bound = stats.beta.ppf(1 - (1 - conf) / 2, a, b)
    return 1 - lower_bound, 1 - upper_bound
