# Ensembling MWPM decoding for surface code

This repository implements the perturbation based ensembling method for minimum weight perfect
matching for the rotated and unrotated planar surface code as well as the toric code as used in [1].

## Setup

Tested with python 3.11 and 3.12, to install the dependencies e.g. in an virtual environment, run
```
pip install -r requirements.txt
```

## Usage

The code supports two types of experiments from [1]. **Note:** Both, the optimization and the
simulation of ensembling MWPM will take several hours to run on a typical desktop/laptop CPU.

### Optimization of perturbation standard deviation

The ensembling method requires a standard deviation as hyper-parameter, which controls
the sampled perturbations of the matching graph. To run the optimization procedure for a uniform
bit-flip noise model and the toric code, run
```
python3 opt_pert_uniform_scale.py -p 4 -o opt_uniform.csv
```
For non-uniform bit-flip noise:
```
python3 opt_pert_nonuniform_scale.py -p 4 -o opt_nonuniform.csv
```
The command line parameters have the following meaning:
```
  -o,--out: Output file
    (default: 'opt_uniform.csv')
  -p,--procs: No. of processes
    (default: '8')
    (an integer)
```
The resulting `*.csv` will be written to `results/opt`, the scripts create the directories, if they
don't already exist. These files can then be consumed by the result processing in **TODO**.


### Vanilla MWPM vs. ensembling

To reproduce the simulations with uniform bit-flip noise and all three QEC codes in [1], run the
script `uniform_bitflip.py` with the following parameters:
```
uniform_bitflip.py:
  -c,--code: <toric|planar|rotated>: QEC code
    (default: 'toric')
  -s,--num_shots: number of shots for each trial
    (default: '1000')
    (an integer)
  -p,--num_variations: number of perturbations for each trial
    (default: '100')
    (an integer)
  -n,--num_workers: number of worker processes
    (default: '8')
    (an integer)
```
Each run will produce a `*.csv` file in `results/` that can be consumed by the post processing setup
found [here](https://github.com/dfki-ric-quantum/qec_pf_post_processing).

The experiments with non-uniform bit-flip noise and the toric code can be reproduced with
`nonuniform_bitflip.py`:
```
nonuniform_bitflip.py:
  -s,--num_shots: number of shots for each trial
    (default: '1000')
    (an integer)
  -p,--num_variations: number of permutations/perturbations for each trial
    (default: '100')
    (an integer)
  -n,--num_workers: number of worker processes
    (default: '8')
    (an integer)
```
which will also write a `*.csv` file to `results/`.


## License

Licensed under the BSD 3-clause license, see `LICENSE` for details.

## Acknowledgments

This work was funded by the German Ministry of Economic Affairs and Climate Action (BMWK) and the
German Aerospace Center (DLR) in project QuDA-KI under grant no. 50RA2206 and through a
Leverhulme-Peierls Fellowship at the University of Oxford, funded by grant no. LIP-2020-014.

## References

[1] Wichette, L., Hohenfeld, H., Mounzer, E., & Grans-Samuelsson, L. (2025). *A partition function
framework for estimating logical error curves in stabilizer codes*.  arXiv preprint
[arXiv:2505.15758](https://arxiv.org/abs/2505.15758).
