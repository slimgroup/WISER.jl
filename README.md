<h1 align="center">WISER: Multimodal variational inference for full-waveform inversion without dimensionality reduction</h1>

[![][license-img]][license-status] [![](https://img.shields.io/badge/DOI-10.1190/geo2024--0483.1-blue)](https://doi.org/10.1190/geo2024-0483.1)

Code to reproduce results in Ziyi Yin, Rafael Orozco, Felix J. Herrmann, "[WISER: Multimodal variational inference for full-waveform inversion without dimensionality reduction](https://doi.org/10.1190/geo2024-0483.1)", published in Geophysics.

WISER is an extension to [WISE](https://doi.org/10.1190/geo2023-0744.1), which is also published in Geophysics.

## Software descriptions

All of the software packages used in this paper are fully *open source, scalable, interoperable, and differentiable*. The readers are welcome to learn about our software design principles from [this open-access article](https://library.seg.org/doi/10.1190/tle42070474.1).

#### Wave modeling

We use [JUDI.jl](https://github.com/slimgroup/JUDI.jl) for wave modeling and inversion, which calls the highly optimized propagators of [Devito](https://www.devitoproject.org/).

#### Conditional normalizing flows

We use [InvertibleNetworks.jl] to train the conditional normalizing flows (CNFs). This package implements memory-efficient invertible networks via hand-written derivatives. This ensures that these invertible networks are scalable to realistic 3D problems.

## Installation

First, install [Julia](https://julialang.org/) and [Python](https://www.python.org/). The scripts will contain package installation commands at the beginning so the packages used in the experiments will be automatically installed.

## Scripts

[wiser.jl](scripts/wiser.jl) runs the WISER algorithm in the paper to perform physics-based latent space correction.

The script [utils.jl](scripts/utils.jl) parses the input as keywords for each experiment.

The following keyword arguments can be used to reproduce the results in the WISER paper:

- Case 1: `julia wiser.jl --lr_wiser=0.004`
- Case 2: `julia wiser.jl --test_snr=0.0 --amplitude=0.2 --lambda=10.0 --lr_pre=0.0004`

## LICENSE

The software used in this repository can be modified and redistributed according to [MIT license](LICENSE).

## Reference

If you use our software for your research, we appreciate it if you cite us following the bibtex in [CITATION.bib](CITATION.bib).

## Authors

This repository is written by [Ziyi Yin] and [Rafael Orozco] from the [Seismic Laboratory for Imaging and Modeling] (SLIM) at the Georgia Institute of Technology.

If you have any question, we welcome your contributions to our software by opening issue or pull request.

SLIM Group @ Georgia Institute of Technology, [https://slim.gatech.edu](https://slim.gatech.edu/).      
SLIM public GitHub account, [https://github.com/slimgroup](https://github.com/slimgroup).    

[license-status]:LICENSE
[license-img]:http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat?style=plastic
[Seismic Laboratory for Imaging and Modeling]:https://slim.gatech.edu/
[InvertibleNetworks.jl]:https://github.com/slimgroup/InvertibleNetworks.jl
[Ziyi Yin]:https://ziyiyin97.github.io/
[Rafael Orozco]:https://slim.gatech.edu/people/rafael-orozco