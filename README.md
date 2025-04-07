# ProbFEM
This repository is used to investigate various probabilistic formations of the finite element method that have been proposed over the years.

## Getting started
The ProbFEM code relies on FEM code provided by [MyJive](https://gitlab.tudelft.nl/apoot1/myjive), as well as the original C++-based jem/jive libraries. Because this repo is under development, a few steps are needed to get the code up and running:

```
# build a docker container and run it
docker build -t probfem-container .
docker run -it --rm -v $(pwd):/workspace -w /workspace probfem-container
# install python dependencies with anaconda
conda env create -f ENVIRONMENT.yml
conda activate probfem
# compile c++ backend code
cd fem/jive/src
make
cd ../../..
# verify installation
pytest
```

## Reproduction
The repository contains code to reproduce results of various papers, all of which are located under `experiments/reproduction/`.
Papers for which reproduction code is available:

- `experiments/reproduction/bfem` reproduces all figures from "A Bayesian Approach to Modeling Finite Element Discretization Error" by Poot, Rocha, Kerfriden and Van der Meer (2024), found [here](https://doi.org/10.1007/s11222-024-10463-z).
- `experiments/reproduction/inverse` reproduces all figures from a yet-to-be-published paper comparing BFEM, RM-FEM and statFEM in the context of inverse problems by Poot, Rocha, Kerfriden and Van der Meer (2025).
- `experiments/reproduction/probnum25` reproduces all figures from "Effects of Interpolation Error and Bias on the Random Mesh Finite Element Method for Inverse Problems" by Poot, Rocha, Kerfriden and Van der Meer (2025), found [here](https://doi.org/10.48550/arXiv.2504.03393).
- `experiments/reproduction/rmfem` reproduces figures 2, 3 and 10 from "A probabilistic finite element method based on random meshes: A posteriori error estimators and Bayesian inverse problems" by Garegnani and Abdulle (2021), found [here](https://doi.org/10.1007/s11222-024-10463-z).
