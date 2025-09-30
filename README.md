# ProbFEM
This repository is used to investigate various probabilistic formations of the finite element method that have been proposed over the years.

## Getting started
The ProbFEM code relies on FEM code provided by [MyJive](https://gitlab.tudelft.nl/apoot1/myjive), as well as the original C++-based jem/jive libraries.
Using the docker container defined by the Dockerfile in the repo is recommended, to take care of all C++ and Python dependencies, and to ensure the two are linked together.
Note that because we have a lot of dependencies, the container takes up quite a bit of space (a couple of GB), and takes quite some time to build (around 15 minutes):
```
# build the docker container
cd /path/to/probfem/
docker build --target jupyter -t probfem-jupyter .

# run tests
docker run probfem-jupyter pytest

# run Juptyer notebooks
docker run -p 8888:8888 probfem-jupyter
```

## Reproduction
The repository contains code to reproduce results of various papers, all of which are located under `experiments/reproduction/`.
Papers for which reproduction code is available:

-`experiments/reproduction/bfem` reproduces all figures from "A Bayesian Approach to Modeling Finite Element Discretization Error" by Poot, Rocha, Kerfriden and Van der Meer (2024), found [here](https://doi.org/10.1007/s11222-024-10463-z).
- `experiments/reproduction/inverse` reproduces all figures from "The Bayesian Finite Element Method in Inverse Problems: a Critical Comparison between Probabilistic Models for Discretization Error" by Poot, Rocha, Kerfriden and Van der Meer (2025), found [here](https://doi.org/10.48550/arXiv.2506.02815).
- `experiments/reproduction/probnum25` reproduces all figures from "Effects of Interpolation Error and Bias on the Random Mesh Finite Element Method for Inverse Problems" by Poot, Rocha, Kerfriden and Van der Meer (2025), found [here](https://doi.org/10.48550/arXiv.2504.03393).
- `experiments/reproduction/rmfem` reproduces figures 2, 3 and 10 from "A probabilistic finite element method based on random meshes: A posteriori error estimators and Bayesian inverse problems" by Garegnani and Abdulle (2021), found [here](https://doi.org/10.1007/s11222-024-10463-z).