# RMFEM
Code to reproduce the Random Mesh FEM proposed by Abdulle and Garegnane [here](https://doi.org/10.1016/j.cma.2021.113961).

## Getting started
The RMFEM code relies on FEM code provided by [MyJive](https://gitlab.tudelft.nl/apoot1/myjive). Since this is all in-house code, it is not available on PyPI or an anaconda channel. Instead, use git to clone both the MyJive and RMFEM repositories. Then, you can use anaconda to handle the dependencies of BFEM as follows:

```
conda env create -f ENVIRONMENT.yml
conda activate rmfem
conda develop /path/to/rmfem/
```

This should be sufficient to handle all dependencies of the RMFEM code.
