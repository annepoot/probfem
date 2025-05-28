#! /bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate probfem
# conda develop /home/apoot1/git/probfem
# conda develop /home/apoot1/git/myjive
export PYTHONPATH="/home/apoot1/git/probfem:$PYTHONPATH"
export PYTHONPATH="/home/apoot1/git/myjive:$PYTHONPATH"
python inverse_fem_blue.py $1 $2

