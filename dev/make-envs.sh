#!/bin/bash
echo "Activating base environment"
eval "$(conda shell.bash hook)"
conda activate base

message(){
	let "n = ${#1} + 4"
	echo ""
	for i in $(seq $n); do echo -n "#"; done
	echo ""
	echo -n "# "
	echo -n "$1"
	echo -n " #"
	echo ""
	for i in $(seq $n); do echo -n "#"; done
	echo ""
	echo ""
}

build_rmfem(){
	message "CREATING rmfem ENVIRONMENT"
	conda env create -f ../ENVIRONMENT.yml

	message "ADDING LOCAL PATHS"
	conda activate rmfem
	conda develop ~/Storage/git/rmfem
	conda deactivate
}

build_rmfem_dev(){
	message "CREATING rmfem-dev ENVIRONMENT"
	conda env create -f ENVIRONMENT-dev.yml -y

	message "UPDATING rmfem-dev ENVIRONMENT"
	conda env update --name rmfem-dev --file ~/Storage/git/myjive/ENVIRONMENT.yml

	message "ADDING LOCAL PATHS"
	conda activate rmfem-dev
	conda develop ~/Storage/git/rmfem
	conda develop ~/Storage/git/myjive
	conda deactivate
}

# (re)build rmfem environment
if conda env list | grep -q "^rmfem "; then
	while true; do
		read -p "rmfem environment already exists
Do you want to rebuild it? [Y/n] " yn
		case $yn in
			[Yy]* )
				message "REMOVING rmfem ENVIRONMENT"
				conda remove --name rmfem --all -y
				build_rmfem
				break
				;;
			[Nn]* )
				echo "Skipping rmfem build"
				break
				;;
			* )
				echo "Please answer yes or no."
				;;
		esac
	done
else
	build_rmfem
fi


# (re)build rmfem-dev environment
if conda env list | grep -q "^rmfem-dev "; then
	while true; do
		read -p "rmfem-dev environment already exists.
Do you want to rebuild it? [Y/n] " yn
		case $yn in
			[Yy]* )
				message "REMOVING rmfem-dev ENVIRONMENT"
				conda remove --name rmfem-dev --all -y
				build_rmfem_dev
				break
				;;
			[Nn]* )
				echo "Skipping rmfem-dev build"
				break
				;;
			* )
				echo "Please answer yes or no."
				;;
		esac
	done
else
	build_rmfem_dev
fi

