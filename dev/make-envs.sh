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

build_probfem(){
	message "CREATING probfem ENVIRONMENT"
	conda env create -f ../ENVIRONMENT.yml

	message "ADDING LOCAL PATHS"
	conda activate probfem
	conda develop ~/Storage/git/probfem
	conda deactivate
}

build_probfem_dev(){
	message "CREATING probfem-dev ENVIRONMENT"
	conda env create -f ENVIRONMENT-dev.yml -y

	message "UPDATING probfem-dev ENVIRONMENT"
	conda env update --name probfem-dev --file ~/Storage/git/myjive/ENVIRONMENT.yml

	message "ADDING LOCAL PATHS"
	conda activate probfem-dev
	conda develop ~/Storage/git/probfem
	conda develop ~/Storage/git/myjive
	conda deactivate
}

# (re)build probfem environment
if conda env list | grep -q "^probfem "; then
	while true; do
		read -p "probfem environment already exists
Do you want to rebuild it? [Y/n] " yn
		case $yn in
			[Yy]* )
				message "REMOVING probfem ENVIRONMENT"
				conda remove --name probfem --all -y
				build_probfem
				break
				;;
			[Nn]* )
				echo "Skipping probfem build"
				break
				;;
			* )
				echo "Please answer yes or no."
				;;
		esac
	done
else
	build_probfem
fi


# (re)build probfem-dev environment
if conda env list | grep -q "^probfem-dev "; then
	while true; do
		read -p "probfem-dev environment already exists.
Do you want to rebuild it? [Y/n] " yn
		case $yn in
			[Yy]* )
				message "REMOVING probfem-dev ENVIRONMENT"
				conda remove --name probfem-dev --all -y
				build_probfem_dev
				break
				;;
			[Nn]* )
				echo "Skipping probfem-dev build"
				break
				;;
			* )
				echo "Please answer yes or no."
				;;
		esac
	done
else
	build_probfem_dev
fi

