DOT:= .
.PHONY: all clean
all: virtualenv
virtualenv:
	./setup_anaconda.sh venv dgmix_env 3.9
	. ./activate_python.sh; conda update -y -n base -c defaults conda
	. ./activate_python.sh; conda update -y -n base -c conda-forge conda
	. ./activate_python.sh; cd ../; pip install -e .
	. ./activate_python.sh; conda install -y ipython ipykernel numpy pandas scikit-learn matplotlib librosa timm pyyaml tensorboard black flake8 umap-learn tabulate -c conda-forge
	. ./activate_python.sh; conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
clean:
	rm -fr venv miniconda.sh activate_python.sh
	find -iname "*.pyc" -delete