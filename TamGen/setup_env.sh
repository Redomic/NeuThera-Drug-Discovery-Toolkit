## You need to first give a name of your environment

conda install pytorch==2.3.0 torchvision torchaudio -c pytorch

pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

conda install -c conda-forge rdkit  

pip install tensorboardX einops ipykernel pandas  
python -m pip install -e .[chem]
pip install --upgrade --force-reinstall numpy==1.26.4

conda install -c conda-forge dm-tree

## Legacy Support of Development Environment
 
# As our project has been in development for over two years, the PyTorch ecosystem has undergone significant changes. To facilitate replication and extension of our work, we are providing the details of our original development environment as a point of reference.

# To restore the original development environment used in our project, please execute the commands listed below:
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
# conda install pyg -c pyg -y
# conda install pytorch-cluster -c pyg -y
# pip install rdkit-pypi
# pip install tensorboardX
# pip install einops
# pip install ipykernel

# python -m pip install -e .[chem]
