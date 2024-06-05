#!/bin/bash

cd m2bo_new
sudo apt update
sudo apt install g++
sudo apt-get upgrade libstdc++6
sudo apt-get dist-upgrade
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3

wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210_linux.tar.gz
mkdir ~/.mujoco
tar -zxvf mujoco210_linux.tar.gz -C ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

conda env create -f environment.yml
conda activate off-moo
pip install -r requirements.txt
pip install torchvision==0.11.1
pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
cd m2bo_bench/problem/lambo
pip install -e .
cd ../../../
pip install scipy==1.10.1
pip install scikit-learn==0.21.3
pip install --upgrade pandas
pip install --upgrade kiwisolver

# Sklearn==0.21.3 might be wrong in some scenarios, thus we fix bugs with the scripts below.
# Please set your path to conda here
bash fix_contents.sh ${YOUR_PATH_TO_CONDA}/envs/off-moo/lib/python3.8/site-packages/sklearn/cross_decomposition/pls_.py "pinv2" "pinv"

# Make sure that you have paste your Google Drive API below
curl -H "Authorization: Bearer <Your Google Drive APIs>" https://drive.google.com/file/d/11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO/view?usp=sharing -o database.zip
curl -H "Authorization: Bearer <Your Google Drive APIs>" https://drive.google.com/file/d/1r0iSCq1gLFs5xnmp1MDiqcqxNcY5q6Hp/view?usp=sharing -o data.zip

unzip database.zip -d m2bo_bench/problems/mo_nas/
unzip data.zip -d m2bo_bench/problems/mo_nas/

conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia

python config_evoxbench.py 
python tests/test_mujoco.py
python tests/test_env.py
python tests/test_vallina_modules.py



