#! /bin/bash 

echo "Starting Installation"


# echo "Colab? y/n:"  
# read colab

# Linux
#wget=/usr/bin/wget
WORK_DIR  = ./

map_expansion = "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/nuScenes-map-expansion-v1.3.zip?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=sVge3r41w66xRxJlbyNaMR7hUXE%3D&Expires=1664004309"
mini_nuscenes = "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-mini.tgz?AWSAccessKeyId=AKIA6RIK4RRMFUKM7AM2&Signature=4NjqwjxAAS8rNloDRjXfB%2Fg3fbs%3D&Expires=1664004394"

echo "Downloading map-expansion"
wget -O "nuScenes-map-expansion-v1.3.zip" map_expansion
echo "Downloading mini-dataset"
wget -O "v1.0-mini.tgz" mini_nuscenes

echo "unziping mini-dataset"
tar zxvf v1.0-mini.tgz -C ./data/nuscenes 
echo "unziping map-expansion"
unzip -d ./data/nuscenes/maps/ nuScenes-map-expansion-v1.3.zip

if colab == "y";
then
python -m venv ./.venv
source ./venv/bin/activate
fi
echo "starting env installation"
pip uninstall -y torch torchvision torchaudio torchtext resampy tensorflow jax
pip install wheels
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

python setup.py develop

python /content/EMT_BEV/tools/create_data.py nuscenes --root-path /content/EMT_BEV/data/nuscenes --out-dir /content/EMT_BEV/data/nuscenes --extra-tag nuscenes --version v1.0-mini
