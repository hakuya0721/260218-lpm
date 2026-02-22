
ImportError: libOpenGL.so.0:报错：
    apt-get install -y libopengl0 libgl1 libegl1
ImportError: libEGL.so.1: cannot open shared object file: No such file or directory报错：
    apt-get install -y libegl1 libgl1 libglib2.0-0

export DATA_DIR='/root/OVSegDT/data'
export MATTERPORT_TOKEN_ID='2c13021de10bd9ba'
export MATTERPORT_TOKEN_SECRET='1dc3c27504ea13981b90d3f8f1d6a86e'

安装环境：
/root/OVSegDT/habitat-lab/habitat-lab/requirements.txt  requirements.txt中添加对应的

conda create -n ovon python=3.8 cmake=3.14.0 -y
conda activate ovon

conda install -n ovon habitat-sim=0.2.3 headless -c conda-forge -c aihabitat -y

conda install -n ovon pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia -c conda-forge -y
conda install -c conda-forge gxx=9
conda install nvidia/label/cuda-11.8.0::cuda-nvcc
conda install -c nvidia cuda-toolkit==11.8

echo $CONDA_PREFIX读取env路径

export CUDA_HOME=‘路径’

pip install -e .

git clone https://github.com/naokiyokoyama/frontier_exploration/tree/a8890d68cfa0d10254238abe9266a76856cb1f17
cd frontier_exploration && pip install -e . && cd ..
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
pip install ftfy regex tqdm GPUtil trimesh seaborn timm scikit-learn einops transformers
这里我之前用了一下--no-dep 然后再运行全部的，担心会替换torch版本
pip install git+https://github.com/openai/CLIP.git

pip install --extra-index-url https://miropsota.github.io/torch_packages_builder \
  "detectron2==0.6+pt2.2.1cu118"
这样拉区wheel或者pip install wheels/detectron2-0.6+pt2.2.1cu118-cp38-cp38-linux_x86_64.whl
python -c "import detectron2; import detectron2._C; print('detectron2 ok', detectron2.__version__)"测试
会有报错libGL.so.1: cannot open shared object file: No such file or directory
apt-get install -y libgl1 libglib2.0-0

pip install ultralytics

cp modeling_llama.py <<PATH_TO_YOUR_CONDA_ENV>/lib/python3.8/site-packages/transformers/models/llama/modeling_llama.py



conda update conda
conda install -n base -c defaults conda=24.11.1 -y
conda create -n ovon python=3.8 cmake=3.14.0 -y
conda activate ovon
conda install -n ovon habitat-sim=0.2.3 headless -c conda-forge -c aihabitat -y
conda install -n ovon pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia -c conda-forge -y
echo $CONDA_PREFIX
export CUDA_HOME='/root/miniconda3/envs/ovon'
git clone https://github.com/hakuya0721/2026-02-16-yize.git
cd 2026-02-16-yize/
pip install -e .
git clone https://github.com/naokiyokoyama/frontier_exploration.git
cd frontier_exploration && pip install -e . && cd .. 
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
pip install ftfy regex tqdm GPUtil trimesh seaborn timm scikit-learn einops transformers
pip install git+https://github.com/openai/CLIP.git
cd ..
pip install wheels/wheels/detectron2-0.6+pt2.2.1cu118-cp38-cp38-linux_x86_64.whl
pip install ultralytics
apt-get update
apt-get install -y libegl1 libgl1 libglx-mesa0 libgles2-mesa
ldconfig -p | grep -E "libEGL\.so\.1"
python -c "import habitat_sim; print('habitat_sim import ok')"
pip install ifcfg
pip install lmdb
pip install webdataset==0.1.103
pip install faster_fifo

scp -rP 37518 C:\Users\16545\Downloads\OVSegDT\data root@connect.westc.gpuhub.com:/root/autodl-tmp

python -m ovon.run --run-type train \
--debug-datapath \
--exp-config config/experiments/transformer_dagger_ppo_segm_loss.yaml

export DATA_DIR='/root/autodl-tmp/data'
export MATTERPORT_TOKEN_ID='2c13021de10bd9ba'
export MATTERPORT_TOKEN_SECRET='1dc3c27504ea13981b90d3f8f1d6a86e'
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_train_v0.2 \
  --data-path $DATA_DIR &&
python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_val_v0.2 \
  --data-path $DATA_DIR


mv /root/2026-02-16-yize/data /root/2026-02-16-yize/data.bak.$(date +%F_%H%M%S)
ln -s /root/autodl-tmp/data /root/2026-02-16-yize/data
ls -l /root/2026-02-16-yize | grep data
readlink -f /root/2026-02-16-yize/data
成功后你会看到类似：

lrwxrwxrwx ... data -> /root/autodl-tmp/data

readlink -f 输出 /root/autodl-tmp/data

 export HF_ENDPOINT=https://hf-mirror.com