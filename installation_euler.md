# AGILE3D Setup and Evaluation Guide (Euler Cluster)

This guide provides step-by-step instructions to set up and run the [AGILE3D](https://github.com/ywyue/AGILE3D) project on the Euler cluster.

## 📦 Clone the Repository

```bash
cd /cluster/scratch/plukovic
git clone https://github.com/ywyue/AGILE3D.git
cd AGILE3D
```

## ⚙️ Load Required Modules

```bash
# maybe needed for later
module load stack/2024-06 gcc/12.2.0 cuda/11.8.0 eth_proxy

# use this to install
module load stack/2024-04 gcc/8.5.0 cuda/11.8.0 eth_proxy
```

## 🐍 Create and Configure Conda Environment

```bash
conda create -n agile3d python=3.10 pip
conda activate agile3d
pip install pip==22.3
```

## 🔥 Install PyTorch with CUDA 11.8

```bash
# euler
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

# local
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

## 🧪 Interactive GPU Session

Launch an interactive GPU session:

```bash
srun --partition=gpu --gpus=rtx_3090:1 --cpus-per-task=1 --mem-per-cpu=256G --time=12:00:00 --pty bash
```

## 🛠️ Install Dependencies

```bash
pip install ninja cmake
pip install setuptools==59.5.0
conda install openblas-devel -c anaconda
conda install -c conda-forge openblas
pip install numpy==1.26.4
```

## 🔍 Locate OpenBLAS Library

```bash
find /cluster/scratch/plukovic -name "libopenblas.so"
```

Set environment variables:

```bash
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## ⚙️ Install MinkowskiEngine

```bash
# euler
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
  --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
  --install-option="--blas=openblas"

# local
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

```

## ➕ Install Additional Packages

```bash
pip install open3d wandb h5py segment_anything imageio pypng scikit-learn
conda install opencv
```

## 🧪 Run Evaluation on ScanNet40

Launch a longer GPU job for evaluation:

```bash

sbatch --partition=gpu --gpus=rtx_3090:1 --cpus-per-task=1 --mem-per-cpu=256G --time=72:00:00 ./scripts/eval_single_scannet40_euler.sh

srun --time=72:00:00 --cpus-per-task=1 --mem-per-cpu=512g ./scripts/parallel_download_scannet.sh --val

rsync -av --include '*/' --include '*.ply' --exclude '*' /home/plukovic/interactive_segmentation/AGILE3D-SAM/data/scannet/scans/ plukovic@euler:/cluster/scratch/plukovic/scannet/scannet_v2/scans/

scp euler:/cluster/scratch/plukovic/scannet/scannet_v2/results/* /home/plukovic/interactive_segmentation/AGILE3D-SAM/data/scannet/results/

scp /home/plukovic/interactive_segmentation/AGILE3D-SAM/weights/* euler:/cluster/home/plukovic/AGILE3D-SAM/weights/

```




