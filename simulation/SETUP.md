# Setup Instructions

## Set Up Conda Environment

```bash
# Create and activate conda environment
conda create -n clip-rt python=3.10 -y
conda activate clip-rt

# Install PyTorch
pip install -U pip
pip install open_clip_torch

pip install -r simulation/requirements.txt 