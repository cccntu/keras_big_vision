# Keras SigLIP

This is an implementation of the [SigLIP](https://arxiv.org/abs/2303.15343) model in Keras (keras 3/keras core)
Based on the original [Jax/Flax implementation](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP_demo.ipynb)



## Setup
```
git clone https://github.com/cccntu/keras_big_vision
cd keras_big_vision
git submodule update --init --recursive

pip install -r big_vision/big_vision/requirements.txt
pip install keras-core
# pip install torch torchvision torchaudio
# pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# seems jax is more sensitive to cudnn version, I had to reinstall jax after installing torch to fix it
```

## Usage

This repo is a work in progress. Current there are
- nb_convert_weight.py: converts the original weights to keras weights
- notebook.py: shows how to instantiate the model, load keras weights, and run inference


# TODO
- [ ] upload converted weights & add code to pull converted weights
- [ ] cleanup

```
git submodule add
```