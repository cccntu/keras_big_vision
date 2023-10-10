# %%
import os

os.environ["KERAS_BACKEND"] = "torch"
import torch

os.environ["KERAS_BACKEND"] = "jax"

import gc

import numpy as np
from keras_core import ops

from keras_siglip import TwoTowerModel
from siglip_utils import VARIANTS, load_demo_imgs, load_demo_txts, load_original_model
from weight_conversion_utils import (
    flatten_dict,
    keras_weight_to_nested_dict,
    load_weight,
    load_weight_with_conversion,
    npz_to_state_dict,
)

VARIANT, RES = "L/16", 384
# %%
# we don't need to load the model
(
    model,
    model_cfg,
    params,
    cached_path,
    CKPT,
    TXTVARIANT,
    EMBDIM,
    SEQLEN,
    VOCAB,
) = load_original_model(VARIANT, RES, load_model=False)
# %%
imgs = load_demo_imgs(RES)
txts, texts = load_demo_txts(SEQLEN, VOCAB)
# %%


with torch.no_grad():  # this is important, otherwise I get OOM on 24GB GPU
    keras_model = TwoTowerModel(
        variant=VARIANT,
        text_variant=TXTVARIANT,
        pool_type="map",
        vocab_size=VOCAB,
        out_dim=EMBDIM,
        temperature_init=10.0,
        bias_init=-10.0,
        name="model",
    )
    zimgk, ztxtk, outk = keras_model(imgs, txts)
# %%
zimgk, ztxtk, outk = keras_model(image=None, text=txts)
# %%
# zimgk.shape, ztxtk.shape
# %%
# def convert_weight


# %%
# keras_model.weights
# %%
# if backend is pytorch, then we can use state_dict, but the serialization is different
# keras_model.state_dict()
# so we use the backend-agnostic way to load the original weights
# %%
def load_npz_to_keras_model(npz_path, keras_model):
    pretrained_weights = np.load(npz_path, allow_pickle=True)
    # loading npz into dict takes a few seconds
    pretrained_state_dict = npz_to_state_dict(pretrained_weights)
    # 2. convert keras weights to a nested dict (from .weights)
    state_dict = keras_weight_to_nested_dict(keras_model.weights)
    # 3. map/copy the nested dict
    load_weight_with_conversion(pretrained_state_dict["params"], state_dict["model"])
    # 4. flatten the nested dict back to a flat dict
    flattened = flatten_dict(state_dict)
    # 5. map the flat dict to list of weights and call .set_weights
    load_weight(keras_model, flattened)


# %%
load_npz_to_keras_model(CKPT, keras_model)
# %%
# finally test the model
with torch.no_grad():
    zimgk, ztxtk, outk = keras_model(imgs, txts, training=False)
# %%
outk["t"]
# %%
outk["b"]
# %%
print(f"Learned temperature {outk['t'].item():.1f}, learned bias: {outk['b'].item():.1f}")
probs = ops.nn.sigmoid(zimgk @ ztxtk.T * outk["t"] + outk["b"])
print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")
# the numbers are slightly different from the original model, even if the backend is jax, I don't know why
# but the difference is not big (a few percent)
# %%
save_path = f"{VARIANT.replace('/','-')}:{RES}.weights.h5"

# %%
save_path = CKPT.replace(".npz", ".weights.h5")
keras_model.save_weights(save_path)


# %%
# now let's convert every model
def load_keras_model(VARIANT, RES):
    print(f"######## {VARIANT} {RES} ########")
    (
        model,
        model_cfg,
        params,
        cached_path,
        CKPT,
        TXTVARIANT,
        EMBDIM,
        SEQLEN,
        VOCAB,
    ) = load_original_model(VARIANT, RES, load_model=False)
    if VARIANT.endswith("-i18n"):
        VARIANT = VARIANT[: -len("-i18n")]
    save_path = CKPT.replace(".npz", ".weights.h5")
    if os.path.exists(save_path):
        print(f"Already exists: {save_path}")
    # return
    imgs = load_demo_imgs(RES)
    txts, texts = load_demo_txts(SEQLEN, VOCAB)
    with torch.no_grad():  # this is important, otherwise I get OOM on 24GB GPU
        keras_model = TwoTowerModel(
            variant=VARIANT,
            text_variant=TXTVARIANT,
            pool_type="map",
            vocab_size=VOCAB,
            out_dim=EMBDIM,
            temperature_init=10.0,
            bias_init=-10.0,
            name="model",
        )
        zimgk, ztxtk, outk = keras_model(imgs, txts)

    load_npz_to_keras_model(cached_path, keras_model)
    # sanity check
    with torch.no_grad():
        zimgk, ztxtk, outk = keras_model(imgs, txts, training=False)
    print(f"Learned temperature {outk['t'].item():.1f}, learned bias: {outk['b'].item():.1f}")
    probs = ops.nn.sigmoid(zimgk @ ztxtk.T * outk["t"] + outk["b"])
    print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
    print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")
    # the numbers are slightly different from the original model, even if the backend is jax, I don't know why
    # but the difference is not big (a few percent)

    keras_model.save_weights(save_path)
    print(f"Saved to {save_path}")
    print(f"################")


# %%
for VARIANT, RES in VARIANTS.keys():
    load_keras_model(VARIANT, RES)
    gc.collect()
    torch.cuda.empty_cache()

# %%
VARIANTS.keys()
