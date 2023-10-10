# %%
##########  load keras from converted weights ##########
import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERAS_BACKEND"] = "torch"

from keras_core import ops
from siglip_utils import VARIANTS, load_demo_imgs, load_demo_txts, load_original_model

from keras_siglip import TwoTowerModel

VARIANT, RES = "L/16", 384
VARIANT, RES = "So400m/14", 384
# VARIANT, RES = 'B/16-i18n', 256
CKPT, TXTVARIANT, EMBDIM, SEQLEN, VOCAB = VARIANTS[VARIANT, RES]
if VARIANT.endswith("-i18n"):
    VARIANT = VARIANT[: -len("-i18n")]
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
# build the model
imgs = load_demo_imgs(RES)
txts, texts = load_demo_txts(SEQLEN, VOCAB)

from contextlib import nullcontext

if os.environ["KERAS_BACKEND"] == "torch":
    import torch

    context_manager = torch.no_grad
else:
    context_manager = nullcontext
# %%
with context_manager():
    zimgk, ztxtk, outk = keras_model(imgs, txts)

# load weights
save_path = CKPT.replace(".npz", ".weights.h5")
keras_model.load_weights(save_path)

# inference
with context_manager():
    zimgk, ztxtk, outk = keras_model(imgs, txts, training=False)
print(f"Learned temperature {outk['t'].item():.1f}, learned bias: {outk['b'].item():.1f}")
probs = ops.nn.sigmoid(zimgk @ ztxtk.T * outk["t"] + outk["b"])
print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")
# %%
