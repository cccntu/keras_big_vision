# %%
###### Cache utils ######
import hashlib
import os
from urllib.parse import urlparse

import requests


def generate_readable_unique_filename(url):
    """Generate a readable and unique filename using the original name and a hash of the URL."""
    # this is a bit overkill
    # original_name = urlparse(url).path.lstrip('/').replace('/', '__')
    original_name = os.path.basename(url)

    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]  # taking only first 8 chars for brevity
    return f"{url_hash}_{original_name}"


def download_public_blob(url, destination_folder="/tmp"):
    """Downloads a public blob from the URL.
    Supports both http and gs URLs.
    """

    # convert url if url starts with gs://
    if url.startswith("gs://"):
        url = url.replace("gs://", "https://storage.googleapis.com/", 1)

    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Generate a readable and unique filename for caching
    unique_filename = generate_readable_unique_filename(url)
    destination_file_path = os.path.join(destination_folder, unique_filename)

    # Check if the file already exists (cache hit)
    if os.path.exists(destination_file_path):
        print(f"Cache hit: {url} -> {destination_file_path}")
        return destination_file_path

    # Temporary file path for downloading
    temp_file_path = destination_file_path + ".temp"

    # Download the blob to a temporary local file (cache miss)
    response = requests.get(url)
    if response.status_code == 200:
        with open(temp_file_path, "wb") as f:
            f.write(response.content)

        # Rename the temporary file to the intended name after successful download
        os.rename(temp_file_path, destination_file_path)

        print(f"Blob downloaded to {destination_file_path}")
        return destination_file_path
    else:
        print(f"Failed to download blob: HTTP {response.status_code}")

        # Remove any partially downloaded temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        return None


# Usage: both of these work
# public_url = 'https://storage.googleapis.com/big_vision/siglip/webli_en_l16_384_63634585.npz'
# public_url = 'gs://big_vision/siglip/webli_en_l16_384_63634585.npz'
# cached_path = download_public_blob(public_url)
# print(f"Cached path: {cached_path}")
# %%

# %%
# This file is heavily copied from the demo notebook and related code the repo
# https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP_demo.ipynb
import sys
from pathlib import Path

# big_vision is not pip-installable, so we add it to the path

current_dir = Path(__file__).parent.absolute()
big_vision_dir = current_dir / "big_vision"
sys.path.append(str(big_vision_dir))  # change to path to big_vision

import big_vision.pp.builder as pp_builder
import big_vision.pp.ops_general
import big_vision.pp.ops_image
import big_vision.pp.ops_text
import numpy as np
import PIL

# %%


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "G": 1664,
            "e": 1792,
        }[v],
        "depth": {
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "G": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "G": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "G": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }


# decode_variant('L/16')
# %%
VARIANTS = {
    ("B/16", 224): ("webli_en_b16_224_63724782.npz", "B", 768, 64, 32_000),
    ("B/16", 256): ("webli_en_b16_256_60500360.npz", "B", 768, 64, 32_000),
    ("B/16", 384): ("webli_en_b16_384_68578854.npz", "B", 768, 64, 32_000),
    ("B/16", 512): ("webli_en_b16_512_68580893.npz", "B", 768, 64, 32_000),
    ("L/16", 256): ("webli_en_l16_256_60552751.npz", "L", 1024, 64, 32_000),
    ("L/16", 384): ("webli_en_l16_384_63634585.npz", "L", 1024, 64, 32_000),
    ("So400m/14", 224): (
        "webli_en_so400m_224_57633886.npz",
        "So400m",
        1152,
        16,
        32_000,
    ),
    ("So400m/14", 384): (
        "webli_en_so400m_384_58765454.npz",
        "So400m",
        1152,
        64,
        32_000,
    ),
    ("B/16-i18n", 256): ("webli_i18n_b16_256_66117334.npz", "B", 768, 64, 250_000),
}


def load_original_model(VARIANT, RES, load_model=True):
    # then we try to load the weights from the original model
    # Pick your hero: (WHEN CHANGING THIS, RERUN IMAGE/TEXT EMBEDDING CELLS)
    # Give this cell 1-3mins.
    if (VARIANT, RES) == (None, None):
        # VARIANT, RES = 'B/16', 224
        # VARIANT, RES = 'B/16', 256
        # VARIANT, RES = 'B/16', 384
        # VARIANT, RES = 'B/16', 512
        # VARIANT, RES = 'L/16', 256
        VARIANT, RES = "L/16", 384
        # VARIANT, RES = 'So400m/14', 224
        # VARIANT, RES = 'So400m/14', 384
        # VARIANT, RES = 'B/16-i18n', 256

    CKPT, TXTVARIANT, EMBDIM, SEQLEN, VOCAB = VARIANTS[VARIANT, RES]

    # It is significantly faster to first copy the checkpoint (30s vs 8m30 for B and 1m vs ??? for L)
    # this line below is commented out so this file can be imported, uncomment it if you want to run this in notebook mode
    #!test -f /tmp/{CKPT} || gsutil cp gs://big_vision/siglip/{CKPT} /tmp/
    #!test -f /tmp/{CKPT} || gsutil cp gs://big_vision/siglip/webli_en_l16_384_63634585.npz /tmp/
    cached_path = download_public_blob(f"gs://big_vision/siglip/{CKPT}")

    if VARIANT.endswith("-i18n"):
        VARIANT = VARIANT[: -len("-i18n")]

    import big_vision.models.proj.image_text.two_towers as model_mod
    import ml_collections

    model_cfg = ml_collections.ConfigDict()
    model_cfg.image_model = "vit"  # TODO(lbeyer): remove later, default
    model_cfg.text_model = "proj.image_text.text_transformer"  # TODO(lbeyer): remove later, default
    model_cfg.image = dict(variant=VARIANT, pool_type="map")
    model_cfg.text = dict(variant=TXTVARIANT, vocab_size=VOCAB)
    model_cfg.out_dim = (None, EMBDIM)  # (image_out_dim, text_out_dim)
    model_cfg.bias_init = -10.0
    model_cfg.temperature_init = 10.0

    model = model_mod.Model(**model_cfg)

    # Using `init_params` is slower but will lead to `load` below performing sanity-checks.
    # init_params = jax.jit(model.init, backend="cpu")(jax.random.PRNGKey(42), jnp.zeros([1, RES, RES, 3], jnp.float32), jnp.zeros([1, SEQLEN], jnp.int32))['params']
    init_params = None  # Faster but bypasses loading sanity-checks.

    if load_model:
        params = model_mod.load(init_params, cached_path, model_cfg)
    else:
        params = None
    return (
        model,
        model_cfg,
        params,
        cached_path,
        CKPT,
        TXTVARIANT,
        EMBDIM,
        SEQLEN,
        VOCAB,
    )


# %%
# model, model_cfg, params, CKPT, TXTVARIANT, EMBDIM, SEQLEN, VOCAB = load_original_model('L/16', 384)
# download all models
# for VARIANT, RES in VARIANTS:
#    load_original_model(VARIANT, RES)
# %%


def load_demo_imgs(RES):
    """Loads demo images (& preprocess)."""

    """
    !wget -q https://cdn.openai.com/multimodal-neurons/assets/apple/apple-ipod.jpg
    !wget -q https://cdn.openai.com/multimodal-neurons/assets/apple/apple-blank.jpg
    !wget -q 'https://images.unsplash.com/photo-1566467021888-b03548769dd1?ixlib=rb-4.0.3&q=85&fm=jpg&crop=entropy&cs=srgb&dl=svetlana-gumerova-hQHm2D1fH70-unsplash.jpg&w=640' -O cold_drink.jpg
    !wget -q 'https://images.rawpixel.com/image_1300/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdXB3azU4ODU5NzY1LXdpa2ltZWRpYS1pbWFnZS1rb3diMmhkeC5qcGc.jpg' -O hot_drink.jpg
    !wget -q https://storage.googleapis.com/big_vision/siglip/authors.jpg
    !wget -q https://storage.googleapis.com/big_vision/siglip/siglip.jpg
    !wget -q https://storage.googleapis.com/big_vision/siglip/caffeine.jpg
    !wget -q https://storage.googleapis.com/big_vision/siglip/robosign.jpg
    !wget -q https://storage.googleapis.com/big_vision/siglip/fried_fish.jpeg
    !wget -q 'https://pbs.twimg.com/media/FTyEyxyXsAAyKPc?format=jpg&name=small' -O cow_beach.jpg
    !wget -q 'https://storage.googleapis.com/big_vision/siglip/cow_beach2.jpg' -O cow_beach2.jpg
    !wget -q 'https://pbs.twimg.com/media/Frb6NIEXwAA8-fI?format=jpg&name=medium' -O mountain_view.jpg

    !mkdir -p imgs
    !mv *.jpg imgs
    !mv *.jpeg imgs
    """

    imgdir = Path("imgs")
    if not imgdir.exists():
        raise RuntimeError(f"Please download the demo images to {imgdir} first. see {__file__}")
    images = [
        PIL.Image.open(imgdir / fname)
        for fname in (
            "apple-ipod.jpg",
            "apple-blank.jpg",
            "cold_drink.jpg",
            "hot_drink.jpg",
            "caffeine.jpg",
            "siglip.jpg",
            "authors.jpg",
            "robosign.jpg",
            "cow_beach.jpg",
            "cow_beach2.jpg",
            "mountain_view.jpg",
        )
    ]

    pp_img = pp_builder.get_preprocess_fn(f"resize({RES})|value_range(-1, 1)")
    imgs = np.array([pp_img({"image": np.array(image)})["image"] for image in images])
    return imgs


# imgs = load_demo_imgs(RES=384)
# %%
# zimg, _, out = model.apply({'params': params}, imgs, None)
# %%
# @title Tokenize and embed texts
def load_demo_txts(SEQLEN, VOCAB):
    texts = [
        "an apple",
        "a picture of an apple",
        "an ipod",
        "granny smith",
        'an apple with a note saying "ipod"',
        "a cold drink on a hot day",
        "a hot drink on a cold day",
        "a photo of a cold drink on a hot day",
        "a photo of a hot drink on a cold day",
        #
        "a photo of two guys in need of caffeine",
        "a photo of two guys in need of water",
        "a photo of the SigLIP authors",
        "a photo of a rock band",
        "a photo of researchers at Google Brain",
        "a photo of researchers at OpenAI",
        #
        "a robot on a sign",
        "a photo of a robot on a sign",
        "an empty street",
        "autumn in Toronto",
        "a photo of autumn in Toronto",
        "a photo of Toronto in autumn",
        "a photo of Toronto in summer",
        "autumn in Singapore",
        #
        "cow",
        "a cow in a tuxedo",
        "a cow on the beach",
        "a cow in the prairie",
        #
        "the real mountain view",
        "Zürich",
        "San Francisco",
        "a picture of a laptop with the lockscreen on, a cup of cappucino, salt and pepper grinders. The view through the window reveals lake Zürich and the Alps in the background of the city.",
    ]

    TOKENIZERS = {
        32_000: "c4_en",
        250_000: "mc4",
    }
    pp_txt = pp_builder.get_preprocess_fn(
        f'tokenize(max_len={SEQLEN}, model="{TOKENIZERS[VOCAB]}", eos="sticky", pad_value=1, inkey="text")'
    )
    txts = np.array([pp_txt({"text": text})["labels"] for text in texts])
    return txts, texts


# txts, texts = load_demo_txts(SEQLEN=SEQLEN, VOCAB=VOCAB)
# %%
# _, ztxt, out = model.apply({'params': params}, None, txts)
# print(txts.shape, ztxt.shape)
# %%
"""
import jax
# This is how to get all probabilities:
print(f"Learned temperature {out['t'].item():.1f}, learned bias: {out['b'].item():.1f}")
probs = jax.nn.sigmoid(zimg @ ztxt.T * out['t'] + out['b'])
print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")
"""
