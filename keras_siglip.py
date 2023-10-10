# %%
# This implementation is based on the repo: https://github.com/google-research/big_vision
import os
from typing import Optional, Sequence, Union

import keras_core as keras

# Note that keras_core should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import numpy as np
from keras_core import activations, initializers, layers, ops

# os.environ["KERAS_BACKEND"] = "torch"
# os.environ["KERAS_BACKEND"] = "jax"


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


class MlpBlock(keras.layers.Layer):
    def __init__(self, mlp_dim: Optional[int] = None, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, inputs_shape):
        n, l, d = inputs_shape  # pylint: disable=unused-variable
        inits = dict(
            # keras doc: The Glorot normal initializer, also called Xavier normal initializer
            kernel_initializer=initializers.GlorotNormal(),
            bias_initializer=initializers.RandomNormal(stddev=1e-6),
        )
        self.Dense_0 = keras.layers.Dense(self.mlp_dim or 4 * d, name="Dense_0", **inits)
        self.gelu = keras.layers.Activation(activations.gelu)
        self.Dropout_0 = keras.layers.Dropout(rate=self.dropout)
        self.Dense_1 = keras.layers.Dense(d, name="Dense_1", **inits)

    def call(self, inputs, training=None):
        x = self.Dense_0(inputs)
        x = self.gelu(x)
        x = self.Dropout_0(x, training=training)
        x = self.Dense_1(x)
        return x


class Encoder1DBlock(keras.layers.Layer):
    def __init__(
        self,
        mlp_dim: Optional[int] = None,  # Defaults to 4x input dim
        num_heads: int = 12,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout = dropout

    def build(self, inputs_shape):
        n, l, d = inputs_shape  # pylint: disable=unused-variable
        key_dim = d // self.num_heads
        self.LayerNorm_0 = keras.layers.LayerNormalization(name="LayerNorm_0")
        self.MultiHeadDotProductAttention_0 = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            kernel_initializer=initializers.GlorotUniform(),
            name="MultiHeadDotProductAttention_0",
        )
        self.Dropout_0 = keras.layers.Dropout(rate=self.dropout)
        self.LayerNorm_1 = keras.layers.LayerNormalization(name="LayerNorm_1")
        self.MlpBlock_0 = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            name="MlpBlock_0",
        )
        self.Dropout_1 = keras.layers.Dropout(rate=self.dropout)

    def call(self, inputs, training=None):
        out = {}
        x = inputs
        y = self.LayerNorm_0(x)
        y = out["sa"] = self.MultiHeadDotProductAttention_0(y, y)
        y = self.Dropout_0(y, training=training)
        x = out["+sa"] = x + y

        y = self.LayerNorm_1(x)
        y = out["mlp"] = self.MlpBlock_0(y)
        y = self.Dropout_1(y, training=training)
        x = out["+mlp"] = x + y
        return x, out


class Encoder(keras.layers.Layer):
    def __init__(
        self,
        depth: int,
        mlp_dim: Optional[int] = None,  # Defaults to 4x input dim
        num_heads: int = 12,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout = dropout
        # def build(self, inputs_shape):
        #    n, l, d = inputs_shape  # pylint: disable=unused-variable
        self.layers = []
        for lyr in range(self.depth):
            self.layers.append(
                Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
            )
        self.LayerNorm_0 = keras.layers.LayerNormalization(name="encoder_norm")

    def call(self, inputs, training=None):
        out = {}
        x = inputs
        # Input Encoder
        for lyr in range(self.depth):
            block = self.layers[lyr]
            x, out[f"block{lyr:02d}"] = block(x, training=training)
        out["pre_ln"] = x  # Alias for last block, but without the number in it.

        return self.LayerNorm_0(x), out


class MAPHead(keras.layers.Layer):
    def __init__(
        self,
        mlp_dim: Optional[int] = None,  # Defaults to 4x input dim
        num_heads: int = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads

    def build(self, inputs_shape):
        n, l, d = inputs_shape  # pylint: disable=unused-variable
        key_dim = d // self.num_heads
        self.probe = self.add_weight(
            name="probe",
            shape=(1, 1, d),
            initializer=initializers.GlorotUniform(),
            trainable=True,
            # dtype=inputs.dtype,
        )
        self.MultiHeadDotProductAttention_0 = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            kernel_initializer=initializers.GlorotUniform(),
            name="MultiHeadDotProductAttention_0",
        )
        self.LayerNorm_0 = keras.layers.LayerNormalization(name="LayerNorm_0")
        self.MlpBlock_0 = MlpBlock(
            mlp_dim=self.mlp_dim,
            name="MlpBlock_0",
        )

    def call(self, inputs, training=None):
        n, l, d = inputs.shape  # pylint: disable=unused-variable
        probe = self.probe
        probe = ops.tile(probe, [n, 1, 1])

        x = self.MultiHeadDotProductAttention_0(probe, inputs)

        y = self.LayerNorm_0(x)
        x = x + self.MlpBlock_0(y)
        return x[:, 0]


class ViT(keras.layers.Layer):
    def __init__(
        self,
        num_classes: Optional[int] = None,
        patch_size: Sequence[int] = (16, 16),
        width: int = 768,
        depth: int = 12,
        mlp_dim: Optional[int] = None,  # Defaults to 4x input dim
        num_heads: int = 12,
        posemb: str = "learn",  # Can also be "sincos2d"
        rep_size: Union[int, bool] = False,
        dropout: float = 0.0,
        pool_type: str = "gap",  # Can also be "map" or "tok"
        head_zeroinit: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.posemb = posemb
        self.rep_size = rep_size
        self.dropout = dropout
        self.pool_type = pool_type
        self.head_zeroinit = head_zeroinit
        self.Conv_0 = keras.layers.Conv2D(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
        )

        c = width
        if self.pool_type == "tok":
            self.cls = self.add_weight(
                name="cls",
                shape=(1, 1, c),
                initializer=initializers.Zeros(),
                trainable=True,
                # dtype=x.dtype,
            )
        elif self.pool_type == "map":
            self.MAPHead_0 = MAPHead(num_heads=self.num_heads, mlp_dim=self.mlp_dim, name="MAPHead_0")

        self.Dropout_0 = keras.layers.Dropout(rate=self.dropout)
        self.Encoder_0 = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            name="Transformer",
        )
        self.Dropout_1 = keras.layers.Dropout(rate=self.dropout)

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            self.pre_logits = keras.layers.Dense(rep_size, name="pre_logits")

        kw = {"kernel_initializer": initializers.Zeros()} if self.head_zeroinit else {}
        self.head = keras.layers.Dense(self.num_classes, name="head", **kw)

    def build(self, inputs_shape):
        n, W, H, C = inputs_shape
        # manually compute the shape of the sequence after patching (going through the conv layer)
        w, h = W // self.patch_size[0], H // self.patch_size[1]
        c = self.width
        if self.posemb == "learn":
            self.pos_embedding = self.add_weight(
                name="pos_embedding",
                shape=(1, h * w, c),
                initializer=initializers.RandomNormal(stddev=1 / np.sqrt(c)),
                trainable=True,
                # dtype=inputs_shape.dtype,
            )

        elif self.posemb == "sincos2d":
            raise ValueError(f"sincos2d is not implemented in keras yet")
            # self.param = posemb_sincos_2d(*self.seqshape, self.width, dtype=self.dtype)
        else:
            raise ValueError(f"Unknown posemb type: {self.typ}")

    def call(self, inputs, training=None):
        out = {}

        # Patch extraction
        x = out["stem"] = self.Conv_0(inputs)

        n, h, w, c = x.shape
        x = ops.reshape(x, [n, h * w, c])

        # Add posemb before adding extra token.

        x = out["with_posemb"] = x + self.pos_embedding

        if self.pool_type == "tok":
            x = ops.concatenate([ops.tile(self.cls, [n, 1, 1]), x], axis=1)

        n, l, c = x.shape  # pylint: disable=unused-variable
        x = self.Dropout_0(x, training=training)

        x, out["encoder"] = self.Encoder_0(x, training=training)
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = self.MAPHead_0(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = ops.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = ops.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            hid = self.pre_logits
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = ops.tanh(hid(x_2d))
            x = ops.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            head = self.head
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


# %%

from typing import Optional, Tuple, Union


class EmbeddingWithAttend(keras.layers.Embedding):
    """Embedding layer with an additional attend method."""

    # like in flax: https://flax.readthedocs.io/en/latest/_modules/flax/linen/linear.html#Embed
    #     query, embedding = promote_dtype(query, self.embedding, dtype=self.dtype)
    # return jnp.dot(query, embedding.T)
    #
    # keras.layers.Embedding:
    # self.embeddings = self.add_weight(
    #       shape=(self.input_dim, self.output_dim),

    def attend(self, query):
        """Attend to the embedding, given a query."""
        # query, embedding = promote_dtype(query, self.embedding, dtype=self.dtype)
        # return jnp.dot(query, embedding.T)
        return ops.matmul(query, ops.transpose(self.embeddings))


# Note: this is from bigvision
# https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/image_text/text_transformer.py
# this implementation reuses the same ViT code from above
class TextTransformer(keras.layers.Layer):
    def __init__(
        self,
        num_classes: int,
        width: int = 512,
        depth: int = 12,
        mlp_dim: Optional[int] = None,  # Defaults to 4x input dim
        num_heads: int = 12,
        dropout: float = 0.0,
        vocab_size: int = 32_000,
        pool_type: str = "last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.pool_type = pool_type

        self.embedding = EmbeddingWithAttend(input_dim=self.vocab_size, output_dim=self.width, name="Embed_0")

        self.Encoder_0 = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            name="Encoder_0",
        )

        if self.pool_type == "map":
            self.MAPHead_0 = MAPHead(num_heads=self.num_heads, mlp_dim=self.mlp_dim, name="MAPHead_0")

        self.head = keras.layers.Dense(self.num_classes, name="head")

    def build(self, inputs_shape):
        n, l = inputs_shape
        d = self.width
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, l, d),
            initializer=initializers.RandomNormal(stddev=1 / np.sqrt(d)),
            trainable=True,
            # dtype=inputs_shape.dtype,
        )

    def call(self, inputs, training=None):
        out = {}

        # We can't use where/argwhere since the output shape is not fixed.
        # Here we use the fact that sequences are padded with EOS tokens, that the
        # EOS token has value 1, and that argmin returns the first index.
        # eos_indices = jnp.argmin(text, axis=1)

        text = inputs
        x = out["embedded"] = self.embedding(text)

        # Add posemb
        n, l, d = x.shape  # pylint: disable=unused-variable
        x = x + self.pos_embedding
        x, encoder_out = self.Encoder_0(x, training=training)

        out.update({"transformed": x, **encoder_out})

        # Share weights between embeddings and logit transformation.
        out["vocab_logits"] = self.embedding.attend(x)

        if self.pool_type == "last":
            # Assuming "sticky" EOS tokenization, last token is always EOS.
            x = out["pre_logits"] = x[:, -1, :]
        elif self.pool_type == "first":
            x = out["pre_logits"] = x[:, 0, :]
        elif self.pool_type in ("mean", "gap"):
            x = out["pre_logits"] = x.mean(axis=1)
        elif self.pool_type in ("max", "gmp"):
            x = out["pre_logits"] = x.max(axis=1)
        elif self.pool_type == "map":
            x = out["pre_logits"] = self.MAPHead_0(x)
        else:
            raise NotImplementedError(f"Cannot do pooling '{self.pool_type}'")

        x = out["logits"] = self.head(x)
        return x, out


def keras_linalg_norm(x, axis=None, keepdims=False):
    """Computes the norm along axes."""
    return ops.sqrt(ops.sum(x * x, axis=axis, keepdims=keepdims))


# https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/image_text/two_towers.py
class TwoTowerModel(keras.Model):
    # the configuration is not very thought through, but it can load siglip model
    def __init__(
        self,
        variant=None,
        text_variant=None,
        pool_type="map",  # for img
        vocab_size: int = 32_000,
        out_dim: Union[int, Tuple[int, int]] = 128,
        temperature_init: float = 1.0,
        bias_init: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.temperature_init = temperature_init
        self.bias_init = bias_init

        out_dims = self.out_dim
        if isinstance(out_dims, int):
            out_dims = (out_dims, out_dims)

        if variant is not None:
            self.vit = ViT(pool_type=pool_type, **decode_variant(variant), name="img")

        if text_variant is not None:
            self.text = TextTransformer(
                num_classes=out_dims[1],
                **decode_variant(text_variant),
                vocab_size=vocab_size,
                name="txt",
            )

    def build(self, inputs_shape):
        temp_init = ops.log(self.temperature_init)
        self.t = self.add_weight(
            name="t",
            shape=(1,),
            initializer=initializers.Constant(temp_init),
            trainable=True,
            # dtype=x.dtype,
        )
        bias_init = self.bias_init
        if bias_init is not None:
            self.b = self.add_weight(
                name="b",
                shape=(1,),
                initializer=initializers.Constant(bias_init),
                trainable=True,
                # dtype=x.dtype,
            )

    def call(self, image, text=None, **kw):
        """Returns (B,C) image and (B,C) text representations."""
        out = {}
        zimg, ztxt = None, None
        ## text ##
        if text is not None:
            ztxt, out_txt = self.text(text, **kw)
            for k, v in out_txt.items():
                out[f"txt/{k}"] = v

            # Normalize the embeddings the models give us.
            out["txt/norm"] = keras_linalg_norm(ztxt, axis=1, keepdims=True)
            out["txt/normalized"] = ztxt = ztxt / (out["txt/norm"] + 1e-8)

        ## Image ##
        if image is not None:
            zimg, out_img = self.vit(image, **kw)
            for k, v in out_img.items():
                out[f"img/{k}"] = v

            # Normalize the embeddings the models give us.
            out["img/norm"] = keras_linalg_norm(zimg, axis=1, keepdims=True)
            out["img/normalized"] = zimg = zimg / (out["img/norm"] + 1e-8)

        # Compute the logits.
        t = self.t
        out["t"] = ops.exp(self.t)

        out["t/parameter"] = t
        if self.bias_init is not None:
            out["b"] = self.b * 1  # Make sure it doesn't return a reference. (happens with keras w/ torch backend)

        # We could actually play with pre-multiplying by temperature here, such
        # that out["t"] is nothing special to the trainer anymore.

        return zimg, ztxt, out
