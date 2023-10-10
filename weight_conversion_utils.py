import numpy as np
import torch


def keras_weight_to_nested_dict(keras_weights):
    weights = {}
    for w in keras_weights:
        if isinstance(w.value, torch.Tensor):
            value = w.value.detach().cpu().numpy()
        else:
            value = np.array(w.value)
        weights[w.path] = value

    nested_dict = {}
    for path, value in weights.items():
        path = path.split("/")
        d = nested_dict
        for p in path[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[path[-1]] = value
    return nested_dict


def npz_to_state_dict(npz):
    state_dict = {}
    for k, v in npz.items():
        path = k.split("/")
        d = state_dict
        for p in path[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[path[-1]] = v
    return state_dict


def load_weight_with_conversion(state_dict_npz, state_dict_keras, parent_key=None, path=""):
    """recursively load weights from npz to keras model with some coversion logic
    This function is developed in a notebook
    It's an interactive process, so we print out all the info if there is mismatch
    """
    if parent_key is None:
        pass
    elif parent_key.startswith("LayerNorm_") or parent_key == "encoder_norm":
        state_dict_npz = {
            "gamma": state_dict_npz["scale"],
            "beta": state_dict_npz["bias"],
        }
    elif parent_key.startswith("MultiHeadDotProductAttention_"):
        state_dict_npz = {
            "attention_output": state_dict_npz["out"],
            **{k: v for k, v in state_dict_npz.items() if k in ["key", "value", "query"]},
        }
    elif parent_key == "Embed_0":
        state_dict_npz = {"embeddings": state_dict_npz["embedding"]}
    matched_keys, unmatched_keys, unused_keys = [], [], []
    npz_keys = list(state_dict_npz.keys())
    keras_keys = list(state_dict_keras.keys())
    matched_keys = [k for k in npz_keys if k in keras_keys]
    unused_keys = [k for k in npz_keys if k not in keras_keys]
    unmatched_keys = [k for k in keras_keys if k not in npz_keys]
    info = f"""
    matched_keys: {matched_keys}
    unused_keys: {unused_keys} (in npz but not in keras)
    unmatched_keys: {unmatched_keys} (in keras but not in npz)
    parent_key: {parent_key}
    path: {path}
    """
    if unmatched_keys or unused_keys:
        raise ValueError(info)
    for k in matched_keys:
        if isinstance(state_dict_npz[k], np.ndarray):
            if not isinstance(state_dict_keras[k], np.ndarray):
                raise ValueError(
                    f"type mismatch for {k}, npz: {type(state_dict_npz[k])}, keras: {type(state_dict_keras[k])}\n{info}"
                )
            # check shape
            if state_dict_npz[k].shape != state_dict_keras[k].shape:
                raise ValueError(
                    f"shape mismatch for {path} {k}, npz: {state_dict_npz[k].shape}, keras: {state_dict_keras[k].shape}"
                )
            state_dict_keras[k] = state_dict_npz[k]
        elif isinstance(state_dict_npz[k], dict):
            if not isinstance(state_dict_keras[k], dict):
                raise ValueError(
                    f"type mismatch for {k}, npz: {type(state_dict_npz[k])}, keras: {type(state_dict_keras[k])}\n{info}"
                )
            load_weight_with_conversion(
                state_dict_npz[k],
                state_dict_keras[k],
                parent_key=k,
                path=path + "/" + k,
            )
        else:
            raise ValueError("unknown type")


def flatten_dict(d, path="", sep="/", outputs=[]):
    """flatten a nested dict to a flat dict in keras style"""

    def _flatten_dict(d, path="", sep="/", outputs=[]):
        if not isinstance(d, dict):
            outputs.append((path, d))
            return outputs
        else:
            for k, v in d.items():
                _flatten_dict(v, path + sep + k, sep, outputs)
            return outputs

    outputs = _flatten_dict(d, path, sep, outputs)
    return {k: v for k, v in outputs}


def load_weight(keras_model, flattened_dict):
    weights_arr_list = []
    for w in keras_model.weights:
        path = "/" + w.path
        if path not in flattened_dict:
            raise ValueError(f"path not found in flattened_dict: {path}")
        weights_arr_list.append(flattened_dict[path])
    keras_model.set_weights(weights_arr_list)
