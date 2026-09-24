"""Single entry point for loading ChromBPNet models (chrombpnet 1.x TF-Keras .h5 files and Keras 3 files)."""
import io
import json
import warnings

import keras

from chrombpnet.training.utils.layers import LogSumExp, LogSumExpCompat
from chrombpnet.training.utils.losses import multinomial_nll

_NESTED_MODEL_CLASSES = ("Functional", "Model", "Sequential")


def require_jax_backend():
    """Fail early with a clear message if Keras was initialised with another backend."""
    backend = keras.backend.backend()
    if backend != "jax":
        raise RuntimeError(
            "chrombpnet needs the Keras JAX backend but Keras is using {!r}. Set KERAS_BACKEND=jax before "
            "anything imports keras (importing chrombpnet first does this).".format(backend))


def custom_objects():
    return {"Lambda": LogSumExpCompat, "LogSumExp": LogSumExp, "multinomial_nll": multinomial_nll}


def _shift_nested_node_indices(model_config):
    """Renumber calls to nested models in a TF-Keras 2.x functional config for Keras 3.

    TF-Keras counts a nested model's own construction as its inbound node 0, so the first call inside the outer
    model is node 1 (e.g. `["model_wo_bias", 1, 0, {}]` in chrombpnet.h5). Keras 3 does not create that node,
    so those references are off by one. Returns True if anything changed.
    """
    config = model_config.get("config", {})
    layers = config.get("layers", [])
    nested = {layer["name"] for layer in layers if layer.get("class_name") in _NESTED_MODEL_CLASSES}
    changed = False
    for layer in layers:
        if layer.get("class_name") in _NESTED_MODEL_CLASSES:
            changed |= _shift_nested_node_indices(layer)
        for node in layer.get("inbound_nodes", []):
            for ref in node if isinstance(node, list) else []:
                if isinstance(ref, list) and ref and ref[0] in nested and ref[1] >= 1:
                    ref[1] -= 1
                    changed = True
    for key in ("input_layers", "output_layers"):
        for ref in config.get(key, []):
            if isinstance(ref, list) and ref and ref[0] in nested and ref[1] >= 1:
                ref[1] -= 1
                changed = True
    return changed


def _load_legacy_h5(path, compile):
    """Load a TF-Keras 2.x .h5 whose graph calls nested models, fixing the config on an in-memory copy."""
    import h5py
    from keras.src.legacy.saving import legacy_h5_format

    with open(path, "rb") as fh:
        buffer = io.BytesIO(fh.read())
    with h5py.File(buffer, "r+") as f:
        config = json.loads(f.attrs["model_config"])
        if _shift_nested_node_indices(config):
            f.attrs["model_config"] = json.dumps(config)
        return legacy_h5_format.load_model_from_hdf5(f, custom_objects=custom_objects(), compile=compile)


def _is_legacy_h5_with_nested_models(path):
    if not str(path).endswith((".h5", ".hdf5")):
        return False
    import h5py

    with h5py.File(path, "r") as f:
        version = f.attrs.get("keras_version", "")
        version = version.decode() if isinstance(version, bytes) else str(version)
        if not version.startswith("2.") or "model_config" not in f.attrs:
            return False
        config = json.loads(f.attrs["model_config"])
    return any(layer.get("class_name") in _NESTED_MODEL_CLASSES
               for layer in config.get("config", {}).get("layers", []))


def load_model(path, compile=False):
    """Load a bias / chrombpnet / chrombpnet_nobias model.

    Works for .keras files, .h5 files written by Keras 3, and legacy TF-Keras 2.x .h5 files, including full
    `chrombpnet.h5` models (nested bias + no-bias models and a logsumexp Lambda count head). Optimizer state
    is never restored.
    """
    if keras.backend.backend() != "jax":
        warnings.warn("Loading {} with the Keras {!r} backend; chrombpnet is tested on JAX only.".format(
            path, keras.backend.backend()))
    if _is_legacy_h5_with_nested_models(path):
        return _load_legacy_h5(str(path), compile)
    return keras.saving.load_model(str(path), compile=compile, custom_objects=custom_objects())


def load_model_wrapper(model_h5):
    """Backwards-compatible name used by chrombpnet 1.x code and downstream pipelines."""
    model = load_model(model_h5)
    print("got the model")
    model.summary()
    return model
