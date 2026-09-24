"""Single entry point for loading ChromBPNet models (chrombpnet 1.x TF-Keras .h5 files and Keras 3 files)."""
import warnings

import keras

from chrombpnet.training.utils.layers import LogSumExp, LogSumExpCompat
from chrombpnet.training.utils.losses import multinomial_nll


def require_jax_backend():
    """Fail early with a clear message if Keras was initialised with another backend."""
    backend = keras.backend.backend()
    if backend != "jax":
        raise RuntimeError(
            "chrombpnet needs the Keras JAX backend but Keras is using {!r}. Set KERAS_BACKEND=jax before "
            "anything imports keras (importing chrombpnet first does this).".format(backend))


def custom_objects():
    return {"Lambda": LogSumExpCompat, "LogSumExp": LogSumExp, "multinomial_nll": multinomial_nll}


def load_model(path, compile=False):
    """Load a bias / chrombpnet / chrombpnet_nobias model.

    Works for .keras files, .h5 files written by Keras 3, and legacy TF-Keras 2.x .h5 files, including full
    `chrombpnet.h5` models whose count head is a logsumexp Lambda. Optimizer state is never restored.
    """
    if keras.backend.backend() != "jax":
        warnings.warn("Loading {} with the Keras {!r} backend; chrombpnet is tested on JAX only.".format(
            path, keras.backend.backend()))
    return keras.saving.load_model(str(path), compile=compile, custom_objects=custom_objects())


def load_model_wrapper(model_h5):
    """Backwards-compatible name used by chrombpnet 1.x code and downstream pipelines."""
    model = load_model(model_h5)
    print("got the model")
    model.summary()
    return model
