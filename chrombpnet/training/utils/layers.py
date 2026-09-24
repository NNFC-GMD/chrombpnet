import base64

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="chrombpnet")
class LogSumExp(keras.layers.Layer):
    """Combine the bias and no-bias log-count heads: logsumexp over the last axis, keepdims.

    Replaces the `Lambda(lambda x: tf.math.reduce_logsumexp(...))` of chrombpnet 1.x, which cannot be
    serialized portably (marshalled Python bytecode) and cannot run on the JAX backend.
    """

    def call(self, inputs):
        return ops.logsumexp(inputs, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (1,)


_LAMBDA_ONLY_KEYS = ("function", "output_shape", "mask", "arguments", "module", "function_type",
                     "output_shape_type", "output_shape_module")


def _lambda_code_bytes(function):
    """Raw marshalled code of a serialized Lambda function (list or dict form), or b"" if unknown."""
    code = None
    if isinstance(function, (list, tuple)) and function:
        code = function[0]
    elif isinstance(function, dict):
        code = function.get("config", {}).get("code", function.get("code"))
    elif isinstance(function, str):
        code = function
    if not isinstance(code, str):
        return b""
    try:
        return base64.decodebytes(code.encode("ascii"))
    except (ValueError, TypeError):
        return b""


class LogSumExpCompat(LogSumExp):
    """Stand-in for the logsumexp Lambda inside chrombpnet 1.x `chrombpnet.h5` files.

    Passed to load_model as custom_objects={"Lambda": LogSumExpCompat}. It never unmarshals the stored bytecode
    (which only loads on the Python version that wrote it). Any other Lambda is refused rather than silently
    replaced. from_config returns a plain LogSumExp, so a loaded 1.x model re-saves as `chrombpnet>LogSumExp`
    and loads again.
    """

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        function = config.get("function")
        name = config.get("name")
        if name != "logcount_predictions" or b"reduce_logsumexp" not in _lambda_code_bytes(function):
            raise ValueError(
                "Cannot load Lambda layer {!r}: only the logsumexp count head of chrombpnet 1.x models "
                "(`logcount_predictions`) is supported. Rebuild custom architectures without Lambda layers "
                "(see chrombpnet.training.utils.layers).".format(name))
        for key in _LAMBDA_ONLY_KEYS:
            config.pop(key, None)
        return LogSumExp(**config)
