from tensorflow.keras.models import load_model
from chrombpnet.training.utils.layers import LogcountSum


def load_model_compat(filepath, custom_objects=None, compile=False):
    """Load trusted ChromBPNet model artifacts across Keras 2 and Keras 3."""
    custom_objects = dict(custom_objects or {})
    custom_objects.setdefault("LogcountSum", LogcountSum)
    custom_objects.setdefault("chrombpnet>LogcountSum", LogcountSum)
    try:
        return load_model(
            filepath,
            custom_objects=custom_objects,
            compile=compile,
            safe_mode=False,
        )
    except TypeError:
        return load_model(
            filepath,
            custom_objects=custom_objects,
            compile=compile,
        )
