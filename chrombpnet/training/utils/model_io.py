from tensorflow.keras.models import load_model


def load_model_compat(filepath, custom_objects=None, compile=False):
    """Load trusted ChromBPNet model artifacts across Keras 2 and Keras 3."""
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
