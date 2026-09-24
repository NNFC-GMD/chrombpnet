"""Process-wide runtime settings for training: matmul precision, mixed precision and the device check."""
import keras

PRECISIONS = ("default", "highest", "bf16")
DEVICES = ("auto", "gpu", "cpu")
MIXED_BF16 = "mixed_bfloat16"


def configure_precision(precision="default"):
    """Apply a --precision choice.

    default  leave JAX's default (TF32 for float32 matmuls/convs on Ampere and newer GPUs, like TF 2.x)
    highest  full float32 matmuls/convs (jax_default_matmul_precision='highest'); used for parity runs
    bf16     keras mixed_bfloat16 policy (bfloat16 compute, float32 variables) for models built afterwards;
             the architecture files keep the output heads in float32 (see head_dtype)
    """
    precision = precision or "default"
    if precision not in PRECISIONS:
        raise ValueError("precision must be one of {}, got {!r}".format(PRECISIONS, precision))
    if precision == "highest":
        import jax
        jax.config.update("jax_default_matmul_precision", "highest")
    elif precision == "bf16":
        keras.config.set_dtype_policy(MIXED_BF16)
    print("precision: {} (dtype policy: {})".format(precision, keras.config.dtype_policy().name))
    return precision


def bf16_active():
    return keras.config.dtype_policy().name == MIXED_BF16


def head_dtype():
    """dtype for the output-head layers: float32 under mixed_bfloat16, else None (the global policy)."""
    return "float32" if bf16_active() else None


def set_float32_policy(model):
    """Switch every layer of `model` (nested models included) to the float32 policy, e.g. before saving a model
    trained with mixed_bfloat16 so that predict / interpret run it in float32. Variables are float32 already."""
    for layer in model._flatten_layers(include_self=True, recursive=True):
        if layer.dtype_policy.name != "float32":
            layer.dtype_policy = "float32"


def assert_gpu_if_requested(device="auto"):
    """--device: 'gpu' fails unless JAX runs on a GPU, 'cpu' makes the CPU the default JAX device, 'auto' only
    logs what JAX found. Returns the JAX backend name."""
    import jax
    device = device or "auto"
    if device not in DEVICES:
        raise ValueError("device must be one of {}, got {!r}".format(DEVICES, device))
    backend = jax.default_backend()
    print("jax backend: {}, devices: {}".format(backend, jax.devices()))
    if device == "gpu" and backend not in ("gpu", "cuda", "rocm"):
        raise RuntimeError(
            "--device gpu was requested but JAX is running on {!r} (devices: {}). Check that the CUDA build of "
            "jax is installed (pixi -e cuda13) and that a GPU is visible (nvidia-smi).".format(
                backend, jax.devices()))
    if device == "cpu" and backend != "cpu":
        jax.config.update("jax_default_device", jax.devices("cpu")[0])
        backend = "cpu"
    return backend


def runtime_info():
    """Versions, devices and precision settings, for the args.json of a run (JSON-serializable)."""
    import jax
    return {"keras": keras.__version__, "jax": jax.__version__, "keras_backend": keras.backend.backend(),
            "jax_backend": jax.default_backend(), "jax_devices": [str(d) for d in jax.devices()],
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            "dtype_policy": keras.config.dtype_policy().name}
