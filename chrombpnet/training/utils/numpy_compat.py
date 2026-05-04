def patch_numpy_for_deepdish():
    """Restore NumPy 1.x aliases used by deepdish 0.3.7."""
    import numpy as np

    if "ComplexWarning" not in np.__dict__:
        np.ComplexWarning = getattr(np.exceptions, "ComplexWarning", RuntimeWarning)
    if "unicode_" not in np.__dict__:
        np.unicode_ = np.str_
    if "string_" not in np.__dict__:
        np.string_ = np.bytes_
    if "object" not in np.__dict__:
        np.object = np.object_
    if "int" not in np.__dict__:
        np.int = int
    if "float" not in np.__dict__:
        np.float = float
    if "complex" not in np.__dict__:
        np.complex = complex
