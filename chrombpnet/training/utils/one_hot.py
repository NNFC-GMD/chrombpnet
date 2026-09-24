"""
One-hot encoding of DNA sequences. The encoding and one_hot_to_dna were written by Alex Tseng:
https://gist.github.com/amtseng/010dd522daaabc92b014f075a34a0a0b
"""

import numpy as np

# One row per byte value: A, C, G, T and a, c, g, t are one-hot; every other byte (N, other IUPAC codes, anything
# else) is all zeros.
_ONE_HOT_LUT = np.zeros((256, 4), dtype=np.int8)
for _i, _base in enumerate(b"ACGT"):
    _ONE_HOT_LUT[_base, _i] = 1
    _ONE_HOT_LUT[_base + 32, _i] = 1  # lowercase
del _i, _base

# Sequences joined at a time: the joined string and its bytes span one block (4096 x 2114 bp is about 9 MB), however
# many sequences there are.
_BLOCK = 4096
# Bases looked up at a time: np.take converts its indices to int64 (8 bytes per base) before the lookup.
_CHUNK = 1 << 18


def _encode_into(text, out):
    # Non-ASCII characters become "?" (one byte each, so lengths are kept), which encodes to zeros like any other
    # non-ACGT character. The byte values are always valid indices, so mode="clip" never clips; it keeps np.take from
    # staging the output in a temporary of its own size, which mode="raise" does when out= is given.
    codes = np.frombuffer(text.encode("ascii", "replace"), dtype=np.uint8)
    if len(codes) != len(out):
        # only reachable with ragged input under python -O (the length assert is gone); rows would stay unwritten
        raise ValueError("sequences are not all the same length")
    for start in range(0, len(codes), _CHUNK):
        np.take(_ONE_HOT_LUT, codes[start:start + _CHUNK], axis=0, out=out[start:start + _CHUNK], mode="clip")


def dna_to_one_hot(seqs):
    """
    Converts a list of DNA ("ACGT") sequences to one-hot encodings, where the
    position of 1s is ordered alphabetically by "ACGT". `seqs` must be a list
    of N strings, where every string is the same length L. Returns an N x L x 4
    NumPy array of one-hot encodings, in the same order as the input sequences.
    All bases will be converted to upper-case prior to performing the encoding.
    Any bases that are not "ACGT" will be given an encoding of all 0s.

    A single string is taken as a list of one-base sequences (an L x 1 x 4 array), as in chrombpnet 1.x.

    The output is identical to the 1.x implementation (Alex Tseng's, https://gist.github.com/amtseng/010dd522daaabc92b014f075a34a0a0b),
    which encoded everything in one pass through np.unique(return_inverse=True). Its int64 inverse alone cost 8 bytes
    per base, several times the int8 result. Here each byte indexes a 256-row lookup table, in blocks written straight
    into the output.
    """
    if isinstance(seqs, str):
        out = np.empty((len(seqs), 1, 4), dtype=np.int8)
        _encode_into(seqs, out.reshape(-1, 4))
        return out

    seqs = list(seqs)
    if not seqs:
        return np.zeros((0, 0, 4), dtype=np.int8)
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)

    out = np.empty((len(seqs), seq_len, 4), dtype=np.int8)
    for start in range(0, len(seqs), _BLOCK):
        block = seqs[start:start + _BLOCK]
        _encode_into("".join(block), out[start:start + len(block)].reshape(-1, 4))
    return out


def one_hot_to_dna(one_hot):
    """
    Converts a one-hot encoding into a list of DNA ("ACGT") sequences, where the
    position of 1s is ordered alphabetically by "ACGT". `one_hot` must be an
    N x L x 4 array of one-hot encodings. Returns a lits of N "ACGT" strings,
    each of length L, in the same order as the input array. The returned
    sequences will only consist of letters "A", "C", "G", "T", or "N" (all
    upper-case). Any encodings that are all 0s will be translated to "N".
    """
    bases = np.array(["A", "C", "G", "T", "N"])
    # Create N x L array of all 5s
    one_hot_inds = np.tile(one_hot.shape[2], one_hot.shape[:2])

    # Get indices of where the 1s are
    batch_inds, seq_inds, base_inds = np.where(one_hot)

    # In each of the locations in the N x L array, fill in the location of the 1
    one_hot_inds[batch_inds, seq_inds] = base_inds

    # Fetch the corresponding base for each position using indexing
    seq_array = bases[one_hot_inds]
    return ["".join(seq) for seq in seq_array]
