"""dna_to_one_hot must give exactly what the 1.x implementation gave.

The 1.x function (np.unique over the whole concatenated input) is kept here as the oracle and compared on the cases
where lookup-table encoders tend to differ: lowercase, N, other IUPAC codes, inputs that contain fewer than four
distinct bases, block boundaries, and a single string (a list of one-base sequences, which build_pwm_from_bigwig
passes for a whole chromosome).
"""
import random
import tracemalloc

import numpy as np
import pytest

from chrombpnet.training.utils import one_hot


def dna_to_one_hot_1x(seqs):
    """chrombpnet 1.x chrombpnet/training/utils/one_hot.py, verbatim (Alex Tseng)."""
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)
    seq_concat = "".join(seqs).upper() + "ACGT"
    one_hot_map = np.identity(5)[:, :-1].astype(np.int8)
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85
    _, base_inds = np.unique(base_vals, return_inverse=True)
    return one_hot_map[base_inds[:-4]].reshape((len(seqs), seq_len, 4))


def _check(seqs):
    got = one_hot.dna_to_one_hot(seqs)
    want = dna_to_one_hot_1x(seqs)
    assert got.dtype == want.dtype == np.int8
    assert got.shape == want.shape
    assert np.array_equal(got, want)


@pytest.mark.parametrize("seqs", [
    pytest.param(["ACGT"], id="minimal"),
    pytest.param(["acgt"], id="lowercase"),
    pytest.param(["AcGt"], id="mixed-case"),
    pytest.param(["NNNN"], id="all-N"),
    pytest.param(["ACGN"], id="trailing-N"),
    pytest.param(["RYKMSWBDHV-."], id="other-iupac-and-gaps"),
    pytest.param(["AAAA"], id="one-base-only"),
    pytest.param(["AAAA", "CCCC", "GGGG", "TTTT"], id="one-base-per-seq"),
    pytest.param(["ACGT", "TGCA", "NNNN", "acgt"], id="mixed-batch"),
    pytest.param(("ACGT", "TTTT"), id="tuple"),
    pytest.param(np.array(["ACGT", "ggNN"]), id="numpy-str-array"),
])
def test_matches_1x(seqs):
    _check(seqs)


def test_matches_1x_on_random_regions():
    rng = random.Random(0)
    _check(["".join(rng.choices("ACGTNacgtn", k=2114)) for _ in range(300)])


def test_block_and_chunk_boundaries(monkeypatch):
    rng = random.Random(1)
    real_empty = np.empty

    def poisoned_empty(shape, dtype=float):  # fresh memory is usually zero, which hides an unwritten (N) row
        return np.full(shape, 99, dtype=dtype)

    monkeypatch.setattr(one_hot.np, "empty", poisoned_empty)
    monkeypatch.setattr(one_hot, "_BLOCK", 7)
    monkeypatch.setattr(one_hot, "_CHUNK", 13)  # not a multiple of the sequence length
    for n in (1, 6, 7, 8, 14, 15):
        _check(["".join(rng.choices("ACGTN", k=11)) for _ in range(n)])
    _check("".join(rng.choices("ACGTNacgt", k=40)))
    assert np.empty is poisoned_empty and real_empty is not poisoned_empty


def test_default_block_boundary():
    rng = random.Random(2)
    _check(["".join(rng.choices("ACGTacgtN", k=8)) for _ in range(one_hot._BLOCK + 3)])


def test_string_is_a_list_of_one_base_sequences():
    rng = random.Random(3)
    chrom = "".join(rng.choices("ACGTNacgtn", k=10001))
    _check(chrom)
    assert one_hot.dna_to_one_hot(chrom).squeeze().shape == (10001, 4)  # what build_pwm_from_bigwig relies on


def test_non_ascii_characters_encode_to_zeros():
    # 1.x encoded the UTF-8 bytes, so a multi-byte character broke the reshape; here it is one all-zero position
    got = one_hot.dna_to_one_hot(["ACéT"])
    assert got.shape == (1, 4, 4)
    assert got[0].tolist() == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]


def test_ragged_input_is_rejected():
    with pytest.raises(AssertionError):
        one_hot.dna_to_one_hot(["ACGT", "AC"])


def test_ragged_input_is_rejected_without_the_assert(monkeypatch):
    """Under python -O the length assert is skipped; the block must not come back with rows never written."""
    monkeypatch.setattr(one_hot, "_CHUNK", 4)
    with pytest.raises(ValueError, match="same length"):
        one_hot._encode_into("ACGT", np.empty((8, 4), dtype=np.int8))  # 4 bases for 2 x 4 rows, a multiple of _CHUNK


def test_empty_list():
    assert one_hot.dna_to_one_hot([]).shape == (0, 0, 4)


def test_round_trip():
    seqs = ["ACGTN", "TTGCA"]
    assert one_hot.one_hot_to_dna(one_hot.dna_to_one_hot(seqs)) == seqs


def test_peak_memory_is_close_to_the_output():
    """1.x peaked at ~7x the int8 output (the int64 inverse of np.unique); the lookup table stays near 1x."""
    rng = random.Random(4)
    seqs = ["".join(rng.choices("ACGTN", k=2114)) for _ in range(2000)]
    tracemalloc.start()
    try:
        out = one_hot.dna_to_one_hot(seqs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * out.nbytes
