"""Preprocessing helpers: seeded read sampling, shift decisions, subprocess error propagation, bigWig building."""
import argparse
import shutil
import subprocess
import warnings

import numpy as np
import pyBigWig
import pytest

from chrombpnet.data import DefaultDataFile, get_default_data_path
from chrombpnet.helpers.preprocessing import auto_shift_detect, reads_to_bigwig

CHROM_LEN = 5000


@pytest.fixture
def genome(tmp_path):
    rng = np.random.RandomState(0)
    fasta = tmp_path / "genome.fa"
    sizes = tmp_path / "genome.chrom.sizes"
    with open(fasta, "w") as f:
        for chrom in ("chr1", "chr2"):
            seq = "".join(rng.choice(list("ACGT"), CHROM_LEN))
            f.write(">{}\n".format(chrom))
            for i in range(0, CHROM_LEN, 60):
                f.write(seq[i:i + 60] + "\n")
    sizes.write_text("chr1\t{0}\nchr2\t{0}\n".format(CHROM_LEN))
    return str(fasta), str(sizes)


def write_tagalign(path, reads):
    with open(path, "w") as f:
        for chrom, start, end, strand in reads:
            f.write("{}\t{}\t{}\tN\t1000\t{}\n".format(chrom, start, end, strand))
    return str(path)


def random_reads(n, seed=1):
    rng = np.random.RandomState(seed)
    reads = []
    for i in range(n):
        chrom = "chr1" if i % 2 else "chr2"
        start = int(rng.randint(100, CHROM_LEN - 200))
        reads.append((chrom, start, start + 50, "+" if rng.rand() < 0.5 else "-"))
    return reads


def test_sample_reads_is_seeded_and_filters_unknown_chroms(tmp_path, genome):
    fasta, _ = genome
    reads = random_reads(400) + [("chrEBV", 10, 60, "+")]
    tagalign = write_tagalign(tmp_path / "reads.tagAlign", reads)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plus1, minus1 = auto_shift_detect.sample_reads(None, None, tagalign, 50, fasta, seed=7)
    assert any("chromosomes not in the reference" in str(w.message) for w in caught)
    plus2, minus2 = auto_shift_detect.sample_reads(None, None, tagalign, 50, fasta, seed=7)
    plus3, minus3 = auto_shift_detect.sample_reads(None, None, tagalign, 50, fasta, seed=8)

    assert len(plus1) + len(minus1) == 100
    assert plus1.equals(plus2) and minus1.equals(minus2)
    assert not (plus1.equals(plus3) and minus1.equals(minus3))
    sampled = set(map(tuple, plus1.values.tolist() + minus1.values.tolist()))
    assert len(sampled) == 100  # without replacement (all synthetic reads are distinct)
    known = {(c, str(s), str(e)) for c, s, e, _ in reads}
    assert sampled <= known
    assert "chrEBV" not in set(plus1["chr"]) | set(minus1["chr"])


def test_sample_reads_returns_everything_when_input_is_small(tmp_path, genome):
    fasta, _ = genome
    reads = random_reads(30)
    tagalign = write_tagalign(tmp_path / "reads.tagAlign", reads)
    plus, minus = auto_shift_detect.sample_reads(None, None, tagalign, 50, fasta, seed=1)
    assert len(plus) + len(minus) == 30
    assert len(plus) == sum(r[3] == "+" for r in reads)


def test_reservoir_sample_is_uniform():
    # every line should be picked with probability k/n
    n, k, trials = 50, 5, 4000
    lines = ["chr1\t{}\t{}\tN\t0\t+\n".format(i, i + 1).encode() for i in range(n)]

    class Stream:
        def __init__(self):
            import io
            self.stdout = io.BytesIO(b"".join(lines))

    class FakeFasta:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def keys(self):
            return ["chr1"]

    orig = auto_shift_detect.pyfaidx.Fasta
    auto_shift_detect.pyfaidx.Fasta = FakeFasta
    try:
        counts = np.zeros(n)
        for seed in range(trials):
            sample, unknown = auto_shift_detect.sample_filtered_tagaligns(Stream(), "unused.fa", k, seed)
            assert len(sample) == k and len(set(sample)) == k and not unknown
            for line in sample:
                counts[int(line.split(b"\t")[1])] += 1
    finally:
        auto_shift_detect.pyfaidx.Fasta = orig
    expected = trials * k / n
    # binomial sd ~ sqrt(400*0.9) = 19; allow 5 sd
    assert np.all(np.abs(counts - expected) < 5 * np.sqrt(expected * (1 - k / n)))


def test_sample_reads_without_known_chroms_raises(tmp_path, genome):
    fasta, _ = genome
    tagalign = write_tagalign(tmp_path / "reads.tagAlign", [("chrUn", 10, 60, "+")])
    with pytest.raises(ValueError, match="No reads"):
        auto_shift_detect.sample_reads(None, None, tagalign, 50, fasta)


def test_check_returncode_raises_on_failure():
    auto_shift_detect.check_returncode(subprocess.Popen(["true"]))
    with pytest.raises(subprocess.CalledProcessError):
        auto_shift_detect.check_returncode(subprocess.Popen(["false"]))


@pytest.mark.needs_cli
def test_missing_bam_raises(tmp_path, genome):
    if shutil.which("bedtools") is None:
        pytest.skip("bedtools not on PATH")
    fasta, _ = genome
    with pytest.raises(subprocess.CalledProcessError):
        auto_shift_detect.sample_reads(str(tmp_path / "missing.bam"), None, None, 10, fasta)


def embed(pwm, offset, length=40):
    out = np.full((length, 4), 0.25)
    out[offset:offset + len(pwm)] = pwm / pwm.sum(-1, keepdims=True)
    return out


@pytest.mark.parametrize("shift", [(0, 0), (4, -5), (3, -6), (5, -4)])
def test_compute_shift_ATAC_on_embedded_reference(shift):
    plus, minus = auto_shift_detect.get_ref_pwms(get_default_data_path(DefaultDataFile.atac_ref_motifs))
    assert len(plus) == 3 and len(minus) == 3
    plus_pwm = embed(plus["GSE101074_naive_hESC_ATAC_plus"], 14 - shift[0])
    minus_pwm = embed(minus["GSE101074_naive_hESC_ATAC_minus"], 5 - shift[1])
    assert auto_shift_detect.compute_shift_ATAC(plus, minus, plus_pwm, minus_pwm) == shift


def test_compute_shift_ATAC_rejects_nonstandard_shift():
    plus, minus = auto_shift_detect.get_ref_pwms(get_default_data_path(DefaultDataFile.atac_ref_motifs))
    plus_pwm = embed(plus["GSE101074_naive_hESC_ATAC_plus"], 14)
    minus_pwm = embed(minus["GSE101074_naive_hESC_ATAC_minus"], 4)
    with pytest.raises(ValueError, match="non-standard"):
        auto_shift_detect.compute_shift_ATAC(plus, minus, plus_pwm, minus_pwm)


@pytest.mark.parametrize("shift", [(0, 0), (0, 1)])
def test_compute_shift_DNASE_on_embedded_reference(shift):
    plus, minus = auto_shift_detect.get_ref_pwms(get_default_data_path(DefaultDataFile.dnase_ref_motifs))
    assert len(plus) == 6 and len(minus) == 6
    plus_pwm = embed(plus["ENCSR000EMN_DNASE_plus"], 10 - shift[0])
    minus_pwm = embed(minus["ENCSR000EMN_DNASE_minus"], 10 - shift[1])
    assert auto_shift_detect.compute_shift_DNASE(plus, minus, plus_pwm, minus_pwm) == shift


def needs_bigwig_tools():
    missing = [t for t in ("bedtools", "bedGraphToBigWig", "awk", "sort") if shutil.which(t) is None]
    if missing:
        pytest.skip("missing tools: " + ", ".join(missing))


def expected_insertions(reads, plus_delta, minus_delta):
    # bedtools genomecov -5: + strand counts `start`, - strand counts `end - 1`
    counts = {}
    for chrom, start, end, strand in reads:
        pos = start + plus_delta if strand == "+" else end + minus_delta - 1
        counts[(chrom, pos)] = counts.get((chrom, pos), 0) + 1
    return counts


@pytest.mark.needs_cli
@pytest.mark.parametrize("no_st", [False, True])
def test_reads_to_bigwig_with_explicit_shift(tmp_path, genome, no_st):
    needs_bigwig_tools()
    fasta, sizes = genome
    reads = random_reads(300, seed=3)
    tagalign = write_tagalign(tmp_path / "reads.tagAlign", reads)
    args = argparse.Namespace(genome=fasta, input_bam_file=None, input_fragment_file=None,
                              input_tagalign_file=tagalign, chrom_sizes=sizes,
                              output_prefix=str(tmp_path / "out"), data_type="ATAC", bsort=False,
                              no_st=no_st, tmpdir=None, plus_shift=0, minus_shift=0,
                              ATAC_ref_path=None, DNASE_ref_path=None, num_samples=100, seed=1234)
    reads_to_bigwig.main(args)

    expected = expected_insertions(reads, 4, -4)
    bw = pyBigWig.open(str(tmp_path / "out_unstranded.bw"))
    for chrom in ("chr1", "chr2"):
        vals = np.nan_to_num(np.array(bw.values(chrom, 0, CHROM_LEN)))
        want = np.zeros(CHROM_LEN)
        for (c, pos), n in expected.items():
            if c == chrom:
                want[pos] = n
        np.testing.assert_array_equal(vals, want)
    bw.close()


@pytest.mark.needs_cli
def test_reads_to_bigwig_propagates_bedgraphtobigwig_failure(tmp_path, genome):
    needs_bigwig_tools()
    fasta, sizes = genome
    tagalign = write_tagalign(tmp_path / "reads.tagAlign", random_reads(20))
    with pytest.raises(subprocess.CalledProcessError):
        reads_to_bigwig.generate_bigwig(None, None, tagalign, str(tmp_path / "out"), fasta, False, None, False,
                                        str(tmp_path / "missing.chrom.sizes"), 4, -4)


def test_build_pwm_from_bigwig_plot(tmp_path, genome):
    from chrombpnet.helpers.preprocessing.analysis import build_pwm_from_bigwig
    import matplotlib
    matplotlib.use("Agg")

    fasta, sizes = genome
    bw_path = str(tmp_path / "cov.bw")
    bw = pyBigWig.open(bw_path, "w")
    bw.addHeader([("chr1", CHROM_LEN), ("chr2", CHROM_LEN)])
    starts = list(range(100, CHROM_LEN - 100, 37))
    bw.addEntries(["chr1"] * len(starts), starts, ends=[s + 1 for s in starts], values=[1.0 + (s % 3) for s in starts])
    bw.close()

    args = argparse.Namespace(bigwig=bw_path, genome=fasta, output_prefix=str(tmp_path / "qc"), chr="chr1",
                              chrom_sizes=sizes, pwm_width=24)
    build_pwm_from_bigwig.main(args)
    assert (tmp_path / "qc.png").stat().st_size > 0
