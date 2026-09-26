"""get_seq, get_cts and get_coords must give exactly what the per-row iterrows() loops before them gave.

Those loops are kept here verbatim as the reference and compared (values, dtype, shape) on a small genome with
lowercase and N bases, a contig name longer than 21 characters (it sets get_coords' string dtype), a bigWig with gaps
(NaN), +-inf values and a chromosome it lacks, and regions that are unsorted, duplicated, overlapping, touching both
chromosome ends or reaching past them. get_cts reads nearby regions together; the window limits are shrunk to split
the reads at every boundary.
"""
import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
import pytest

from chrombpnet.training.utils import data_utils, one_hot

NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]
LONG_NAME = "chrUn_KI270742v1_random_contig"
CHROMS = {"chr1": 12_000, "chr2": 9_000, LONG_NAME: 3_000, "chrM": 700}   # chrM is left out of the bigWig
BW_CHROMS = ["chr1", "chr2", LONG_NAME]


# --- the implementations before, verbatim --------------------------------------------------------------------------

def get_seq_ref(peaks_df, genome, width):
    """
    Same as get_cts, but fetches sequence from a given genome.
    """
    vals = []

    for i, r in peaks_df.iterrows():
        sequence = str(genome[r['chr']][(r['start']+r['summit'] - width//2):(r['start'] + r['summit'] + width//2)])
        vals.append(sequence)

    return one_hot.dna_to_one_hot(vals)


def get_cts_ref(peaks_df, bw, width):
    """
    Fetches values from a bigwig bw, given a df with minimally
    chr, start and summit columns. Summit is relative to start.
    Retrieves values of specified width centered at summit.

    "cts" = per base counts across a region
    """
    vals = []
    for i, r in peaks_df.iterrows():
        vals.append(np.nan_to_num(bw.values(r['chr'],
                                            r['start'] + r['summit'] - width//2,
                                            r['start'] + r['summit'] + width//2)))

    return np.array(vals)


def get_coords_ref(peaks_df, peaks_bool):
    """
    Fetch the co-ordinates of the regions in bed file
    returns a list of tuples with (chrom, summit)
    """
    vals = []
    for i, r in peaks_df.iterrows():
        vals.append([r['chr'], r['start']+r['summit'], "f", peaks_bool])

    return np.array(vals)


# --- data ----------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("data_utils")
    rng = np.random.default_rng(7)
    with open(d / "genome.fa", "w") as f:
        for chrom, size in CHROMS.items():
            seq = rng.choice(list("ACGT"), size)
            for _ in range(4):   # soft-masked stretches, N runs and a few other IUPAC codes
                lo = int(rng.integers(0, size - 300))
                seq[lo:lo + 300] = np.char.lower(seq[lo:lo + 300])
                lo = int(rng.integers(0, size - 50))
                seq[lo:lo + 50] = "N"
            seq[rng.integers(0, size, 10)] = rng.choice(list("RYnK"), 10)
            seq[:20] = "N"
            seq = "".join(seq.tolist())
            f.write(">{}\n".format(chrom))
            f.writelines(seq[i:i + 50] + "\n" for i in range(0, size, 50))

    bw = pyBigWig.open(str(d / "signal.bw"), "w")
    bw.addHeader([(c, CHROMS[c]) for c in BW_CHROMS])
    for chrom in BW_CHROMS:
        size = CHROMS[chrom]
        lens = rng.integers(1, 20, size)
        gaps = rng.integers(0, 30, size)
        starts = np.cumsum(gaps + np.r_[0, lens[:-1]])
        ends = starts + lens
        keep = (ends <= size) & ~((starts < 4_500) & (ends > 4_000))   # and a 500 bp stretch without data
        starts, ends = starts[keep], ends[keep]
        values = rng.integers(0, 40, len(starts)) * rng.choice([0.1, 0.5, 1.0, -1.0], len(starts))
        if chrom == "chr1":
            values[[0, 1]] = [np.inf, -np.inf]   # inside the regions that start at 0
        bw.addEntries([chrom] * len(starts), starts.tolist(), ends=ends.tolist(), values=values.tolist())
    bw.close()

    genome = pyfaidx.Fasta(str(d / "genome.fa"))
    signal = pyBigWig.open(str(d / "signal.bw"))
    yield d, genome, signal
    signal.close()
    genome.close()


def regions(d, width, n=400, chroms=BW_CHROMS, seed=0):
    """
    narrowPeak rows (read back with read_csv, as the pipeline does) whose width-wide windows lie inside their
    chromosome, both ends included; shuffled, with duplicates and a non-default index.
    """
    rng = np.random.default_rng(seed)
    half = width // 2
    rows = []
    for i in range(n):
        chrom = chroms[i % len(chroms)]
        size = CHROMS[chrom]
        centre = [half, size - half][i % 2] if i < 4 * len(chroms) else int(rng.integers(half, size - half + 1))
        offset = int(rng.integers(0, min(centre, 400) + 1))
        rows.append([chrom, centre - offset, centre - offset + 500, "r{}".format(i), 1000, ".", 1.5, 2.0, 3.25, offset])
    rows += rows[:20]
    pd.DataFrame(rows).to_csv(d / "regions.bed", sep="\t", header=False, index=False)
    df = pd.read_csv(d / "regions.bed", sep="\t", header=None, names=NARROWPEAK_SCHEMA)
    return df.sample(frac=1, random_state=seed).iloc[3:]


def frame(rows):
    return pd.DataFrame(rows, columns=["chr", "start", "summit"])


def assert_identical(got, want):
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    assert np.array_equal(got, want)


def assert_same_outcome(new, ref, *args):
    """Both return identical arrays, or both raise the same exception."""
    try:
        want = ref(*args)
    except Exception as e:
        with pytest.raises(type(e)) as raised:
            new(*args)
        assert str(raised.value) == str(e)
        return
    assert_identical(new(*args), want)


class CountingBigWig:
    def __init__(self, bw):
        self.bw, self.calls = bw, 0

    def chroms(self, *args):
        return self.bw.chroms(*args)

    def values(self, *args, **kwargs):
        self.calls += 1
        return self.bw.values(*args, **kwargs)


# --- tests ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("width", [100, 201, 1000])
def test_get_seq(data, width):
    d, genome, _ = data
    df = regions(d, width, chroms=list(CHROMS) if width < 700 else BW_CHROMS)
    assert_identical(data_utils.get_seq(df, genome, width), get_seq_ref(df, genome, width))


@pytest.mark.parametrize("gap,span", [(None, None), (0, 1), (40, 300)], ids=["default", "per-region", "small"])
@pytest.mark.parametrize("width", [100, 201, 1000])
def test_get_cts(data, monkeypatch, width, gap, span):
    d, _, bw = data
    if gap is not None:
        monkeypatch.setattr(data_utils, "_CTS_MAX_GAP", gap)
        monkeypatch.setattr(data_utils, "_CTS_MAX_SPAN", span)
    df = regions(d, width)
    want = get_cts_ref(df, bw, width)
    assert np.isinf(bw.values("chr1", 0, CHROMS["chr1"])).sum() and (want == np.finfo(np.float64).max).any()

    counting = CountingBigWig(bw)
    assert_identical(data_utils.get_cts(df, counting, width), want)
    distinct = len(set(zip(df.chr, df.start + df.summit)))
    expected = {None: len(BW_CHROMS), 0: distinct}   # one read per chromosome; one per distinct region
    if gap in expected:
        assert counting.calls == expected[gap]
    else:
        assert len(BW_CHROMS) < counting.calls < distinct

    # float32 straight from the reads, as get_seq_cts_coords stores it: +-inf stays +-inf, as when casting the float64
    with np.errstate(over="ignore"):
        assert_identical(data_utils.get_cts(df, bw, width, dtype=np.float32), want.astype(np.float32))


@pytest.mark.parametrize("width", [201, 1000])
def test_get_cts_region_cap(data, monkeypatch, width):
    """A read holds at most _CTS_MAX_REGIONS regions, however close they are (dense or duplicated regions)."""
    d, _, bw = data
    monkeypatch.setattr(data_utils, "_CTS_MAX_REGIONS", 7)
    df = regions(d, width)
    df = pd.concat([df] + [df[df.chr == "chr2"].iloc[:1]] * 30)   # 30 more copies of one region
    counting = CountingBigWig(bw)
    assert_identical(data_utils.get_cts(df, counting, width), get_cts_ref(df, bw, width))
    assert counting.calls >= -(-len(df) // 7) > len(BW_CHROMS)


@pytest.mark.parametrize("peaks_bool", [0, 1])
def test_get_coords(data, peaks_bool):
    d, _, _ = data
    df = regions(d, 1000)
    got = data_utils.get_coords(df, peaks_bool)
    assert got.dtype == np.dtype("<U{}".format(len(LONG_NAME)))
    assert_identical(got, get_coords_ref(df, peaks_bool))
    short = df[df.chr != LONG_NAME]
    assert_identical(data_utils.get_coords(short, peaks_bool), get_coords_ref(short, peaks_bool))


def test_get_seq_cts_coords(data):
    d, genome, bw = data
    df = regions(d, 1100)   # input windows reach both chromosome ends; the 800 bp count windows stay inside
    got = data_utils.get_seq_cts_coords(df, genome, bw, 1100, 800, 1)
    want = get_seq_ref(df, genome, 1100), get_cts_ref(df, bw, 800).astype(np.float32), get_coords_ref(df, 1)
    for g, w in zip(got, want):
        assert_identical(g, w)


@pytest.mark.parametrize("rows", [
    pytest.param([("chr1", 11_950, 0), ("chr1", 5_000, 0)], id="past-the-end"),
    pytest.param([("chr2", 10, 20), ("chr1", 5_000, 0)], id="negative-start"),
    pytest.param([("chr1", 11_950, 10)], id="only-past-the-end"),
    pytest.param([("chr2", 10, 20)], id="only-negative-start"),
    pytest.param([("chr2", -2_000, 0)], id="wraps-around"),
    pytest.param([("chrM", 350, 0), ("chr1", 5_000, 0)], id="not-in-bigwig"),
    pytest.param([("chrX", 5_000, 0)], id="not-in-either"),
    pytest.param([("chr1", 5_000.0, 7.0)], id="float-coordinates"),
    pytest.param([(0, 5_000, 0), (1, 400, 3)], id="numeric-chromosomes"),
    pytest.param([("chr1", np.int32(4_000), 3), ("chr2", 400, 3)], id="plain-frame"),
])
def test_other_regions(data, rows):
    """Input the windowed read does not take: every function must still do exactly what the loops did."""
    _, genome, bw = data
    df = frame(rows)
    assert_same_outcome(data_utils.get_seq, get_seq_ref, df, genome, 200)
    assert_same_outcome(data_utils.get_cts, get_cts_ref, df, bw, 200)
    assert_same_outcome(data_utils.get_coords, get_coords_ref, df, 1)


def test_empty(data):
    _, genome, bw = data
    df = frame([]).astype({"chr": str, "start": np.int64, "summit": np.int64})
    assert_identical(data_utils.get_seq(df, genome, 200), get_seq_ref(df, genome, 200))
    assert_identical(data_utils.get_cts(df, bw, 200), get_cts_ref(df, bw, 200))
    assert_identical(data_utils.get_coords(df, 0), get_coords_ref(df, 0))
