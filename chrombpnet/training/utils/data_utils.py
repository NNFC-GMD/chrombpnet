import numpy as np
import pandas as pd
import pyBigWig
import pyfaidx
from numpy.lib.stride_tricks import sliding_window_view
from chrombpnet.training.utils import one_hot

# get_cts reads nearby regions with one bw.values() call: a call costs ~70 us however short it is, about as much as
# decoding 40 kb of signal. Regions less than _CTS_MAX_GAP apart share a read, and a read covers at most _CTS_MAX_SPAN
# bases plus one region (4 MB of float32 at most) and at most _CTS_MAX_REGIONS regions, which bounds the temporary
# copies for dense or duplicated regions (12 bytes per base: ~25 MB at 2000 bp).
_CTS_MAX_GAP = 1 << 15
_CTS_MAX_SPAN = 1 << 20
_CTS_MAX_REGIONS = 1 << 10


def _columns(peaks_df):
    """
    The chr, start and summit columns, holding the objects the per-row iterrows() loops here used to get, so the
    coordinate arithmetic, the pyfaidx/pyBigWig lookups and get_coords' string dtype stay the same. iterrows() reads
    the rows from peaks_df.values: Python objects when any column is not numeric (chr holds strings), otherwise numpy
    scalars of the frame's common dtype.
    """
    names = ("chr", "start", "summit")
    if peaks_df.iloc[:0].values.dtype == object:
        return [peaks_df[name].tolist() for name in names]
    values = peaks_df.values
    return [values[:, peaks_df.columns.get_loc(name)] for name in names]


def get_seq(peaks_df, genome, width):
    """
    Same as get_cts, but fetches sequence from a given genome.
    """
    # One pyfaidx read per region (a few us): reading whole chromosomes and slicing them is no faster, and would
    # have to redo pyfaidx's handling of regions off the chromosome ends (a negative start wraps around).
    vals = [str(genome[c][(s + m - width//2):(s + m + width//2)]) for c, s, m in zip(*_columns(peaks_df))]
    return one_hot.dna_to_one_hot(vals)


def _get_cts_windowed(peaks_df, bw, width, dtype):
    """
    get_cts for the usual input: string chromosomes that are all in the bigWig, integer coordinates and every region
    inside its chromosome. Returns None for anything else, where pyBigWig would raise or the element types differ,
    and get_cts then reads region by region.
    """
    half = width // 2
    length = 2 * half
    coords = ("start", "summit")
    if (len(peaks_df) == 0 or not isinstance(half, (int, np.integer)) or length <= 0
            or not getattr(pyBigWig, "numpy", 0)   # values(numpy=True) needs pyBigWig built with numpy
            or not all(isinstance(peaks_df[c].dtype, np.dtype) and peaks_df[c].dtype.kind in "iu" for c in coords)):
        return None
    codes, names = pd.factorize(peaks_df["chr"], use_na_sentinel=False)
    sizes = bw.chroms()
    if not all(isinstance(name, str) and name in sizes for name in names):
        return None
    starts = peaks_df["start"].to_numpy(np.int64) + peaks_df["summit"].to_numpy(np.int64) - half
    if not ((starts >= 0) & (starts + length <= np.array([sizes[name] for name in names])[codes])).all():
        return None

    # windows over the regions sorted by chromosome and start: a new one at each chromosome, at each gap over
    # _CTS_MAX_GAP, every _CTS_MAX_SPAN bases of start positions within a run of close regions and every
    # _CTS_MAX_REGIONS regions
    order = np.lexsort((starts, codes))
    codes, starts = codes[order], starts[order]
    new_run = np.ones(len(starts), dtype=bool)
    new_run[1:] = (codes[1:] != codes[:-1]) | (starts[1:] - starts[:-1] > length + _CTS_MAX_GAP)
    run = np.cumsum(new_run) - 1
    span = (starts - starts[new_run][run]) // _CTS_MAX_SPAN
    new_window = new_run.copy()
    new_window[1:] |= span[1:] != span[:-1]
    new_window[::_CTS_MAX_REGIONS] = True
    bounds = np.append(np.flatnonzero(new_window), len(starts))

    out = np.empty((len(starts), length), dtype=dtype)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        first = int(starts[lo])
        vals = bw.values(names[codes[lo]], first, int(starts[hi - 1]) + length, numpy=True)
        # float64 before nan_to_num, as the per-region values were: it maps +-inf to +-float64 max
        cts = sliding_window_view(vals, length)[starts[lo:hi] - first].astype(np.float64)
        out[order[lo:hi]] = np.nan_to_num(cts, copy=False)
    return out


def get_cts(peaks_df, bw, width, dtype=np.float64):
    """
    Fetches values from a bigwig bw, given a df with minimally
    chr, start and summit columns. Summit is relative to start.
    Retrieves values of specified width centered at summit.

    "cts" = per base counts across a region

    dtype=np.float32 returns exactly get_cts(...).astype(np.float32) without making the float64 array.
    """
    cts = _get_cts_windowed(peaks_df, bw, width, dtype)
    if cts is not None:
        return cts
    vals = [np.nan_to_num(bw.values(c, s + m - width//2, s + m + width//2)) for c, s, m in zip(*_columns(peaks_df))]
    return np.array(vals).astype(dtype, copy=False)

def get_coords(peaks_df, peaks_bool):
    """
    Fetch the co-ordinates of the regions in bed file
    returns a list of tuples with (chrom, summit)
    """
    return np.array([[c, s + m, "f", peaks_bool] for c, s, m in zip(*_columns(peaks_df))])

def get_seq_cts_coords(peaks_df, genome, bw, input_width, output_width, peaks_bool):

    seq = get_seq(peaks_df, genome, input_width)
    # bigWig values are float32: storing them as float32 (not float64) is exact and halves the counts' memory
    cts = get_cts(peaks_df, bw, output_width, dtype=np.float32)
    coords = get_coords(peaks_df, peaks_bool)
    return seq, cts, coords

def load_data(bed_regions, nonpeak_regions, genome_fasta, cts_bw_file, inputlen, outputlen, max_jitter):
    """
    Load sequences and corresponding base resolution counts for training, 
    validation regions in peaks and nonpeaks (2 x 2 x 2 = 8 matrices).

    For training peaks/nonpeaks, values for inputlen + 2*max_jitter and outputlen + 2*max_jitter 
    are returned centered at peak summit. This allows for jittering examples by randomly
    cropping. Data of width inputlen/outputlen is returned for validation
    data.

    If outliers is not None, removes training examples with counts > outlier%ile
    """

    cts_bw = pyBigWig.open(cts_bw_file)
    genome = pyfaidx.Fasta(genome_fasta)

    train_peaks_seqs=None
    train_peaks_cts=None
    train_peaks_coords=None
    train_nonpeaks_seqs=None
    train_nonpeaks_cts=None
    train_nonpeaks_coords=None

    if bed_regions is not None:
        train_peaks_seqs, train_peaks_cts, train_peaks_coords = get_seq_cts_coords(bed_regions,
                                              genome,
                                              cts_bw,
                                              inputlen+2*max_jitter,
                                              outputlen+2*max_jitter,
                                              peaks_bool=1)
    
    if nonpeak_regions is not None:
        train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords = get_seq_cts_coords(nonpeak_regions,
                                              genome,
                                              cts_bw,
                                              inputlen,
                                              outputlen,
                                              peaks_bool=0)



    cts_bw.close()
    genome.close()

    return (train_peaks_seqs, train_peaks_cts, train_peaks_coords,
            train_nonpeaks_seqs, train_nonpeaks_cts, train_nonpeaks_coords)
