import numpy as np

events_struct = [("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)]

# many functions in this file have been copied from https://gitlab.com/synsense/aermanager/-/blob/master/aermanager/parsers.py

def make_structured_array(x, y, t, p, dtype=events_struct):
    """
    Make a structured array given lists of x, y, t, p

    Args:
        x: List of x values
        y: List of y values
        t: List of times
        p: List of polarities boolean
    Returns:
        xytp: numpy structured array
    """
    return np.fromiter(zip(x, y, t, p), dtype=dtype)


def read_mnist_file(bin_file, dtype):
    """
    Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'
    (Code adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py)
    """
    f = open(bin_file, "rb")
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    # Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    xytp = make_structured_array(
        all_x[td_indices],
        all_y[td_indices],
        all_ts[td_indices],
        all_p[td_indices],
        dtype,
    )
    return xytp


def read_fnmnist_file(bin_file, dtype):
    """
    Reads in the TD events contained in the (FE)MNIST dataset file specified by 'filename' (in this file, spikes are
    stored as an unsigned 8 bit Nx5 array and each event is 40 bits).
    Returns an array Nx4 where columns are organized as t,x,y,p.
    (Code adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py and
    https://bitbucket.org/bamsumit/spikefilesreadwrite/src/master/Read_Ndataset.m)
    """
    f = open(bin_file, "rb")
    raw_data = np.uint32(np.fromfile(f, dtype=np.uint8))
    f.close()

    # X-COORDINATES are stored in all 8 bits of array[0::5]
    all_x = (raw_data[0::5] & 255)
    # Y-COORDINATES are stored in all 8 bits of array[1::5]
    all_y = (raw_data[1::5] & 255)
    # POLARITY are stored in only 1 of the 8 bits of array[2::5], the other 7 bits are for the timestamp
    all_p = (raw_data[2::5] & 128) >> 7
    # TIMESTAMPS are stored in 23 bits in total: the last 7 bits of array[2::5] (together with the polarity) +
    # all 8 bits in array[3::5] + all 8 bits in array[4::5]
    all_t = ((raw_data[2::5] & 127) << 16) | ((raw_data[3::5] & 255) << 8) | (raw_data[4::5] & 255)

    xytp = make_structured_array(
        all_x,
        all_y,
        all_t,
        all_p,
        dtype,
    )
    return xytp
