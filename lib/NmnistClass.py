import numpy as np
from sklearn.neighbors import NearestNeighbors
from random import shuffle


class NmnistDataset:
    """
    A class to handle N-MNIST dataset events.
    """
    class Events:
        """
        Temporal Difference events.
        data: a NumPy Record Array with the following named fields
            x: pixel x coordinate, unsigned 16bit int
            y: pixel y coordinate, unsigned 16bit int
            p: polarity value, boolean. False=off, True=on
            ts: timestamp in microseconds, unsigned 64bit int
        width: The width of the frame. Default = 34 (for N-MNIST).
        height: The height of the frame. Default = 34 (for N-MNIST).
        """
        def __init__(self, num_events, width=34, height=34):
            self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.uint64)], shape=(num_events))
            self.width = width
            self.height = height

    def __init__(self, filename):
        self.filename = filename
        self.events = self.read_dataset(filename)

    @classmethod
    def read_dataset(cls, filename):
        """Reads in the TD events contained in the N-MNIST dataset file specified by 'filename'"""
        with open(filename, 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)

        raw_data = np.uint32(raw_data)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | raw_data[4::5]

        # Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        # Filter out overflow events
        td_indices = np.where(all_y != 240)[0]

        td = cls.Events(td_indices.size, 34, 34)
        td.data.x = all_x[td_indices]
        td.data.y = all_y[td_indices]
        td.data.ts = all_ts[td_indices]
        td.data.p = all_p[td_indices]

        return td

    
    def get_events(self):
        return self.events
