import os
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CamReader(object):
    """
    Reader class for .cam files.
    Allows transparent access to frames by reading them from the file on the fly (without loading the whole file).
    Supports determining length, indexing (only positive single-element indices) and iteration.
    Args:
        path(str): path to .cam file.
        same_size(bool): if `True`, assume that all frames have the same size, which speeds up random access and obtaining number of frames;
            otherwise, the first time the length is determined or a large-index frame is accessed can take a long time (all subsequent calls are faster).
    """

    def __init__(self, path, same_size=False):
        self.path = self.normalize_path(path)
        self.frame_offsets = [0]
        self.frames_num = None
        self.same_size = same_size
        self.channel_intensities = None

    @staticmethod
    def eof(f, strict=False):
        """
        Standard EOF function.
        Return `True` if the marker is at the end of the file.
        If `strict==True`, only return True if the marker is exactly at the end of file; otherwise, return True if it's at the end or further.
        """
        p = f.tell()
        f.seek(0, 2)
        ep = f.tell()
        f.seek(p)
        return (ep == p) or (ep <= p and not strict)

    def _read_cam_frame(self, f, skip=False):
        size = np.fromfile(f, "<u4", count=2)
        if len(size) == 0 and self.eof(f):
            raise StopIteration
        if len(size) < 2:
            raise IOError("not enough cam data to read the frame size")
        w, h = size
        if not skip:
            data = np.fromfile(f, "<u2", count=w * h)
            if len(data) < w * h:
                raise IOError("not enough cam data to read the frame: {} pixels available instead of {}".format(len(data), w * h))
            return data.reshape((w, h))
        else:
            f.seek(w * h * 2, 1)
            return None

    @staticmethod
    def normalize_path(p):
        """Normalize filesystem path (case and origin). If two paths are identical, they should be equal when normalized."""
        return os.path.normcase(os.path.abspath(p))

    def _read_frame_at(self, offset):
        with open(self.path, "rb") as f:
            f.seek(offset)
            return self._read_cam_frame(f)

    def _read_next_frame(self, f, skip=False):
        data = self._read_cam_frame(f, skip=skip)
        self.frame_offsets.append(f.tell())
        return data

    def _read_frame(self, idx):
        idx = int(idx)
        if self.same_size:
            if len(self.frame_offsets) == 1:
                with open(self.path, "rb") as f:
                    self._read_next_frame(f, skip=True)
            offset = self.frame_offsets[1] * idx
            return self._read_frame_at(offset)
        else:
            if idx < len(self.frame_offsets):
                return self._read_frame_at(self.frame_offsets[idx])
            next_idx = len(self.frame_offsets) - 1
            offset = self.frame_offsets[-1]
            with open(self.path, "rb") as f:
                f.seek(offset)
                while next_idx <= idx:
                    data = self._read_next_frame(f, next_idx < idx)
                    next_idx += 1
            return data

    def _fill_offsets(self):
        if self.frames_num is not None:
            return
        if self.same_size:
            file_size = os.path.getsize(self.path)
            if file_size == 0:
                self.frames_num = 0
            else:
                with open(self.path, "rb") as f:
                    self._read_next_frame(f, skip=True)
                if file_size % self.frame_offsets[1]:
                    raise IOError("File size {} is not a multiple of single frame size {}".format(file_size, self.frame_offsets[1]))
                self.frames_num = file_size // self.frame_offsets[1]
        else:
            offset = self.frame_offsets[-1]
            try:
                with open(self.path, "rb") as f:
                    f.seek(offset)
                    while True:
                        self._read_next_frame(f, skip=True)
            except StopIteration:
                pass
            self.frames_num = len(self.frame_offsets) - 1

    def size(self):
        """Get the total number of frames"""
        self._fill_offsets()
        return self.frames_num

    __len__ = size

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return list(self.iterrange(idx.start or 0, idx.stop, idx.step or 1))
        try:
            return self._read_frame(idx)
        except StopIteration:
            raise IndexError("index {} is out of range".format(idx))

    def get_data(self, idx):
        """Get a single frame at the given index (only non-negative indices are supported)"""
        return self[idx]

    def __iter__(self):
        return self.iterrange()

    def iterrange(self, *args):
        """
        iterrange([start,] stop[, step])
        Iterate over frames starting with start ending at stop (`None` means until the end of file) with the given step.
        """
        start, stop, step = 0, None, 1
        if len(args) == 1:
            stop, = args
        elif len(args) == 2:
            start, stop = args
        elif len(args) == 3:
            start, stop, step = args
        if step < 0:
            raise IndexError("format doesn't support reversed indexing")
        try:
            n = start
            while True:
                yield self._read_frame(n)
                n += step
                if stop is not None and n >= stop:
                    break
        except StopIteration:
            pass

    def read_all(self):
        """Read all available frames"""
        return list(self.iterrange())

    def binned_name(self, pfx, nbin, ext="bin"):
        """Get the name of the file with the binned data for a given prefix, binning number, and extention"""
        return os.path.join(pfx, "binned_{}.{}".format(nbin, ext))

    def decfunc(self, mode):
        """Get the appropriate decimation function"""
        if mode == "max":
            return np.max
        if mode == "mean":
            return np.mean
        if mode == "min":
            return np.min
        if mode == "skip":
            return lambda a, axis: a.take(0, axis=axis)

    def load_and_bin(self, path, nbin, ntot=None, mode="max"):
        """
        Load and bin .cam file.
        Args:
            path: source path
            nbin: binning factor
            ntot: total number of binned frames to load (so that total number of raw frames to read is `ntot*nbin);
                by default, load all frames
            mode: binning mode; can be `"mean", "max", "min", or "skip".
        """
        dec = self.decfunc(mode)
        if ntot is None:
            ntot = len(self) // nbin
        ntot = min(ntot, len(self) // nbin)
        frame = self.get_data(0)
        result = np.full((ntot,) + frame.shape, 0, dtype=float)  # use np.full instead of np.zeros to actually allocate RAM
        for i in range(ntot):
            frames = self[i * nbin:(i + 1) * nbin]
            binned = dec(frames, axis=0)
            result[i] = binned
        return result

    def save_prebinned(self, frames, pfx, nbin):
        """Save pre-binned data"""
        path = self.binned_name(pfx, nbin)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        frames.astype("<f8").tofile(path)

    def load_prebinned(self, pfx, sfx, nbin):
        """Load already pre-binned data"""
        bname = self.binned_name(pfx, nbin)
        if os.path.exists(bname):
            reader = CamReader("{}_{}.cam".format(pfx, sfx), same_size=True)
            return np.fromfile(bname, "<f8").reshape((-1,) + reader[0].shape)
        return None

    def show_movie(self, frames, nshow=None, **kwargs):
        """
        Show movie from the frames.
        All `kwargs` parameters are passed to imshow.
        """
        img = plt.imshow(frames[0], **kwargs)

        def update(i):
            img.set_data(frames[i])

        ani = animation.FuncAnimation(img.axes.figure, update, interval=1E3 / self.fps, frames=nshow or len(frames), repeat=True)
        return ani

    def make_movie(self, sfx='vid', nbin=100, fps=10, name='test', **kwargs):
        """Load and bin data (or load prebinned if exists and `recalc==False), save it, and save a movie"""
        self.fps = fps
        bdata = self.load_and_bin("{}_{}.cam".format(self.path, sfx), nbin)
        bdata -= np.median(bdata, axis=0)  # subtract median background (wide-field fluorescence)
        ani = self.show_movie(bdata, **kwargs)
        ani.save(name + '.mp4', fps=fps)

    def extract_channel(self, channel_h=[0, 10], channel_v=[0, 10]):
        intensity = []
        for i in range(len(self)):
            intensity.append(self[i][channel_h[0]:channel_h[1], channel_v[0]:channel_v[1]].mean())
        return intensity

    def extract_all_channel_intensities(self, pixels=8, pixel_range=None):
        data = []
        if pixel_range is None:
            for i in range(len(self)):
                data.append(zoom(self[i], 1 / pixels))
        else:
            for i in range(len(self)):
                data.append(zoom(self[i][pixel_range[0][0]:pixel_range[0][1], pixel_range[1][0]:pixel_range[1][1]], 1 / pixels))
        data = np.array(data)
        returndata = pd.DataFrame()
        for i in range(np.shape(data[0])[0]):
            for j in range(np.shape(data[0])[0]):
                returndata[str(i) + ' ' + str(j)] = data[:, i, j]
        data = None
        self.channel_intensities = returndata
        self.get_max_channel_intensity(data=returndata, figure=True)
        return returndata

    def length(self):
        return len(self)

    def get_max_channel_intensity(self, data=None, figure=True):
        if data is None:
            data = self.channel_intensities
        maximum = data.max().max()
        col = data.max().idxmax()
        idx = data[col].idxmax()
        print('Maximum in %s at index %i' % (col, idx))
        if figure:
            fig, ax = plt.subplots()
            ax.plot(data[col])
            ax.scatter(idx, maximum, color='tab:red')
            ax.set_ylabel('Intensity (a.u.)')
            ax.set_ylabel('Index')
