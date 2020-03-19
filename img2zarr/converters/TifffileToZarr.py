import numpy as np
import zarr
import dask.array as da
import tifffile
from concurrent.futures import ThreadPoolExecutor
import shutil
from pathlib import Path


def asarrayZarr(
    TiffPage, zarr_store, lock=None, reopen=True, maxsize=None, maxworkers=None,
):
    """Read image data from file and stream into a zarr store


    Parameters
    ----------
    TiffPage : tifffile.TiffPage
        Buffer where image data will be saved.
        If None (default), a new array will be created.
        If numpy.ndarray, a writable array of compatible dtype and shape.
        If 'memmap', directly memory-map the image data in the TIFF file
        if possible; else create a memory-mapped array in a temporary file.
        If str or open file, the file name or file object used to
        create a memory-map to an array stored in a binary file on disk.
    zarr_store : zarr.Array
        Zarr store where image data will be saved.
    lock : {RLock, NullContext}
        A reentrant lock used to synchronize seeks and reads from file.
        If None (default), the lock of the parent's filehandle is used.
    reopen : bool
        If True (default) and the parent file handle is closed, the file
        is temporarily re-opened and closed if no exception occurs.
    maxsize: int
        Maximum size of data before a ValueError is raised.
        Can be used to catch DOS. Default: 16 TB.
    maxworkers : int or None
        Maximum number of threads to concurrently decode strips ot tiles.
        If None (default), up to half the CPU cores are used.
        See remarks in TiffFile.asarray.

    """
    keyframe = TiffPage.keyframe  # TiffPage or keyframe
    product = tifffile.product
    if not keyframe.shaped or product(keyframe.shaped) == 0:
        return None

    fh = TiffPage.parent.filehandle
    if lock is None:
        lock = fh.lock
    with lock:
        closed = fh.closed
        if closed:
            if reopen:
                fh.open()
            else:
                raise OSError(f"TiffPage {TiffPage.index}: file handle is closed")

    if keyframe.is_contiguous:
        # read contiguous bytes to array
        if keyframe.is_subsampled:
            raise NotImplementedError(
                f"TiffPage {TiffPage.index}: chroma subsampling not supported"
            )
        with lock:
            fh.seek(TiffPage._offsetscounts[0][0])
            result = fh.read_array(
                keyframe.parent.byteorder + keyframe._dtype.char,
                product(keyframe.shaped),
                out=None,
            )
        zarr_store[:] = result.reshape(keyframe.shaped)
    else:
        # decode individual strips or tiles
        decodeargs = {}
        decodefunc = keyframe.decode
        #    result = tifffile.create_output(None, keyframe.shaped, keyframe._dtype)
        out = zarr_store[0]
        if keyframe.compression in (6, 7):  # COMPRESSION.JPEG
            if 347 in keyframe.tags:
                # lazy load JPEGTables for TiffFrame
                decodeargs["tables"] = TiffPage._gettags({347}, lock=lock)[0][1].value
            # TODO: obtain table from OJPEG tags
            # elif ('JPEGInterchangeFormat' in tags and
            #       'JPEGInterchangeFormatLength' in tags and
            #       tags['JPEGInterchangeFormat'].value != offsets[0]):
            #     fh.seek(tags['JPEGInterchangeFormat'].value)
            #     fh.read(tags['JPEGInterchangeFormatLength'].value)

        def decode(args):
            # decode strip or tile and store in result array
            # TODO: use flat indexing for strips?
            segment, (_, s, d, l, w, _), shape = decodefunc(*args, **decodeargs)
            if segment is None:
                segment = keyframe.nodata
            else:
                segment = segment[
                    : keyframe.imagedepth - d,
                    : keyframe.imagelength - l,
                    : keyframe.imagewidth - w,
                ]
            zarr_store[
                s, d : d + shape[0], l : l + shape[1], w : w + shape[2]
            ] = segment

        offsets, bytecounts = TiffPage._offsetscounts
        segmentiter = fh.read_segments(offsets, bytecounts, lock=lock)

        if maxworkers is None or maxworkers < 1:
            maxworkers = keyframe.maxworkers
        if maxworkers < 2:
            for segment in segmentiter:
                decode(segment)
        else:
            # decode first segment un-threaded to catch exceptions
            decode(next(segmentiter))
            with ThreadPoolExecutor(maxworkers) as executor:
                executor.map(decode, segmentiter)

    return


class TiffFileToZarr:
    def __init__(self, img_path):
        self.img_path = Path(img_path)

        self.tf_im = tifffile.TiffFile(str(self.img_path))

        self.base_series_idx = self._series_check()
        self.base_series = self.tf_im.series[self.base_series_idx]

        # essential information for storage
        self.dim_order = self.base_series.axes
        self.is_rgb = self._rgb_check(self.base_series)
        self.shape = self.base_series.shape
        self.dtype = self.base_series.dtype

    # some data may already be pyramidal and stored as a 'series'
    # this returns the series with the largest XY value a the base plane
    # but in this instance the pyramid should just be entirely written to zarr
    def _series_check(self):
        def _n_px_plane(series):
            # find X,Y axes
            y_dim_idx, x_dim_idx = series.axes.index("Y"), series.axes.index("X")
            return np.prod(np.array(series.shape)[[y_dim_idx, x_dim_idx]])

        if len(self.tf_im.series) > 1:
            return np.argmax([_n_px_plane(series) for series in self.tf_im.series])
        else:
            return 0

    # uses tiff photometric tag to indicate RGB or not
    def _rgb_check(self, series):
        photometric = self.base_series._pages[0].photometric
        return True if photometric.name == "RGB" else False

    # read tiff directly to zarr store
    def tiff_to_zarr(self, zarr_store_path, name="image_name", tile_size=512):
        store_path = Path(zarr_store_path)
        z = zarr.open_group(zarr_store_path, mode="a")
        root = z.create_group(name, overwrite=False)

        # get dimension indexing
        x_dim = self.dim_order.index("X")
        y_dim = self.dim_order.index("Y")
        chunking = np.ones_like(self.shape, dtype=np.int)
        if self.is_rgb is True:
            root.attrs["rgb"] = True
            # there could potentially be multi time, or multi z stack RGBs
            if len(self.shape) == 3:
                ch_idx = np.argmin(self.shape)
                chunking[:2] = tile_size

            else:
                try:
                    ch_idx = self.dim_order.index("S")
                except ValueError:
                    try:
                        ch_idx = self.dim_order.index("C")
                    except ValueError:
                        raise ValueError("channel index not found in dimensions")

                    chunking[-3] = tile_size
                    chunking[-2] = tile_size
                    chunking[-1] = self.shape[ch_idx]
                    chunking = tuple([int(chunk) for chunk in chunking])

        else:
            chunking[-2] = tile_size
            chunking[-1] = tile_size
            # make sure numbers are JSON serializable
            chunking = tuple([int(chunk) for chunk in chunking])

        # get the shape the data comes during decode, before squeezing by tifffile
        n_pages = len(self.base_series.pages)
        tempstores = ["tempbase_{}".format(str(idx).zfill(3)) for idx in range(n_pages)]
        temp_store_shape = self.base_series.pages[0].shaped
        for idx, page in enumerate(self.base_series.pages):

            temp_base_store = root.create_dataset(
                tempstores[idx],
                shape=temp_store_shape,
                chunks=True,
                dtype=self.dtype,
                overwrite=True,
            )

            asarrayZarr(page, temp_base_store, maxworkers=1)

        temp_dask_arr = da.stack(
            [
                da.squeeze(da.from_zarr(str(store_path / name / store)))
                for store in tempstores
            ]
        )

        # reorder dask array for rechunking and storing in zarr
        if self.is_rgb and ch_idx != len(self.shape) - 1:
            temp_dask_arr = temp_dask_arr.swapaxes(0, 2).swapaxes(0, 1)

        temp_dask_arr.rechunk(chunking).to_zarr(str(store_path / name / "0"))

        [shutil.rmtree(temp) for temp in tempstores]
