"""STISFitsReader / ReaderCollection のテスト."""

from pathlib import Path
import numpy as np
import pytest

from stis_analysis.core.fits_reader import STISFitsReader, ReaderCollection


class TestSTISFitsReader:
    def test_open_returns_reader(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        assert isinstance(reader, STISFitsReader)
        assert reader.filename == sample_fits_path

    def test_headers_loaded_for_all_hdus(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        # HDU 0〜3 すべてのヘッダーが読み込まれている
        for i in range(4):
            assert i in reader.headers

    def test_data_loaded_for_image_hdus(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        # HDU 1〜3 はデータあり
        for i in range(1, 4):
            assert i in reader.data
            assert isinstance(reader.data[i], np.ndarray)

    def test_primary_hdu_has_no_data(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        # HDU 0 (PrimaryHDU) はデータなし
        assert 0 not in reader.data

    def test_header_returns_correct_header(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        h = reader.header(0)
        assert h["ROOTNAME"] == "test_root"

    def test_header_raises_for_missing_hdu(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        with pytest.raises(KeyError, match="HDU 99"):
            reader.header(99)

    def test_image_data_returns_array(self, sample_fits_path: Path, fits_shape):
        reader = STISFitsReader.open(sample_fits_path)
        data = reader.image_data(1)
        assert data.shape == fits_shape

    def test_image_data_raises_for_missing_hdu(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        with pytest.raises(KeyError, match="HDU 99"):
            reader.image_data(99)

    def test_spectrum_data_returns_three_arrays(self, sample_fits_path: Path, fits_shape):
        reader = STISFitsReader.open(sample_fits_path)
        sci, err, dq = reader.spectrum_data()
        for arr in (sci, err, dq):
            assert arr.shape == fits_shape

    def test_info_contains_filename(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        info = reader.info()
        assert str(sample_fits_path) in info

    def test_info_contains_hdu_shapes(self, sample_fits_path: Path, fits_shape):
        reader = STISFitsReader.open(sample_fits_path)
        info = reader.info()
        assert str(fits_shape) in info

    def test_lac_fits_with_cr_mask_hdu4(self, tmp_path: Path, fits_shape):
        """_lac.fits の HDU 4 (LACOSMIC マスク) も正しく読み込めること."""
        from astropy.io import fits as astropy_fits
        rows, cols = fits_shape
        cr_header = astropy_fits.Header()
        cr_header["EXTNAME"] = "LACOSMIC"
        hdul = astropy_fits.HDUList([
            astropy_fits.PrimaryHDU(),
            astropy_fits.ImageHDU(data=np.ones((rows, cols), dtype=np.float32)),
            astropy_fits.ImageHDU(data=np.ones((rows, cols), dtype=np.float32)),
            astropy_fits.ImageHDU(data=np.zeros((rows, cols), dtype=np.int16)),
            astropy_fits.ImageHDU(
                data=np.zeros((rows, cols), dtype=np.uint8), header=cr_header
            ),
        ])
        path = tmp_path / "test_lac.fits"
        hdul.writeto(path)

        reader = STISFitsReader.open(path)
        assert 4 in reader.data
        assert reader.header(4)["EXTNAME"] == "LACOSMIC"


class TestReaderCollection:
    def test_from_paths_creates_collection(self, sample_fits_path: Path):
        col = ReaderCollection.from_paths([sample_fits_path])
        assert len(col) == 1

    def test_from_paths_multiple_files(self, tmp_path: Path, fits_shape):
        from astropy.io import fits as astropy_fits
        rows, cols = fits_shape
        paths = []
        for i in range(3):
            p = tmp_path / f"file_{i}.fits"
            hdul = astropy_fits.HDUList([
                astropy_fits.PrimaryHDU(),
                astropy_fits.ImageHDU(data=np.ones((rows, cols), dtype=np.float32)),
                astropy_fits.ImageHDU(data=np.ones((rows, cols), dtype=np.float32)),
                astropy_fits.ImageHDU(data=np.zeros((rows, cols), dtype=np.int16)),
            ])
            hdul.writeto(p)
            paths.append(p)

        col = ReaderCollection.from_paths(paths)
        assert len(col) == 3

    def test_getitem(self, sample_fits_path: Path):
        col = ReaderCollection.from_paths([sample_fits_path])
        assert isinstance(col[0], STISFitsReader)

    def test_iter(self, sample_fits_path: Path):
        col = ReaderCollection.from_paths([sample_fits_path])
        readers = list(col)
        assert len(readers) == 1
        assert isinstance(readers[0], STISFitsReader)

    def test_info_contains_all_files(self, tmp_path: Path, fits_shape):
        from astropy.io import fits as astropy_fits
        rows, cols = fits_shape
        paths = []
        for i in range(2):
            p = tmp_path / f"file_{i}.fits"
            hdul = astropy_fits.HDUList([
                astropy_fits.PrimaryHDU(),
                astropy_fits.ImageHDU(data=np.ones((rows, cols), dtype=np.float32)),
                astropy_fits.ImageHDU(data=np.ones((rows, cols), dtype=np.float32)),
                astropy_fits.ImageHDU(data=np.zeros((rows, cols), dtype=np.int16)),
            ])
            hdul.writeto(p)
            paths.append(p)

        col = ReaderCollection.from_paths(paths)
        info = col.info()
        for p in paths:
            assert str(p) in info
