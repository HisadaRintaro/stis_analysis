"""共通テストフィクスチャ.

最小構成の STIS FITS ファイル（HDU 0〜3）をメモリ上に作成し、
一時ディレクトリへ書き出す pytest フィクスチャを提供する。
"""

from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits


def _make_primary_header() -> fits.Header:
    h = fits.Header()
    h["ROOTNAME"] = "test_root"
    h["INSTRUME"] = "STIS"
    return h


def _make_sci_header(shape: tuple[int, int]) -> fits.Header:
    h = fits.Header()
    h["ROOTNAME"] = "test_root"  # 実際の STIS ファイルでは SCI extension にも存在する
    h["NAXIS"] = 2
    h["NAXIS1"] = shape[1]
    h["NAXIS2"] = shape[0]
    h["CRVAL1"] = 4990.0   # 参照波長 [Å]
    h["CDELT1"] = 0.554    # 波長分散 [Å/pixel]
    h["CRPIX1"] = 1.0
    return h


def _make_err_header(shape: tuple[int, int]) -> fits.Header:
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = shape[1]
    h["NAXIS2"] = shape[0]
    return h


def _make_dq_header(shape: tuple[int, int]) -> fits.Header:
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = shape[1]
    h["NAXIS2"] = shape[0]
    return h


@pytest.fixture
def fits_shape() -> tuple[int, int]:
    """テスト用画像の形状 (rows, cols)."""
    return (10, 20)


@pytest.fixture
def sample_fits_path(tmp_path: Path, fits_shape: tuple[int, int]) -> Path:
    """最小構成の _crj.fits ファイルパス（HDU 0〜3）."""
    rows, cols = fits_shape
    rng = np.random.default_rng(42)

    sci_data = rng.uniform(100, 1000, size=(rows, cols)).astype(np.float32)
    err_data = rng.uniform(5, 50, size=(rows, cols)).astype(np.float32)
    dq_data = np.zeros((rows, cols), dtype=np.int16)
    # hot pixel を1箇所埋め込む（bit 4 = 16）
    dq_data[3, 5] = 16

    hdul = fits.HDUList([
        fits.PrimaryHDU(header=_make_primary_header()),
        fits.ImageHDU(data=sci_data, header=_make_sci_header(fits_shape)),
        fits.ImageHDU(data=err_data, header=_make_err_header(fits_shape)),
        fits.ImageHDU(data=dq_data, header=_make_dq_header(fits_shape)),
    ])
    path = tmp_path / "test_crj.fits"
    hdul.writeto(path)
    return path


@pytest.fixture
def sample_fits_path_no_err_dq(tmp_path: Path, fits_shape: tuple[int, int]) -> Path:
    """HDU 1 (sci) のみの最小 FITS ファイルパス（err/dq なし）."""
    rows, cols = fits_shape
    sci_data = np.ones((rows, cols), dtype=np.float32) * 500.0

    hdul = fits.HDUList([
        fits.PrimaryHDU(header=_make_primary_header()),
        fits.ImageHDU(data=sci_data, header=_make_sci_header(fits_shape)),
    ])
    path = tmp_path / "test_sci_only.fits"
    hdul.writeto(path)
    return path
