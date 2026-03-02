"""ImageUnit のテスト."""

import numpy as np
import pytest
from astropy.io import fits

from stis_analysis.core.image import ImageUnit


@pytest.fixture
def basic_header() -> fits.Header:
    h = fits.Header()
    h["NAXIS1"] = 20
    h["NAXIS2"] = 10
    return h


@pytest.fixture
def wcs_header() -> fits.Header:
    """CRVAL1/CDELT1/CRPIX1 を持つ WCS ヘッダー."""
    h = fits.Header()
    h["NAXIS1"] = 20
    h["NAXIS2"] = 10
    h["CRVAL1"] = 5000.0   # 参照波長 [Å]
    h["CDELT1"] = 1.0      # 1 Å/pixel
    h["CRPIX1"] = 1.0
    return h


@pytest.fixture
def sample_unit(basic_header: fits.Header) -> ImageUnit:
    data = np.ones((10, 20), dtype=np.float32)
    return ImageUnit(data=data, header=basic_header)


class TestImageUnitConstruction:
    def test_create(self, sample_unit: ImageUnit):
        assert sample_unit.data.shape == (10, 20)

    def test_repr_contains_shape(self, sample_unit: ImageUnit):
        r = repr(sample_unit)
        assert "(10, 20)" in r


class TestImageUnitNaxis:
    def test_naxis1(self, sample_unit: ImageUnit):
        assert sample_unit.naxis1 == 20

    def test_naxis2(self, sample_unit: ImageUnit):
        assert sample_unit.naxis2 == 10

    def test_naxis_zero_when_missing(self):
        unit = ImageUnit(data=np.zeros((5, 5)), header=fits.Header())
        assert unit.naxis1 == 0
        assert unit.naxis2 == 0


class TestImageUnitWavelength:
    def test_wavelength_returns_array(self, wcs_header: fits.Header):
        data = np.ones((10, 20), dtype=np.float32)
        unit = ImageUnit(data=data, header=wcs_header)
        wave = unit.wavelength
        assert wave is not None
        assert len(wave) == 20  # NAXIS1 = 20 pixels

    def test_wavelength_first_value(self, wcs_header: fits.Header):
        """CRPIX1=1, CRVAL1=5000, CDELT1=1 → pixel 0 = 5000 Å."""
        data = np.ones((10, 20), dtype=np.float32)
        unit = ImageUnit(data=data, header=wcs_header)
        assert unit.wavelength[0] == pytest.approx(5000.0)

    def test_wavelength_uses_cd1_1(self):
        """CDELT1 がない場合 CD1_1 にフォールバックすること."""
        h = fits.Header()
        h["NAXIS1"] = 10
        h["NAXIS2"] = 5
        h["CRVAL1"] = 4000.0
        h["CD1_1"] = 0.5
        h["CRPIX1"] = 1.0
        unit = ImageUnit(data=np.ones((5, 10)), header=h)
        assert unit.wavelength is not None
        assert unit.wavelength[0] == pytest.approx(4000.0)

    def test_wavelength_none_when_no_wcs(self, basic_header: fits.Header):
        """CRVAL1/CDELT1 がない場合 None を返すこと."""
        unit = ImageUnit(data=np.ones((10, 20)), header=basic_header)
        assert unit.wavelength is None


class TestImageUnitToHdu:
    def test_to_hdu_returns_image_hdu(self, sample_unit: ImageUnit):
        hdu = sample_unit.to_hdu()
        assert isinstance(hdu, fits.ImageHDU)

    def test_to_hdu_preserves_data(self, sample_unit: ImageUnit):
        hdu = sample_unit.to_hdu()
        np.testing.assert_array_equal(hdu.data, sample_unit.data)

    def test_to_hdu_converts_bool_to_uint8(self, basic_header: fits.Header):
        """bool 型の data は uint8 に変換されること."""
        bool_data = np.array([[True, False], [False, True]], dtype=bool)
        unit = ImageUnit(data=bool_data, header=basic_header)
        hdu = unit.to_hdu()
        assert hdu.data.dtype == np.uint8
        np.testing.assert_array_equal(hdu.data, bool_data.astype(np.uint8))

    def test_to_hdu_float_unchanged(self, basic_header: fits.Header):
        """float 型の data はそのまま保持されること."""
        float_data = np.ones((10, 20), dtype=np.float32)
        unit = ImageUnit(data=float_data, header=basic_header)
        hdu = unit.to_hdu()
        assert hdu.data.dtype == np.float32
