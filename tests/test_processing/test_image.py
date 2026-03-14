"""ProcessingImageModel / ProcessingImageCollection のテスト."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from stis_analysis.core.fits_reader import STISFitsReader, ReaderCollection
from stis_analysis.lacosmic.image import ImageModel
from stis_analysis.processing.image import ProcessingImageModel, ProcessingImageCollection
from stis_analysis.processing.wave_constants import (
    c_kms,
    oiii5007_stp,
    oiii4959_stp,
    oiii5007_oiii4959,
)


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

NGC1068_V = 1148.0  # km/s

CONTINUUM_WINDOWS = [(-4000, -3200), (3000, 4000)]


@pytest.fixture
def proc_fits_shape() -> tuple[int, int]:
    """テスト用画像の形状 (rows, cols): 空間方向 × 波長方向."""
    return (8, 200)


@pytest.fixture
def proc_fits_path(tmp_path: Path, proc_fits_shape: tuple[int, int]) -> Path:
    """WCS 情報付きの最小 _lac.fits フィクスチャ.

    NGC1068 の OIII λ5007 が画像中央付近になるよう CRVAL1/CDELT1 を設定する。
    - CDELT1 = 0.554 Å/pixel
    - CRVAL1 = 5006.843 * (1 + 1148/c_kms) ≈ 5025.98 Å (4959 と 5007 が両方含まれる範囲)
    - スペクトルは定数 500 + 線形バイアス + ガウス状輝線
    """
    rows, cols = proc_fits_shape
    cdelt1 = 0.554  # Å/pixel
    z = NGC1068_V / c_kms
    # 波長範囲: ±50 Å 程度を中央に OIII 5007 obs が来るよう設定
    center_lam = oiii5007_stp * (1.0 + z)
    crval1 = center_lam - cdelt1 * (cols // 2)

    wavelength = crval1 + cdelt1 * np.arange(cols)

    # sci: 定数 + 線形バイアス(連続光) + ガウス輝線(5007, 4959)
    sci_data = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        # 連続光: 500 + row*0.5 + 0.3 * (pixel - cols/2)
        continuum = 500.0 + row * 0.5 + 0.3 * (np.arange(cols) - cols / 2.0)
        # OIII 5007 ガウス輝線
        lam5007_obs = oiii5007_stp * (1.0 + z)
        sigma = 1.5 * cdelt1
        gauss5007 = 2000.0 * np.exp(-0.5 * ((wavelength - lam5007_obs) / sigma) ** 2)
        # OIII 4959 (5007/2.98)
        lam4959_obs = oiii4959_stp * (1.0 + z)
        gauss4959 = (2000.0 / oiii5007_oiii4959) * np.exp(
            -0.5 * ((wavelength - lam4959_obs) / sigma) ** 2
        )
        sci_data[row] = (continuum + gauss5007 + gauss4959).astype(np.float32)

    h_primary = fits.Header()
    h_primary["ROOTNAME"] = "test_proc"
    h_primary["INSTRUME"] = "STIS"

    h_sci = fits.Header()
    h_sci["ROOTNAME"] = "test_proc"
    h_sci["NAXIS"] = 2
    h_sci["NAXIS1"] = cols
    h_sci["NAXIS2"] = rows
    h_sci["CRVAL1"] = crval1
    h_sci["CDELT1"] = cdelt1
    h_sci["CRPIX1"] = 1.0

    hdul = fits.HDUList([
        fits.PrimaryHDU(header=h_primary),
        fits.ImageHDU(data=sci_data, header=h_sci),
    ])
    path = tmp_path / "test_lac.fits"
    hdul.writeto(path)
    return path


@pytest.fixture
def proc_model(proc_fits_path: Path) -> ProcessingImageModel:
    reader = STISFitsReader.open(proc_fits_path)
    model = ImageModel.from_reader(reader)
    return ProcessingImageModel.setup(
        model,
        recession_velocity=NGC1068_V,
        continuum_windows_kms=CONTINUUM_WINDOWS,
    )


# ---------------------------------------------------------------------------
# subtract_continuum
# ---------------------------------------------------------------------------


class TestSubtractContinuum:
    def test_returns_new_instance(self, proc_model: ProcessingImageModel):
        result = proc_model.subtract_continuum()
        assert result is not proc_model

    def test_header_contsub_true(self, proc_model: ProcessingImageModel):
        result = proc_model.subtract_continuum()
        assert result.primary_header.get("CONTSUB") is True

    def test_header_contdeg_stored(self, proc_model: ProcessingImageModel):
        result = proc_model.subtract_continuum(degree=1)
        assert result.primary_header.get("CONTDEG") == 1

    def test_window_keywords_stored(self, proc_model: ProcessingImageModel):
        result = proc_model.subtract_continuum()
        assert result.primary_header.get("CWIN0LO") == pytest.approx(-4000.0)
        assert result.primary_header.get("CWIN0HI") == pytest.approx(-3200.0)
        assert result.primary_header.get("CWIN1LO") == pytest.approx(3000.0)
        assert result.primary_header.get("CWIN1HI") == pytest.approx(4000.0)

    def test_continuum_level_reduced(self, proc_model: ProcessingImageModel):
        """純粋な連続光を差し引いた後の平均値が差し引き前より小さくなること."""
        result = proc_model.subtract_continuum()
        # 輝線から離れた領域では差し引き後の値が 0 に近い
        # continuum window 内のピクセルを確認
        wavelength = proc_model.sci.wavelength
        z = NGC1068_V / c_kms
        lam_ref = oiii5007_stp * (1.0 + z)
        velocity = c_kms * (wavelength / lam_ref - 1.0)
        cont_mask = ((velocity >= -4000) & (velocity <= -3200)) | (
            (velocity >= 3000) & (velocity <= 4000)
        )
        # 連続光差し引き後のウィンドウ内の値は 0 付近（ガウス輝線は含まれない）
        mean_before = float(np.abs(proc_model.sci.data[:, cont_mask]).mean())
        mean_after = float(np.abs(result.sci.data[:, cont_mask]).mean())
        assert mean_after < mean_before

    def test_raises_when_no_wcs(self, tmp_path: Path):
        """WCS がない場合 ValueError が発生すること."""
        h = fits.Header()
        h["NAXIS"] = 2
        h["NAXIS1"] = 20
        h["NAXIS2"] = 5
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((5, 20), dtype=np.float32), header=h),
        ])
        path = tmp_path / "no_wcs.fits"
        hdul.writeto(path)
        reader = STISFitsReader.open(path)
        model = ImageModel.from_reader(reader)
        proc = ProcessingImageModel.setup(
            model,
            recession_velocity=NGC1068_V,
            continuum_windows_kms=[(-500, -200)],
        )
        with pytest.raises(ValueError, match="WCS"):
            proc.subtract_continuum()


# ---------------------------------------------------------------------------
# remove_o3_4959
# ---------------------------------------------------------------------------


class TestRemoveO34959:
    def test_returns_new_instance(self, proc_model: ProcessingImageModel):
        result = proc_model.remove_o3_4959()
        assert result is not proc_model

    def test_header_o3corr_true(self, proc_model: ProcessingImageModel):
        result = proc_model.remove_o3_4959()
        assert result.primary_header.get("O3CORR") is True

    def test_header_o3scale_stored(self, proc_model: ProcessingImageModel):
        result = proc_model.remove_o3_4959()
        expected = 1.0 / oiii5007_oiii4959
        assert result.primary_header.get("O3SCALE") == pytest.approx(expected, rel=1e-4)

    def test_custom_scale(self, proc_model: ProcessingImageModel):
        result = proc_model.remove_o3_4959(scale=0.5)
        assert result.primary_header.get("O3SCALE") == pytest.approx(0.5)

    def test_4959_region_reduced(self, proc_model: ProcessingImageModel):
        """4959 近傍のピクセル値が除去後に減少すること."""
        result = proc_model.remove_o3_4959()
        wavelength = proc_model.sci.wavelength
        z = NGC1068_V / c_kms
        lam4959_obs = oiii4959_stp * (1.0 + z)
        near_4959 = np.abs(wavelength - lam4959_obs) < 5.0
        if near_4959.sum() > 0:
            mean_before = float(proc_model.sci.data[:, near_4959].mean())
            mean_after = float(result.sci.data[:, near_4959].mean())
            assert mean_after < mean_before

    def test_shape_unchanged(self, proc_model: ProcessingImageModel):
        result = proc_model.remove_o3_4959()
        assert result.shape == proc_model.shape


# ---------------------------------------------------------------------------
# clip_velocity_range
# ---------------------------------------------------------------------------


class TestClipVelocityRange:
    def test_returns_new_instance(self, proc_model: ProcessingImageModel):
        result = proc_model.clip_velocity_range(v_min=-2500, v_max=2500)
        assert result is not proc_model

    def test_wavelength_axis_reduced(self, proc_model: ProcessingImageModel):
        """velocity clipping 後に波長方向ピクセル数が減少すること."""
        result = proc_model.clip_velocity_range(v_min=-2500, v_max=2500)
        assert result.shape[1] < proc_model.shape[1]

    def test_spatial_axis_unchanged(self, proc_model: ProcessingImageModel):
        """空間方向の行数は変わらないこと."""
        result = proc_model.clip_velocity_range(v_min=-2500, v_max=2500)
        assert result.shape[0] == proc_model.shape[0]

    def test_header_vclip_true(self, proc_model: ProcessingImageModel):
        result = proc_model.clip_velocity_range(v_min=-2500, v_max=2500)
        assert result.primary_header.get("VCLIP") is True

    def test_header_vclipmin_vclipmax(self, proc_model: ProcessingImageModel):
        result = proc_model.clip_velocity_range(v_min=-2500.0, v_max=2500.0)
        assert result.primary_header.get("VCLIPMIN") == pytest.approx(-2500.0)
        assert result.primary_header.get("VCLIPMAX") == pytest.approx(2500.0)

    def test_header_vreces_stored(self, proc_model: ProcessingImageModel):
        result = proc_model.clip_velocity_range(v_min=-2500, v_max=2500)
        assert result.primary_header.get("VRECES") == pytest.approx(NGC1068_V)

    def test_header_vclipref_stored(self, proc_model: ProcessingImageModel):
        result = proc_model.clip_velocity_range(v_min=-2500, v_max=2500)
        assert result.primary_header.get("VCLIPREF") == pytest.approx(oiii5007_stp)

    def test_header_vclipz_correct(self, proc_model: ProcessingImageModel):
        result = proc_model.clip_velocity_range(v_min=-2500, v_max=2500)
        expected_z = NGC1068_V / c_kms
        assert result.primary_header.get("VCLIPZ") == pytest.approx(expected_z, rel=1e-5)

    def test_wcs_crval1_updated(self, proc_model: ProcessingImageModel):
        """CRVAL1 が clipped 後の最初のピクセル波長に更新されること."""
        result = proc_model.clip_velocity_range(v_min=-2500, v_max=2500)
        new_wl = result.sci.wavelength
        assert new_wl is not None
        # clipped 後の wavelength[0] は元の全波長配列の部分集合
        orig_wl = proc_model.sci.wavelength
        assert orig_wl is not None
        assert new_wl[0] >= orig_wl[0]

    def test_naxis1_updated_in_header(self, proc_model: ProcessingImageModel):
        result = proc_model.clip_velocity_range(v_min=-2500, v_max=2500)
        assert result.sci.header.get("NAXIS1") == result.shape[1]

    def test_raises_when_no_pixels_in_range(self, proc_model: ProcessingImageModel):
        with pytest.raises(ValueError, match="ピクセルが存在しません"):
            proc_model.clip_velocity_range(v_min=1e6, v_max=2e6)

    def test_write_fits_output_exists(self, proc_model: ProcessingImageModel, tmp_path: Path):
        """clip 後に write_fits でファイルが書き出されること."""
        result = proc_model.clip_velocity_range(v_min=-2500, v_max=2500)
        out_path = result.write_fits(output_dir=tmp_path, overwrite=True)
        assert out_path.exists()

    def test_write_fits_vclip_keywords_in_file(
        self, proc_model: ProcessingImageModel, tmp_path: Path
    ):
        """書き出した FITS の primary header に VCLIP キーワードが存在すること."""
        result = proc_model.clip_velocity_range(v_min=-2500.0, v_max=2500.0)
        out_path = result.write_fits(output_dir=tmp_path, overwrite=True)
        with fits.open(out_path) as hdul:
            hdr = hdul[0].header
            assert hdr.get("VCLIP") is True
            assert hdr.get("VCLIPMIN") == pytest.approx(-2500.0)
            assert hdr.get("VCLIPMAX") == pytest.approx(2500.0)
            assert hdr.get("VRECES") == pytest.approx(NGC1068_V)


# ---------------------------------------------------------------------------
# plot_continuum_fit
# ---------------------------------------------------------------------------


class TestPlotContinuumFit:
    def test_returns_axes(self, proc_model: ProcessingImageModel):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result = proc_model.subtract_continuum()
        ax = result.plot_continuum_fit(slit_index=0)
        assert ax is not None
        plt.close("all")

    def test_raises_without_subtract_continuum(self, proc_model: ProcessingImageModel):
        with pytest.raises(ValueError, match="continuum"):
            proc_model.plot_continuum_fit(slit_index=0)


# ---------------------------------------------------------------------------
# ProcessingImageCollection
# ---------------------------------------------------------------------------


class TestProcessingImageCollection:
    def test_setup(self, proc_fits_path: Path):
        col = ReaderCollection.from_paths([proc_fits_path])
        ic = ProcessingImageCollection.setup(
            col,
            recession_velocity=NGC1068_V,
            continuum_windows_kms=CONTINUUM_WINDOWS,
        )
        assert len(ic) == 1
        assert isinstance(ic[0], ProcessingImageModel)

    def test_subtract_continuum_collection(self, proc_fits_path: Path):
        col = ReaderCollection.from_paths([proc_fits_path])
        ic = ProcessingImageCollection.setup(
            col,
            recession_velocity=NGC1068_V,
            continuum_windows_kms=CONTINUUM_WINDOWS,
        )
        result = ic.subtract_continuum()
        assert result[0].primary_header.get("CONTSUB") is True

    def test_clip_velocity_range_collection(self, proc_fits_path: Path):
        col = ReaderCollection.from_paths([proc_fits_path])
        ic = ProcessingImageCollection.setup(
            col,
            recession_velocity=NGC1068_V,
            continuum_windows_kms=CONTINUUM_WINDOWS,
        )
        result = ic.clip_velocity_range(v_min=-2500, v_max=2500)
        assert result[0].shape[1] < ic[0].shape[1]

    def test_write_fits_collection(self, proc_fits_path: Path, tmp_path: Path):
        col = ReaderCollection.from_paths([proc_fits_path])
        ic = ProcessingImageCollection.setup(
            col,
            recession_velocity=NGC1068_V,
            continuum_windows_kms=CONTINUUM_WINDOWS,
        )
        clipped = ic.clip_velocity_range(v_min=-2500, v_max=2500)
        paths = clipped.write_fits(output_dir=tmp_path, overwrite=True)
        assert len(paths) == 1
        assert paths[0].exists()
