"""lacosmic.ImageModel / ImageCollection のテスト."""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from astropy.io import fits

from stis_analysis.core.fits_reader import STISFitsReader
from stis_analysis.core.image import ImageUnit
from stis_analysis.lacosmic.image import ImageModel, ImageCollection


# ---------------------------------------------------------------------------
# ImageModel
# ---------------------------------------------------------------------------

class TestImageModelFromReader:
    def test_from_reader_creates_model(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        assert isinstance(model, ImageModel)

    def test_sci_is_image_unit(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        assert isinstance(model.sci, ImageUnit)

    def test_err_and_dq_present(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        assert model.err is not None
        assert model.dq is not None

    def test_err_dq_none_when_missing(self, sample_fits_path_no_err_dq: Path):
        reader = STISFitsReader.open(sample_fits_path_no_err_dq)
        model = ImageModel.from_reader(reader)
        assert model.err is None
        assert model.dq is None

    def test_dq_mask_generated_from_flags(self, sample_fits_path: Path):
        """DQ フラグ 16 のピクセルが dq_mask で True になること."""
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader, dq_flags=16)
        assert model.dq_mask is not None
        # conftest で (3, 5) に DQ=16 を埋め込んである
        assert model.dq_mask[3, 5] is np.bool_(True)

    def test_dq_mask_none_when_no_dq(self, sample_fits_path_no_err_dq: Path):
        reader = STISFitsReader.open(sample_fits_path_no_err_dq)
        model = ImageModel.from_reader(reader)
        assert model.dq_mask is None

    def test_source_path_set(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        assert model.source_path == sample_fits_path.parent

    def test_dq_flags_stored(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader, dq_flags=256)
        assert model.dq_flags == 256


class TestImageModelProperties:
    def test_shape(self, sample_fits_path: Path, fits_shape):
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        assert model.shape == fits_shape

    def test_filename_from_rootname(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        assert model.filename == "test_root"

    def test_filename_unknown_when_no_rootname(self, tmp_path: Path, fits_shape):
        rows, cols = fits_shape
        h = fits.Header()  # ROOTNAME なし
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.ones((rows, cols), dtype=np.float32), header=h),
        ])
        p = tmp_path / "no_root.fits"
        hdul.writeto(p)
        reader = STISFitsReader.open(p)
        model = ImageModel.from_reader(reader)
        assert model.filename == "UNKNOWN"

    def test_repr_contains_shape(self, sample_fits_path: Path, fits_shape):
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        assert str(fits_shape) in repr(model)


class TestImageModelInterpolateBadPixels:
    def test_returns_new_instance(self, sample_fits_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        result = model.interpolate_bad_pixels()
        assert result is not model

    def test_hot_pixel_interpolated(self, sample_fits_path: Path):
        """DQ=16 の hot pixel が補間されること（元の異常値でなくなること）."""
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader, dq_flags=16)
        # hot pixel に異常値を埋め込む
        from dataclasses import replace
        hot_data = model.sci.data.copy()
        hot_data[3, 5] = 99999.0
        model = replace(model, sci=replace(model.sci, data=hot_data))

        result = model.interpolate_bad_pixels()
        # 補間後は元の異常値でなくなっている
        assert result.sci.data[3, 5] != 99999.0

    def test_negative_pixels_masked(self, sample_fits_path: Path):
        """負値ピクセルが補間されること."""
        from dataclasses import replace
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        neg_data = model.sci.data.copy()
        neg_data[1, 1] = -100.0
        model = replace(model, sci=replace(model.sci, data=neg_data))

        result = model.interpolate_bad_pixels(mask_negative=True)
        assert result.sci.data[1, 1] >= 0.0

    def test_no_negative_masking_when_disabled(self, sample_fits_path: Path):
        """mask_negative=False のとき負値が保持されること."""
        from dataclasses import replace
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        neg_data = model.sci.data.copy()
        neg_data[1, 1] = -100.0
        # DQ マスクが効かない位置 → dq_mask[1,1]=False
        model = replace(model, sci=replace(model.sci, data=neg_data))

        result = model.interpolate_bad_pixels(mask_negative=False)
        assert result.sci.data[1, 1] == pytest.approx(-100.0)


class TestImageModelRemoveCosmicRay:
    def test_returns_new_instance(self, sample_fits_path: Path):
        """remove_cosmic_ray が新しい ImageModel を返すこと（モック使用）."""
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)

        fake_clean = model.sci.data.copy()
        fake_mask = np.zeros(model.shape, dtype=bool)

        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            result = model.remove_cosmic_ray()

        assert result is not model
        mock_rc.assert_called_once()

    def test_cr_mask_stored(self, sample_fits_path: Path):
        """宇宙線マスクが cr_mask に格納されること."""
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)

        fake_clean = model.sci.data.copy()
        fake_mask = np.zeros(model.shape, dtype=bool)
        fake_mask[0, 0] = True

        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            result = model.remove_cosmic_ray()

        assert result.cr_mask is not None
        assert result.cr_mask.header.get("EXTNAME") == "LACOSMIC"
        assert result.cr_mask.data[0, 0] is np.bool_(True)


class TestImageModelWriteFits:
    def test_write_fits_creates_file(self, sample_fits_path: Path, tmp_path: Path):
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)

        fake_clean = model.sci.data.copy()
        fake_mask = np.zeros(model.shape, dtype=bool)
        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            model_cr = model.remove_cosmic_ray()

        out_path = model_cr.write_fits(output_dir=tmp_path, overwrite=True)
        assert out_path.exists()
        assert out_path.suffix == ".fits"

    def test_write_fits_has_lacorr_keyword(self, sample_fits_path: Path, tmp_path: Path):
        """LA-Cosmic 適用済みの場合 LACORR=True がヘッダーに追加されること."""
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)

        fake_clean = model.sci.data.copy()
        fake_mask = np.zeros(model.shape, dtype=bool)
        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            model_cr = model.remove_cosmic_ray()

        out_path = model_cr.write_fits(output_dir=tmp_path, overwrite=True)
        with fits.open(out_path) as hdul:
            assert hdul[0].header.get("LACORR") is True

    def test_write_fits_raises_when_no_output_dir(self, sample_fits_path: Path):
        """source_path も output_dir もない場合 ValueError."""
        from dataclasses import replace
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        model_no_path = replace(model, source_path=None)
        with pytest.raises(ValueError, match="output_dir"):
            model_no_path.write_fits()

    def test_write_fits_raises_if_exists_no_overwrite(
        self, sample_fits_path: Path, tmp_path: Path
    ):
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        fake_clean = model.sci.data.copy()
        fake_mask = np.zeros(model.shape, dtype=bool)
        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            model_cr = model.remove_cosmic_ray()

        model_cr.write_fits(output_dir=tmp_path, overwrite=True)
        with pytest.raises(FileExistsError):
            model_cr.write_fits(output_dir=tmp_path, overwrite=False)

    def test_write_fits_warning_without_lacosmic(self, sample_fits_path: Path, tmp_path: Path):
        """LA-Cosmic 未適用で _lac suffix を使うと UserWarning."""
        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader)
        with pytest.warns(UserWarning, match="LA-Cosmic が未適用"):
            model.write_fits(output_suffix="_lac", output_dir=tmp_path, overwrite=True)


class TestImageModelImshowMaskBugfix:
    def test_imshow_mask_no_attribute_error(self, sample_fits_path: Path):
        """dq_mask (np.ndarray) に .data を呼ぶバグが修正されていること."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reader = STISFitsReader.open(sample_fits_path)
        model = ImageModel.from_reader(reader, dq_flags=16)
        assert model.dq_mask is not None  # dq_mask が存在する状態

        # 修正前は AttributeError: 'numpy.ndarray' object has no attribute 'data'
        ax = model.imshow_mask()
        assert ax is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# ImageCollection
# ---------------------------------------------------------------------------

class TestImageCollection:
    def test_from_readers(self, sample_fits_path: Path):
        from stis_analysis.core.fits_reader import ReaderCollection
        col = ReaderCollection.from_paths([sample_fits_path])
        ic = ImageCollection.from_readers(col)
        assert len(ic) == 1
        assert isinstance(ic[0], ImageModel)

    def test_iter(self, sample_fits_path: Path):
        from stis_analysis.core.fits_reader import ReaderCollection
        col = ReaderCollection.from_paths([sample_fits_path])
        ic = ImageCollection.from_readers(col)
        images = list(ic)
        assert len(images) == 1

    def test_lacosmic_params_stored(self, sample_fits_path: Path):
        from stis_analysis.core.fits_reader import ReaderCollection
        col = ReaderCollection.from_paths([sample_fits_path])
        ic = ImageCollection.from_readers(
            col, contrast=3.0, cr_threshold=4.0,
            neighbor_threshold=4.5,
        )
        assert ic.contrast == 3.0
        assert ic.cr_threshold == 4.0
        assert ic.neighbor_threshold == 4.5

    def test_remove_cosmic_ray_calls_each_model(self, sample_fits_path: Path):
        from stis_analysis.core.fits_reader import ReaderCollection
        col = ReaderCollection.from_paths([sample_fits_path])
        ic = ImageCollection.from_readers(col)

        fake_clean = ic[0].sci.data.copy()
        fake_mask = np.zeros(ic[0].shape, dtype=bool)

        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            result = ic.remove_cosmic_ray()

        assert mock_rc.call_count == 1
        assert result[0].cr_mask is not None

    def test_imshow_mask_dq_fallback_when_none(self, sample_fits_path_no_err_dq: Path):
        """dq_mask=None でも imshow_mask(mask_type='dq') がエラーにならないこと."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from stis_analysis.core.fits_reader import ReaderCollection

        col = ReaderCollection.from_paths([sample_fits_path_no_err_dq])
        ic = ImageCollection.from_readers(col)
        assert ic[0].dq_mask is None  # dq が存在しない

        # 修正前は None.imshow() で TypeError
        ax = ic.imshow_mask(mask_type="dq")
        assert ax is not None
        plt.close("all")

    def test_write_fits_collection(self, sample_fits_path: Path, tmp_path: Path):
        from stis_analysis.core.fits_reader import ReaderCollection
        col = ReaderCollection.from_paths([sample_fits_path])
        ic = ImageCollection.from_readers(col)

        fake_clean = ic[0].sci.data.copy()
        fake_mask = np.zeros(ic[0].shape, dtype=bool)

        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            ic_cr = ic.remove_cosmic_ray()

        paths = ic_cr.write_fits(output_dir=tmp_path, overwrite=True)
        assert len(paths) == 1
        assert paths[0].exists()
