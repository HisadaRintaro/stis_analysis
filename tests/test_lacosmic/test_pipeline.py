"""LaCosmicPipeline / PipelineResult のテスト."""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from astropy.io import fits

from stis_analysis.lacosmic.pipeline import LaCosmicPipeline, PipelineResult
from stis_analysis.lacosmic.image import ImageCollection


def _write_crj(path: Path, rootname: str = "dummy") -> None:
    """最小構成の _crj.fits を書き出す."""
    h0 = fits.Header()
    h0["ROOTNAME"] = rootname
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=h0),
        fits.ImageHDU(data=np.ones((10, 20), dtype=np.float32)),
        fits.ImageHDU(data=np.ones((10, 20), dtype=np.float32) * 5),
        fits.ImageHDU(data=np.zeros((10, 20), dtype=np.int16)),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(path)


@pytest.fixture
def crj_input_dir(tmp_path: Path) -> Path:
    """depth=1 で _crj.fits が 2 つある入力ディレクトリ."""
    subdir = tmp_path / "input" / "obs"
    subdir.mkdir(parents=True)
    _write_crj(subdir / "obj1_crj.fits", rootname="obj1")
    _write_crj(subdir / "obj2_crj.fits", rootname="obj2")
    return tmp_path / "input"


class TestLaCosmicPipelineDefaults:
    def test_default_params(self):
        pipeline = LaCosmicPipeline()
        assert pipeline.contrast == 5.0
        assert pipeline.cr_threshold == 5.0
        assert pipeline.neighbor_threshold == 5.0
        assert pipeline.error == 5.0
        assert pipeline.dq_flags == 16
        assert pipeline.suffix == "_crj"
        assert pipeline.extension == ".fits"
        assert pipeline.depth == 1
        assert pipeline.exclude_files == ()

    def test_custom_params(self):
        pipeline = LaCosmicPipeline(contrast=3.0, cr_threshold=4.0, dq_flags=256)
        assert pipeline.contrast == 3.0
        assert pipeline.cr_threshold == 4.0
        assert pipeline.dq_flags == 256


class TestPipelineResult:
    def test_attributes(self, crj_input_dir: Path, tmp_path: Path):
        """PipelineResult が before / after / output_paths を持つこと（モック使用）."""
        output_dir = tmp_path / "output"
        pipeline = LaCosmicPipeline()

        fake_mask = np.zeros((10, 20), dtype=bool)
        fake_clean = np.ones((10, 20), dtype=np.float32)

        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            result = pipeline.run(
                input_dir=crj_input_dir,
                output_dir=output_dir,
                overwrite=True,
            )

        assert isinstance(result, PipelineResult)
        assert isinstance(result.before, ImageCollection)
        assert isinstance(result.after, ImageCollection)
        assert isinstance(result.output_paths, list)
        assert all(isinstance(p, Path) for p in result.output_paths)


class TestLaCosmicPipelineRun:
    def test_run_finds_files(self, crj_input_dir: Path, tmp_path: Path):
        """入力ディレクトリから 2 ファイルを見つけること."""
        output_dir = tmp_path / "output"
        pipeline = LaCosmicPipeline()

        fake_mask = np.zeros((10, 20), dtype=bool)
        fake_clean = np.ones((10, 20), dtype=np.float32)

        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            result = pipeline.run(
                input_dir=crj_input_dir,
                output_dir=output_dir,
                overwrite=True,
            )

        assert len(result.before) == 2
        assert len(result.after) == 2

    def test_run_writes_output_files(self, crj_input_dir: Path, tmp_path: Path):
        """出力 FITS ファイルが書き出されること."""
        output_dir = tmp_path / "output"
        pipeline = LaCosmicPipeline()

        fake_mask = np.zeros((10, 20), dtype=bool)
        fake_clean = np.ones((10, 20), dtype=np.float32)

        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            result = pipeline.run(
                input_dir=crj_input_dir,
                output_dir=output_dir,
                overwrite=True,
            )

        assert len(result.output_paths) == 2
        for p in result.output_paths:
            assert p.exists()

    def test_run_output_dir_created(self, crj_input_dir: Path, tmp_path: Path):
        """output_dir が存在しなくても自動生成されること."""
        output_dir = tmp_path / "new" / "deep" / "dir"
        assert not output_dir.exists()

        pipeline = LaCosmicPipeline()
        fake_mask = np.zeros((10, 20), dtype=bool)
        fake_clean = np.ones((10, 20), dtype=np.float32)

        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            pipeline.run(input_dir=crj_input_dir, output_dir=output_dir, overwrite=True)

        assert output_dir.exists()

    def test_run_exclude_files(self, crj_input_dir: Path, tmp_path: Path):
        """exclude_files で指定したファイルが除外されること."""
        output_dir = tmp_path / "output"
        pipeline = LaCosmicPipeline(exclude_files=("obj1_crj",))

        fake_mask = np.zeros((10, 20), dtype=bool)
        fake_clean = np.ones((10, 20), dtype=np.float32)

        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            result = pipeline.run(
                input_dir=crj_input_dir,
                output_dir=output_dir,
                overwrite=True,
            )

        assert len(result.before) == 1

    def test_before_after_differ_after_cr_removal(self, crj_input_dir: Path, tmp_path: Path):
        """before と after で sci データが異なること（宇宙線除去が適用されている）."""
        output_dir = tmp_path / "output"
        pipeline = LaCosmicPipeline()

        # before の sci データに宇宙線を埋め込む
        # after では clean_data（宇宙線除去済み）が返される
        fake_mask = np.zeros((10, 20), dtype=bool)
        fake_mask[0, 0] = True
        fake_clean = np.ones((10, 20), dtype=np.float32) * 500.0

        with patch("stis_analysis.lacosmic.image.remove_cosmics") as mock_rc:
            mock_rc.return_value = (fake_clean, fake_mask)
            result = pipeline.run(
                input_dir=crj_input_dir,
                output_dir=output_dir,
                overwrite=True,
            )

        # after の cr_mask に宇宙線マスクが格納されている
        assert result.after[0].cr_mask is not None
        assert result.after[0].cr_mask.data[0, 0] is np.bool_(True)
