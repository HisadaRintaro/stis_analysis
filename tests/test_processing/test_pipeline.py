"""ProcessingPipeline / ProcessingResult のテスト."""

from pathlib import Path

import numpy as np
import pytest
import warnings
from astropy.io import fits

from stis_analysis.processing.pipeline import ProcessingPipeline, ProcessingResult
from stis_analysis.processing.image import ProcessingImageCollection
from stis_analysis.processing.wave_constants import c_kms, oiii5007_stp


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

NGC1068_V = 1148.0
CONTINUUM_WINDOWS = [(-4000.0, -3200.0), (3000.0, 4000.0)]


def _write_proc_input(path: Path, rootname: str = "dummy") -> None:
    """WCS 付き最小 _lac.fits を書き出す（ProcessingPipeline.run(run_x2d=False) 用）."""
    cols, rows = 200, 8
    cdelt1 = 0.554
    z = NGC1068_V / c_kms
    center_lam = oiii5007_stp * (1.0 + z)
    crval1 = center_lam - cdelt1 * (cols // 2)

    wavelength = crval1 + cdelt1 * np.arange(cols)
    sci_data = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        continuum = 500.0 + row * 0.5 + 0.3 * (np.arange(cols) - cols / 2.0)
        lam5007_obs = oiii5007_stp * (1.0 + z)
        sigma = 1.5 * cdelt1
        gauss5007 = 200.0 * np.exp(-0.5 * ((wavelength - lam5007_obs) / sigma) ** 2)
        sci_data[row] = (continuum + gauss5007).astype(np.float32)

    h0 = fits.Header()
    h0["ROOTNAME"] = rootname
    h1 = fits.Header()
    h1["ROOTNAME"] = rootname
    h1["NAXIS"] = 2
    h1["NAXIS1"] = cols
    h1["NAXIS2"] = rows
    h1["CRVAL1"] = crval1
    h1["CDELT1"] = cdelt1
    h1["CRPIX1"] = 1.0

    hdul = fits.HDUList([
        fits.PrimaryHDU(header=h0),
        fits.ImageHDU(data=sci_data, header=h1),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(path)


@pytest.fixture
def lac_input_dir(tmp_path: Path) -> Path:
    """_lac.fits が 1 つある入力ディレクトリ（depth=0）."""
    d = tmp_path / "input"
    d.mkdir()
    _write_proc_input(d / "obj1_lac.fits", rootname="obj1")
    return d


@pytest.fixture
def pipeline() -> ProcessingPipeline:
    return ProcessingPipeline(
        continuum_windows_kms=CONTINUUM_WINDOWS,
        recession_velocity=NGC1068_V,
        depth=0,
        suffix="_lac",
    )


# ---------------------------------------------------------------------------
# ProcessingResult
# ---------------------------------------------------------------------------


class TestProcessingResult:
    def test_has_before_after(self, lac_input_dir: Path, tmp_path: Path, pipeline: ProcessingPipeline):
        output_dir = tmp_path / "output"
        result = pipeline.run(
            input_dir=lac_input_dir,
            output_dir=output_dir,
            run_x2d=False,
        )
        assert isinstance(result, ProcessingResult)
        assert isinstance(result.before, ProcessingImageCollection)
        assert isinstance(result.after, ProcessingImageCollection)
        assert len(result.before.images) == 1
        assert len(result.after.images) == 1

    def test_output_paths_exist(self, lac_input_dir: Path, tmp_path: Path, pipeline: ProcessingPipeline):
        output_dir = tmp_path / "output"
        result = pipeline.run(
            input_dir=lac_input_dir,
            output_dir=output_dir,
            run_x2d=False,
        )
        assert len(result.output_paths) == 1
        assert all(p.exists() for p in result.output_paths)


# ---------------------------------------------------------------------------
# _resolve_output_dir
# ---------------------------------------------------------------------------


class TestResolveOutputDir:
    def test_returns_base_when_empty(self, tmp_path: Path):
        base = tmp_path / "out"
        result = ProcessingPipeline._resolve_output_dir(base, "_proc")
        assert result == base

    def test_returns_base_when_not_exists(self, tmp_path: Path):
        base = tmp_path / "nonexistent"
        result = ProcessingPipeline._resolve_output_dir(base, "_proc")
        assert result == base

    def test_redirects_when_proc_exists(self, tmp_path: Path):
        base = tmp_path / "out"
        base.mkdir()
        (base / "obj1_proc.fits").touch()

        result = ProcessingPipeline._resolve_output_dir(base, "_proc")
        assert result == base.parent / "out-2"

    def test_increments_number_when_both_exist(self, tmp_path: Path):
        base = tmp_path / "out"
        base.mkdir()
        (base / "obj1_proc.fits").touch()
        candidate2 = base.parent / "out-2"
        candidate2.mkdir()
        (candidate2 / "obj1_proc.fits").touch()

        result = ProcessingPipeline._resolve_output_dir(base, "_proc")
        assert result == base.parent / "out-3"

    def test_warns_when_redirecting(self, tmp_path: Path):
        base = tmp_path / "out"
        base.mkdir()
        (base / "obj1_proc.fits").touch()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProcessingPipeline._resolve_output_dir(base, "_proc")

        assert len(w) == 1
        assert "_proc" in str(w[0].message)


# ---------------------------------------------------------------------------
# run() の退避動作
# ---------------------------------------------------------------------------


class TestProcessingPipelineRedirect:
    def test_run_redirects_when_output_exists(
        self, lac_input_dir: Path, tmp_path: Path, pipeline: ProcessingPipeline
    ):
        """output_dir に既存 _proc.fits がある場合、番号付きディレクトリに退避すること."""
        output_dir = tmp_path / "output"

        # 1 回目
        result1 = pipeline.run(
            input_dir=lac_input_dir,
            output_dir=output_dir,
            run_x2d=False,
        )
        assert all(p.parent == output_dir for p in result1.output_paths)

        # 2 回目：同じ output_dir → output-2 に退避
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result2 = pipeline.run(
                input_dir=lac_input_dir,
                output_dir=output_dir,
                run_x2d=False,
            )

        redirected = output_dir.parent / "output-2"
        assert all(p.parent == redirected for p in result2.output_paths)
        assert redirected.exists()
        redirect_warnings = [x for x in w if "_proc" in str(x.message)]
        assert len(redirect_warnings) == 1
