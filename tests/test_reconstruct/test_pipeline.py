"""ReconstructPipeline / ReconstructResult のテスト."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stis_analysis.reconstruct.cube import DataCube
from stis_analysis.reconstruct.pipeline import ReconstructPipeline, ReconstructResult
from stis_analysis.reconstruct.velocity_field import (
    LinearVelocityField,
    PowerLawVelocityField,
)

# ------------------------------------------------------------------
# フィクスチャ
# ------------------------------------------------------------------

N_SLIT, N_Y, N_V = 4, 10, 20
PIXEL_SCALE = 0.1
RECESSION_VELOCITY = 1148.0
REST_WAVELENGTH = 5006.843

SLIT_POSITIONS = [-0.15, -0.05, 0.05, 0.15]
FAKE_PATHS = [Path(f"slit_{i}.fits") for i in range(N_SLIT)]


@pytest.fixture
def raw_cube() -> DataCube:
    rng = np.random.default_rng(42)
    data = rng.uniform(0.0, 100.0, size=(N_SLIT, N_Y, N_V)).astype(np.float64)
    return DataCube(
        data=data,
        velocity_array=np.linspace(-200.0, 200.0, N_V),
        recession_velocity=RECESSION_VELOCITY,
        rest_wavelength=REST_WAVELENGTH,
        x_positions=np.array(SLIT_POSITIONS),
        y_array=np.linspace(-0.2, 0.2, N_Y),
    )


@pytest.fixture
def pipeline() -> ReconstructPipeline:
    return ReconstructPipeline(
        slit_positions=SLIT_POSITIONS,
        velocity_field_model="linear",
        recession_velocity=RECESSION_VELOCITY,
        pixel_scale_arcsec=PIXEL_SCALE,
    )


def _mock_instrument(n_files: int = N_SLIT):
    """InstrumentModel のモックを返す."""
    mock = MagicMock()
    mock.path_list = FAKE_PATHS[:n_files]
    return mock


# ------------------------------------------------------------------
# _build_velocity_field()
# ------------------------------------------------------------------

class TestBuildVelocityField:
    def test_linear_model_returns_linear_vf(self, pipeline: ReconstructPipeline):
        vf = pipeline._build_velocity_field(sigma_v=100.0, sigma_z=0.5)
        assert isinstance(vf, LinearVelocityField)

    def test_linear_model_k_value(self, pipeline: ReconstructPipeline):
        vf = pipeline._build_velocity_field(sigma_v=100.0, sigma_z=0.5)
        assert vf.k == pytest.approx(200.0)  # 100 / 0.5 = 200

    def test_power_law_model_returns_power_law_vf(self):
        pl_pipeline = ReconstructPipeline(
            slit_positions=SLIT_POSITIONS,
            velocity_field_model="power_law",
            alpha=2.0,
        )
        vf = pl_pipeline._build_velocity_field(sigma_v=100.0, sigma_z=0.5)
        assert isinstance(vf, PowerLawVelocityField)

    def test_power_law_model_k_value(self):
        pl_pipeline = ReconstructPipeline(
            slit_positions=SLIT_POSITIONS,
            velocity_field_model="power_law",
            alpha=2.0,
        )
        vf = pl_pipeline._build_velocity_field(sigma_v=100.0, sigma_z=0.5)
        assert vf.k == pytest.approx(100.0 / 0.5**2)  # = 400.0

    def test_invalid_model_raises_value_error(self):
        invalid = ReconstructPipeline(
            slit_positions=SLIT_POSITIONS,
            velocity_field_model="unknown_model",
        )
        with pytest.raises(ValueError, match="velocity_field_model"):
            invalid._build_velocity_field(sigma_v=100.0, sigma_z=0.5)


# ------------------------------------------------------------------
# run() — 正常系
# ------------------------------------------------------------------

class TestPipelineRun:
    def test_run_returns_reconstruct_result(
        self, pipeline: ReconstructPipeline, raw_cube: DataCube, tmp_path: Path
    ):
        with patch(
            "stis_analysis.reconstruct.pipeline.InstrumentModel.load",
            return_value=_mock_instrument(),
        ), patch(
            "stis_analysis.reconstruct.pipeline.DataCube.from_proc_files",
            return_value=raw_cube,
        ):
            result = pipeline.run(input_dir=tmp_path, output_dir=tmp_path)
        assert isinstance(result, ReconstructResult)

    def test_run_raw_cube_is_raw(
        self, pipeline: ReconstructPipeline, raw_cube: DataCube, tmp_path: Path
    ):
        with patch(
            "stis_analysis.reconstruct.pipeline.InstrumentModel.load",
            return_value=_mock_instrument(),
        ), patch(
            "stis_analysis.reconstruct.pipeline.DataCube.from_proc_files",
            return_value=raw_cube,
        ):
            result = pipeline.run(input_dir=tmp_path, output_dir=tmp_path)
        assert result.raw_cube.is_raw

    def test_run_interpolated_cube_is_interpolated(
        self, pipeline: ReconstructPipeline, raw_cube: DataCube, tmp_path: Path
    ):
        with patch(
            "stis_analysis.reconstruct.pipeline.InstrumentModel.load",
            return_value=_mock_instrument(),
        ), patch(
            "stis_analysis.reconstruct.pipeline.DataCube.from_proc_files",
            return_value=raw_cube,
        ):
            result = pipeline.run(input_dir=tmp_path, output_dir=tmp_path)
        assert result.interpolated_cube.is_interpolated

    def test_run_reconstructed_cube_is_reconstructed(
        self, pipeline: ReconstructPipeline, raw_cube: DataCube, tmp_path: Path
    ):
        with patch(
            "stis_analysis.reconstruct.pipeline.InstrumentModel.load",
            return_value=_mock_instrument(),
        ), patch(
            "stis_analysis.reconstruct.pipeline.DataCube.from_proc_files",
            return_value=raw_cube,
        ):
            result = pipeline.run(input_dir=tmp_path, output_dir=tmp_path)
        assert result.reconstructed_cube.is_reconstructed

    def test_run_velocity_field_k_is_set(
        self, pipeline: ReconstructPipeline, raw_cube: DataCube, tmp_path: Path
    ):
        with patch(
            "stis_analysis.reconstruct.pipeline.InstrumentModel.load",
            return_value=_mock_instrument(),
        ), patch(
            "stis_analysis.reconstruct.pipeline.DataCube.from_proc_files",
            return_value=raw_cube,
        ):
            result = pipeline.run(input_dir=tmp_path, output_dir=tmp_path)
        assert not np.isnan(result.velocity_field.k)

    def test_run_creates_output_dir(
        self, pipeline: ReconstructPipeline, raw_cube: DataCube, tmp_path: Path
    ):
        output_dir = tmp_path / "new_dir"
        with patch(
            "stis_analysis.reconstruct.pipeline.InstrumentModel.load",
            return_value=_mock_instrument(),
        ), patch(
            "stis_analysis.reconstruct.pipeline.DataCube.from_proc_files",
            return_value=raw_cube,
        ):
            pipeline.run(input_dir=tmp_path, output_dir=output_dir)
        assert output_dir.exists()


# ------------------------------------------------------------------
# run() — 異常系
# ------------------------------------------------------------------

class TestPipelineRunErrors:
    def test_raises_when_file_count_mismatch(
        self, pipeline: ReconstructPipeline, tmp_path: Path
    ):
        """slit_positions の長さとファイル数が一致しない場合 ValueError."""
        with patch(
            "stis_analysis.reconstruct.pipeline.InstrumentModel.load",
            return_value=_mock_instrument(n_files=2),  # 2 files vs 4 slit_positions
        ):
            with pytest.raises(ValueError, match="slit_positions"):
                pipeline.run(input_dir=tmp_path, output_dir=tmp_path)


# ------------------------------------------------------------------
# ReconstructResult
# ------------------------------------------------------------------

class TestReconstructResult:
    @pytest.fixture
    def result(self, raw_cube: DataCube) -> ReconstructResult:
        interp = raw_cube.interpolate(pixel_scale_arcsec=PIXEL_SCALE)
        _, sigma_v = interp.sigma_v
        sigma_z = interp.sigma_z
        vf = LinearVelocityField().with_k_from_sigmas(sigma_v=sigma_v, sigma_z=sigma_z)
        recon = interp.reconstruct(vf)
        return ReconstructResult(
            raw_cube=raw_cube,
            interpolated_cube=interp,
            velocity_field=vf,
            reconstructed_cube=recon,
        )

    def test_result_holds_all_stages(self, result: ReconstructResult):
        assert result.raw_cube.is_raw
        assert result.interpolated_cube.is_interpolated
        assert result.reconstructed_cube.is_reconstructed

    def test_velocity_field_k_matches_sigmas(self, result: ReconstructResult):
        """k = sigma_v / sigma_z の関係が成立する（線形モデル）."""
        _, sigma_v = result.interpolated_cube.sigma_v
        sigma_z = result.interpolated_cube.sigma_z
        assert result.velocity_field.k == pytest.approx(sigma_v / sigma_z)

    def test_plot_channel_map_not_implemented(self, result: ReconstructResult):
        with pytest.raises(NotImplementedError):
            result.plot_channel_map(v_index=0)

    def test_plot_reconstructed_slice_not_implemented(self, result: ReconstructResult):
        with pytest.raises(NotImplementedError):
            result.plot_reconstructed_slice(z_index=0)
