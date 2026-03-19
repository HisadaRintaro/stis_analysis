"""DataCube のテスト."""

import numpy as np
import pytest

from stis_analysis.reconstruct.cube import DataCube
from stis_analysis.reconstruct.velocity_field import LinearVelocityField, PowerLawVelocityField


# ------------------------------------------------------------------
# フィクスチャ
# ------------------------------------------------------------------

N_SLIT, N_Y, N_V = 4, 10, 30
PIXEL_SCALE = 0.1  # arcsec/pix（テスト用に粗めに設定）
RECESSION_VELOCITY = 1148.0
REST_WAVELENGTH = 5006.843


@pytest.fixture
def velocity_array() -> np.ndarray:
    return np.linspace(-300.0, 300.0, N_V)


@pytest.fixture
def y_array() -> np.ndarray:
    return np.linspace(-0.2, 0.2, N_Y)


@pytest.fixture
def raw_cube(velocity_array: np.ndarray, y_array: np.ndarray) -> DataCube:
    """raw ステージの DataCube（スリット位置は離散的）."""
    rng = np.random.default_rng(0)
    data = rng.uniform(0.0, 100.0, size=(N_SLIT, N_Y, N_V)).astype(np.float64)
    x_positions = np.array([-0.15, -0.05, 0.05, 0.15])
    return DataCube(
        data=data,
        velocity_array=velocity_array,
        recession_velocity=RECESSION_VELOCITY,
        rest_wavelength=REST_WAVELENGTH,
        x_positions=x_positions,
        y_array=y_array,
    )


@pytest.fixture
def interp_cube(raw_cube: DataCube) -> DataCube:
    """interpolated ステージの DataCube."""
    return raw_cube.interpolate(pixel_scale_arcsec=PIXEL_SCALE)


@pytest.fixture
def recon_cube(interp_cube: DataCube) -> DataCube:
    """reconstructed ステージの DataCube（LinearVelocityField 使用）."""
    vf = LinearVelocityField().with_k_from_sigmas(sigma_v=100.0, sigma_z=0.3)
    return interp_cube.reconstruct(vf)


# ------------------------------------------------------------------
# ステージ判定
# ------------------------------------------------------------------

class TestStageFlags:
    def test_raw_cube_is_raw(self, raw_cube: DataCube):
        assert raw_cube.is_raw is True
        assert raw_cube.is_interpolated is False
        assert raw_cube.is_reconstructed is False

    def test_interp_cube_is_interpolated(self, interp_cube: DataCube):
        assert interp_cube.is_raw is False
        assert interp_cube.is_interpolated is True
        assert interp_cube.is_reconstructed is False

    def test_recon_cube_is_reconstructed(self, recon_cube: DataCube):
        assert recon_cube.is_raw is False
        assert recon_cube.is_interpolated is False
        assert recon_cube.is_reconstructed is True


# ------------------------------------------------------------------
# interpolate()
# ------------------------------------------------------------------

class TestInterpolate:
    def test_output_shape_y_v_preserved(self, raw_cube: DataCube, interp_cube: DataCube):
        """y 軸・v 軸の次元数は変わらない."""
        assert interp_cube.data.shape[1] == N_Y
        assert interp_cube.data.shape[2] == N_V

    def test_x_grid_is_set(self, interp_cube: DataCube):
        assert interp_cube.x_grid is not None

    def test_x_positions_is_cleared(self, interp_cube: DataCube):
        assert interp_cube.x_positions is None

    def test_x_grid_range(self, raw_cube: DataCube, interp_cube: DataCube):
        """x_grid の範囲が x_positions と一致する."""
        assert interp_cube.x_grid is not None
        assert raw_cube.x_positions is not None
        assert interp_cube.x_grid[0] == pytest.approx(raw_cube.x_positions.min())
        assert interp_cube.x_grid[-1] == pytest.approx(raw_cube.x_positions.max())

    def test_raises_when_not_raw(self, interp_cube: DataCube):
        with pytest.raises(ValueError, match="raw ステージ"):
            interp_cube.interpolate()


# ------------------------------------------------------------------
# _flux_weighted_stats()
# ------------------------------------------------------------------

class TestFluxWeightedStats:
    def test_uniform_flux_mean_equals_array_mean(self):
        """一様フラックスのとき加重平均 = 単純平均."""
        flux = np.ones((3, 4, 5))
        array = np.linspace(-2.0, 2.0, 5)
        mean, _ = DataCube._flux_weighted_stats(flux, array)
        assert mean == pytest.approx(array.mean())

    def test_zero_flux_returns_nan(self):
        flux = np.zeros((3, 4, 5))
        array = np.ones(5)
        mean, sigma = DataCube._flux_weighted_stats(flux, array)
        assert np.isnan(mean)
        assert np.isnan(sigma)

    def test_negative_flux_clipped(self):
        """負のフラックスは 0 にクリップされる."""
        flux = np.array([-1.0, 1.0, 1.0])
        array = np.array([0.0, 1.0, 2.0])
        mean, _ = DataCube._flux_weighted_stats(flux, array)
        # 負フラックスを除いた重み: [0, 1, 1] → 平均 = (1+2)/2 = 1.5
        assert mean == pytest.approx(1.5)

    def test_sigma_zero_for_single_value(self):
        """全ての array 値が同じなら sigma = 0."""
        flux = np.ones(5)
        array = np.full(5, 3.0)
        _, sigma = DataCube._flux_weighted_stats(flux, array)
        assert sigma == pytest.approx(0.0)


# ------------------------------------------------------------------
# sigma_v プロパティ
# ------------------------------------------------------------------

class TestSigmaV:
    def test_returns_float_tuple(self, interp_cube: DataCube):
        v_mean, sigma_v = interp_cube.sigma_v
        assert isinstance(v_mean, float)
        assert isinstance(sigma_v, float)

    def test_sigma_v_positive(self, interp_cube: DataCube):
        _, sigma_v = interp_cube.sigma_v
        assert sigma_v >= 0.0

    def test_raises_when_not_interpolated(self, raw_cube: DataCube):
        with pytest.raises(ValueError, match="interpolated ステージ"):
            _ = raw_cube.sigma_v

    def test_raises_when_reconstructed(self, recon_cube: DataCube):
        with pytest.raises(ValueError, match="interpolated ステージ"):
            _ = recon_cube.sigma_v


# ------------------------------------------------------------------
# sigma_x / sigma_y / sigma_z プロパティ
# ------------------------------------------------------------------

class TestSigmaXYZ:
    def test_sigma_x_returns_float_tuple(self, recon_cube: DataCube):
        x_mean, sigma_x = recon_cube.sigma_x
        assert isinstance(x_mean, float)
        assert isinstance(sigma_x, float)

    def test_sigma_y_returns_float_tuple(self, recon_cube: DataCube):
        y_mean, sigma_y = recon_cube.sigma_y
        assert isinstance(y_mean, float)
        assert isinstance(sigma_y, float)

    def test_sigma_z_returns_float(self, recon_cube: DataCube):
        assert isinstance(recon_cube.sigma_z, float)

    def test_sigma_z_formula(self, recon_cube: DataCube):
        """sigma_z = sqrt(0.5 * (sigma_x^2 + sigma_y^2))."""
        _, sx = recon_cube.sigma_x
        _, sy = recon_cube.sigma_y
        expected = np.sqrt(0.5 * (sx**2 + sy**2))
        assert recon_cube.sigma_z == pytest.approx(expected)

    def test_sigma_x_raises_when_not_reconstructed(self, interp_cube: DataCube):
        with pytest.raises(ValueError, match="reconstructed ステージ"):
            _ = interp_cube.sigma_x

    def test_sigma_y_raises_when_not_reconstructed(self, interp_cube: DataCube):
        with pytest.raises(ValueError, match="reconstructed ステージ"):
            _ = interp_cube.sigma_y

    def test_sigma_y_raises_when_y_array_none(self, interp_cube: DataCube):
        """y_array が未設定の場合 reconstruct 後も ValueError."""
        vf = LinearVelocityField().with_k_from_sigmas(sigma_v=100.0, sigma_z=0.3)
        cube_no_y = DataCube(
            data=interp_cube.data,
            velocity_array=interp_cube.velocity_array,
            recession_velocity=RECESSION_VELOCITY,
            x_grid=interp_cube.x_grid,
            y_array=None,
        )
        recon = cube_no_y.reconstruct(vf)
        with pytest.raises(ValueError, match="y_array が未設定"):
            _ = recon.sigma_y


# ------------------------------------------------------------------
# reconstruct()
# ------------------------------------------------------------------

class TestReconstruct:
    def test_output_shape_preserved(self, interp_cube: DataCube, recon_cube: DataCube):
        """n_x, n_y は変化しない。n_z == n_v。"""
        assert recon_cube.data.shape[0] == interp_cube.data.shape[0]
        assert recon_cube.data.shape[1] == interp_cube.data.shape[1]
        assert recon_cube.data.shape[2] == interp_cube.data.shape[2]

    def test_z_array_is_set(self, recon_cube: DataCube):
        assert recon_cube.z_array is not None
        assert len(recon_cube.z_array) == N_V

    def test_z_array_is_monotonic(self, recon_cube: DataCube):
        assert recon_cube.z_array is not None
        diffs = np.diff(recon_cube.z_array)
        assert np.all(diffs > 0)

    def test_z_array_range_linear_model(self, interp_cube: DataCube):
        """線形モデルでは z = v/k なので z_array の範囲が velocity_array/k と一致する."""
        k = 200.0
        vf = LinearVelocityField().with_k(k)
        recon = interp_cube.reconstruct(vf)
        assert recon.z_array is not None
        assert recon.z_array[0] == pytest.approx(interp_cube.velocity_array.min() / k)
        assert recon.z_array[-1] == pytest.approx(interp_cube.velocity_array.max() / k)

    def test_raises_when_not_interpolated(self, raw_cube: DataCube):
        vf = LinearVelocityField().with_k(200.0)
        with pytest.raises(ValueError, match="interpolated ステージ"):
            raw_cube.reconstruct(vf)

    def test_raises_when_k_not_set(self, interp_cube: DataCube):
        with pytest.raises(ValueError, match="k が未設定"):
            interp_cube.reconstruct(LinearVelocityField())

    def test_power_law_model(self, interp_cube: DataCube):
        """PowerLawVelocityField でも正常に動作する."""
        vf = PowerLawVelocityField(alpha=2.0).with_k_from_sigmas(
            sigma_v=100.0, sigma_z=0.3
        )
        recon = interp_cube.reconstruct(vf)
        assert recon.is_reconstructed
        assert recon.z_array is not None
