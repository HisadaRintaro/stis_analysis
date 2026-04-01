"""DataCube のテスト."""

from unittest.mock import MagicMock, patch

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
    def test_output_shape_y_v_preserved(self, interp_cube: DataCube):
        """y 軸・v 軸の次元数は変わらない."""
        assert interp_cube.data.shape[1] == N_Y
        assert interp_cube.data.shape[2] == N_V

    def test_x_array_is_set(self, interp_cube: DataCube):
        assert interp_cube.x_array is not None

    def test_x_positions_is_cleared(self, interp_cube: DataCube):
        assert interp_cube.x_positions is None

    def test_x_array_range(self, raw_cube: DataCube, interp_cube: DataCube):
        """x_array の範囲が x_positions と一致する."""
        assert interp_cube.x_array is not None
        assert raw_cube.x_positions is not None
        assert interp_cube.x_array[0] == pytest.approx(raw_cube.x_positions.min())
        assert interp_cube.x_array[-1] == pytest.approx(raw_cube.x_positions.max())

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
            raw_cube.sigma_v  # noqa: B018

    def test_raises_when_reconstructed(self, recon_cube: DataCube):
        with pytest.raises(ValueError, match="interpolated ステージ"):
            recon_cube.sigma_v  # noqa: B018


# ------------------------------------------------------------------
# sigma_x / sigma_y / sigma_z プロパティ
# ------------------------------------------------------------------

class TestSigmaXYZ:
    def test_sigma_x_works_on_interpolated(self, interp_cube: DataCube):
        """sigma_x は interpolated ステージでも使用可能."""
        x_mean, sigma_x = interp_cube.sigma_x
        assert isinstance(x_mean, float)
        assert isinstance(sigma_x, float)

    def test_sigma_x_works_on_reconstructed(self, recon_cube: DataCube):
        x_mean, sigma_x = recon_cube.sigma_x
        assert isinstance(x_mean, float)
        assert isinstance(sigma_x, float)

    def test_sigma_y_works_on_interpolated(self, interp_cube: DataCube):
        """sigma_y は interpolated ステージでも使用可能."""
        y_mean, sigma_y = interp_cube.sigma_y
        assert isinstance(y_mean, float)
        assert isinstance(sigma_y, float)

    def test_sigma_y_works_on_reconstructed(self, recon_cube: DataCube):
        y_mean, sigma_y = recon_cube.sigma_y
        assert isinstance(y_mean, float)
        assert isinstance(sigma_y, float)

    def test_sigma_z_works_on_interpolated(self, interp_cube: DataCube):
        """sigma_z は interpolated ステージでも使用可能（reconstruct 前に k を計算できる）."""
        assert isinstance(interp_cube.sigma_z, float)

    def test_sigma_z_formula(self, interp_cube: DataCube):
        """sigma_z = sqrt(0.5 * (sigma_x^2 + sigma_y^2))."""
        _, sx = interp_cube.sigma_x
        _, sy = interp_cube.sigma_y
        expected = np.sqrt(0.5 * (sx**2 + sy**2))
        assert interp_cube.sigma_z == pytest.approx(expected)

    def test_sigma_x_raises_when_raw(self, raw_cube: DataCube):
        """raw ステージ（x_array=None）では使用不可."""
        with pytest.raises(ValueError, match="x_array が設定"):
            raw_cube.sigma_x  # noqa: B018

    def test_sigma_y_raises_when_y_array_none(self, interp_cube: DataCube):
        """y_array が未設定の場合は ValueError."""
        cube_no_y = DataCube(
            data=interp_cube.data,
            velocity_array=interp_cube.velocity_array,
            recession_velocity=RECESSION_VELOCITY,
            x_array=interp_cube.x_array,
            y_array=None,
        )
        with pytest.raises(ValueError, match="y_array が未設定"):
            cube_no_y.sigma_y  # noqa: B018


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


# ------------------------------------------------------------------
# trim_y()
# ------------------------------------------------------------------

class TestTrimY:
    def test_trim_y_data_shape(self, interp_cube: DataCube):
        """trim_y 後に data の axis 1 が縮小される."""
        assert interp_cube.y_array is not None
        y_mid = float(interp_cube.y_array[len(interp_cube.y_array) // 2])
        trimmed = interp_cube.trim_y(y_min=interp_cube.y_array[0], y_max=y_mid)
        assert trimmed.data.shape[1] < interp_cube.data.shape[1]
        assert trimmed.data.shape[0] == interp_cube.data.shape[0]
        assert trimmed.data.shape[2] == interp_cube.data.shape[2]

    def test_trim_y_array_updated(self, interp_cube: DataCube):
        """trim_y 後の y_array が指定範囲に収まっている."""
        assert interp_cube.y_array is not None
        y_min, y_max = float(interp_cube.y_array[2]), float(interp_cube.y_array[7])
        trimmed = interp_cube.trim_y(y_min=y_min, y_max=y_max)
        assert trimmed.y_array is not None
        assert float(trimmed.y_array[0]) >= y_min
        assert float(trimmed.y_array[-1]) <= y_max
        assert len(trimmed.y_array) == trimmed.data.shape[1]

    def test_trim_y_other_axes_unchanged(self, interp_cube: DataCube):
        """x_array / velocity_array は変化しない."""
        assert interp_cube.y_array is not None
        trimmed = interp_cube.trim_y(y_max=interp_cube.y_array[-2])
        assert np.array_equal(trimmed.x_array, interp_cube.x_array)
        assert np.array_equal(trimmed.velocity_array, interp_cube.velocity_array)

    def test_trim_y_stage_preserved(self, interp_cube: DataCube):
        """ステージ（is_interpolated）が保たれる."""
        assert interp_cube.y_array is not None
        trimmed = interp_cube.trim_y(y_max=interp_cube.y_array[-2])
        assert trimmed.is_interpolated

    def test_trim_y_on_reconstructed(self, recon_cube: DataCube):
        """reconstructed cube でも動作する."""
        assert recon_cube.y_array is not None
        trimmed = recon_cube.trim_y(y_min=recon_cube.y_array[1], y_max=recon_cube.y_array[-2])
        assert trimmed.is_reconstructed
        assert trimmed.data.shape[1] < recon_cube.data.shape[1]

    def test_trim_y_raises_when_y_array_none(self, interp_cube: DataCube):
        """y_array が None のとき ValueError."""
        from dataclasses import replace as dc_replace
        cube_no_y = dc_replace(interp_cube, y_array=None)
        with pytest.raises(ValueError, match="y_array が設定"):
            cube_no_y.trim_y(y_min=0.0, y_max=1.0)

    def test_trim_y_raises_when_no_valid_range(self, interp_cube: DataCube):
        """範囲外の指定で ValueError."""
        assert interp_cube.y_array is not None
        with pytest.raises(ValueError, match="有効な y ピクセルがありません"):
            interp_cube.trim_y(y_min=999.0, y_max=1000.0)


# ------------------------------------------------------------------
# view_3d()
# ------------------------------------------------------------------

class TestView3d:
    def test_raises_when_not_reconstructed(self, interp_cube: DataCube):
        """interpolated ステージでは ValueError."""
        with pytest.raises(ValueError, match="reconstructed ステージ"):
            interp_cube.view_3d()

    def test_save_dir(self, recon_cube: DataCube, tmp_path):
        """save_dir 指定時にスクリーンショットが保存される."""
        mock_viewer = MagicMock()
        with patch("napari.Viewer", return_value=mock_viewer), \
             patch("napari.run"):
            recon_cube.view_3d(save_dir=tmp_path)

        mock_viewer.add_image.assert_called_once_with(
            recon_cube.data, name="flux", colormap="inferno"
        )
        mock_viewer.screenshot.assert_called_once_with(
            path=str(tmp_path / "view_3d.png"), canvas_only=True
        )
        mock_viewer.close.assert_called_once()

    def test_save_dir_with_contrast_limits(self, recon_cube: DataCube, tmp_path):
        """contrast_limits が add_image に渡される."""
        mock_viewer = MagicMock()
        with patch("napari.Viewer", return_value=mock_viewer), \
             patch("napari.run"):
            recon_cube.view_3d(contrast_limits=(0.0, 1.0), save_dir=tmp_path)

        mock_viewer.add_image.assert_called_once_with(
            recon_cube.data, name="flux", colormap="inferno", contrast_limits=(0.0, 1.0)
        )
