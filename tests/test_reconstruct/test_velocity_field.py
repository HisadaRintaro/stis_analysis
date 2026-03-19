"""VelocityField クラス群のテスト."""

import numpy as np
import pytest

from stis_analysis.reconstruct.velocity_field import (
    LinearVelocityField,
    PowerLawVelocityField,
    VelocityField,
)


class TestVelocityFieldBase:
    def test_cannot_instantiate_abstract_class(self):
        """VelocityField は抽象クラスのため直接インスタンス化できない."""
        with pytest.raises(TypeError):
            VelocityField()  # type: ignore[abstract]


class TestLinearVelocityFieldComputeK:
    def test_compute_k_basic(self):
        k = LinearVelocityField().compute_k(sigma_v=100.0, sigma_z=0.5)
        assert k == pytest.approx(200.0)

    def test_compute_k_unit(self):
        """k = sigma_v / sigma_z の単純比率."""
        k = LinearVelocityField().compute_k(sigma_v=50.0, sigma_z=0.25)
        assert k == pytest.approx(200.0)


class TestLinearVelocityFieldWithK:
    def test_with_k_sets_k(self):
        vf = LinearVelocityField().with_k(150.0)
        assert vf.k == pytest.approx(150.0)

    def test_with_k_returns_same_type(self):
        vf = LinearVelocityField().with_k(150.0)
        assert isinstance(vf, LinearVelocityField)

    def test_with_k_from_sigmas(self):
        vf = LinearVelocityField().with_k_from_sigmas(sigma_v=100.0, sigma_z=0.5)
        assert vf.k == pytest.approx(200.0)
        assert isinstance(vf, LinearVelocityField)

    def test_with_k_immutable(self):
        """frozen dataclass なので元のインスタンスは変化しない."""
        original = LinearVelocityField()
        _ = original.with_k(100.0)
        assert np.isnan(original.k)


class TestLinearVelocityFieldVelocityToDepth:
    @pytest.fixture
    def vf(self) -> LinearVelocityField:
        return LinearVelocityField().with_k_from_sigmas(sigma_v=100.0, sigma_z=0.5)

    def test_positive_velocity(self, vf: LinearVelocityField):
        z = vf.velocity_to_depth(np.array([200.0]))
        assert z == pytest.approx([1.0])  # 200 / 200 = 1.0

    def test_negative_velocity(self, vf: LinearVelocityField):
        z = vf.velocity_to_depth(np.array([-200.0]))
        assert z == pytest.approx([-1.0])

    def test_zero_velocity(self, vf: LinearVelocityField):
        z = vf.velocity_to_depth(np.array([0.0]))
        assert z == pytest.approx([0.0])

    def test_array_input(self, vf: LinearVelocityField):
        v = np.array([-200.0, 0.0, 200.0])
        z = vf.velocity_to_depth(v)
        assert z == pytest.approx([-1.0, 0.0, 1.0])

    def test_raises_when_k_not_set(self):
        with pytest.raises(ValueError, match="k が未設定"):
            LinearVelocityField().velocity_to_depth(np.array([100.0]))


class TestPowerLawVelocityFieldComputeK:
    def test_alpha_1_equals_linear(self):
        """alpha=1 のとき LinearVelocityField と同じ k になる."""
        k_pl = PowerLawVelocityField(alpha=1.0).compute_k(sigma_v=100.0, sigma_z=0.5)
        k_lin = LinearVelocityField().compute_k(sigma_v=100.0, sigma_z=0.5)
        assert k_pl == pytest.approx(k_lin)

    def test_alpha_2(self):
        """k = sigma_v / sigma_z^alpha."""
        k = PowerLawVelocityField(alpha=2.0).compute_k(sigma_v=100.0, sigma_z=0.5)
        assert k == pytest.approx(100.0 / 0.5**2)  # 400.0


class TestPowerLawVelocityFieldVelocityToDepth:
    @pytest.fixture
    def vf(self) -> PowerLawVelocityField:
        return PowerLawVelocityField(alpha=2.0).with_k_from_sigmas(
            sigma_v=100.0, sigma_z=0.5
        )

    def test_positive_velocity(self, vf: PowerLawVelocityField):
        """z = (v/k)^(1/alpha)."""
        k = vf.k  # = 400.0
        v = np.array([100.0])
        z = vf.velocity_to_depth(v)
        assert z == pytest.approx([(100.0 / k) ** 0.5])

    def test_negative_velocity_symmetric(self, vf: PowerLawVelocityField):
        """負の速度は sign を保持する."""
        v_pos = np.array([200.0])
        v_neg = np.array([-200.0])
        assert vf.velocity_to_depth(v_neg) == pytest.approx(-vf.velocity_to_depth(v_pos))

    def test_raises_when_k_not_set(self):
        with pytest.raises(ValueError, match="k が未設定"):
            PowerLawVelocityField().velocity_to_depth(np.array([100.0]))
