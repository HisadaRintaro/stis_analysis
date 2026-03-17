"""VelocityField — 速度場モデル.

フラックス加重速度分散マップ（σ_v）と v→z 変換係数 k を保持する。
サブクラスで速度場モデル（v = k·z, v = k·z^α 等）を実装する。

k の決定フロー:
    1. DataCube.compute_sigma_v() → VelocityField (k=np.nan)
    2. ユーザーが幾何モデルから σ_z を算出:
         k = vf.sigma_v.mean() / sigma_z_expected
    3. vf = vf.with_k(k)
    4. cube.reconstruct(vf)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class VelocityField:
    """速度場モデルの基底クラス.

    Attributes
    ----------
    sigma_v : np.ndarray
        フラックス加重速度分散マップ [km/s]。shape: (n_x, n_y)
    k : float
        σ_v = k · σ_z の変換係数 [km/s / arcsec]。
        外部フィット後に `with_k()` で設定する。デフォルト: np.nan（未設定）
    x_grid : np.ndarray | None
        x 軸グリッド [arcsec]。`DataCube.interpolate()` 後の x_grid と対応する。
    """

    sigma_v: np.ndarray
    k: float = np.nan
    x_grid: np.ndarray | None = None

    # ------------------------------------------------------------------
    # イミュータブル更新
    # ------------------------------------------------------------------

    def with_k(self, k: float) -> "VelocityField":
        """変換係数 k を設定した新しい VelocityField を返す.

        Parameters
        ----------
        k : float
            σ_v = k · σ_z の変換係数 [km/s / arcsec]

        Returns
        -------
        VelocityField
            k が設定された同型のサブクラスインスタンス
        """
        return replace(self, k=k)

    # ------------------------------------------------------------------
    # 変換メソッド（サブクラスで実装）
    # ------------------------------------------------------------------

    def velocity_to_depth(self, v: np.ndarray) -> np.ndarray:
        """速度 v [km/s] を深度 z [arcsec] に変換する.

        Parameters
        ----------
        v : np.ndarray
            速度配列 [km/s]

        Returns
        -------
        np.ndarray
            深度配列 [arcsec]

        Raises
        ------
        ValueError
            k が未設定（np.nan）の場合
        NotImplementedError
            サブクラスで未実装の場合
        """
        if np.isnan(self.k):
            raise ValueError(
                "k が未設定です。`with_k(k)` で変換係数を設定してから呼び出してください。"
            )
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 便利プロパティ
    # ------------------------------------------------------------------

    @property
    def sigma_v_median(self) -> float:
        """σ_v マップの中央値 [km/s]."""
        return float(np.nanmedian(self.sigma_v))

    @property
    def sigma_v_mean(self) -> float:
        """σ_v マップの平均値 [km/s]."""
        return float(np.nanmean(self.sigma_v))


@dataclass(frozen=True)
class LinearVelocityField(VelocityField):
    """線形速度場モデル: v = k · z.

    v → z 変換: z = v / k
    """

    def velocity_to_depth(self, v: np.ndarray) -> np.ndarray:
        """v = k · z から z = v / k で深度に変換する.

        Parameters
        ----------
        v : np.ndarray
            速度配列 [km/s]

        Returns
        -------
        np.ndarray
            深度配列 [arcsec]

        Raises
        ------
        ValueError
            k が未設定（np.nan）の場合
        """
        if np.isnan(self.k):
            raise ValueError(
                "k が未設定です。`with_k(k)` で変換係数を設定してから呼び出してください。"
            )
        return v / self.k


@dataclass(frozen=True)
class PowerLawVelocityField(VelocityField):
    """べき乗則速度場モデル: v = k · z^α.

    v → z 変換: z = (v / k)^(1/α)

    Attributes
    ----------
    alpha : float
        べき乗則の指数 α。デフォルト: 1.0（LinearVelocityField と同等）
    """

    alpha: float = 1.0

    def velocity_to_depth(self, v: np.ndarray) -> np.ndarray:
        """v = k · z^α から z = (v/k)^(1/α) で深度に変換する.

        Parameters
        ----------
        v : np.ndarray
            速度配列 [km/s]

        Returns
        -------
        np.ndarray
            深度配列 [arcsec]

        Raises
        ------
        ValueError
            k が未設定（np.nan）の場合
        """
        if np.isnan(self.k):
            raise ValueError(
                "k が未設定です。`with_k(k)` で変換係数を設定してから呼び出してください。"
            )
        return np.sign(v) * np.abs(v / self.k) ** (1.0 / self.alpha)
