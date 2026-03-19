"""VelocityField — 速度場モデル.

速度場の仮定によって constant k の計算式と v→z 変換式が変わるため、
基底クラス + サブクラスで設計する。

k の決定フロー:
    1. DataCube.sigma_v → (v_mean, sigma_v: float)
    2. DataCube.sigma_z → sigma_z: float  （sigma_x, sigma_y から導出）
    3. vf = LinearVelocityField().with_k_from_sigmas(sigma_v, sigma_z)
    4. cube.reconstruct(vf)
    5. 必要に応じて cube.sigma_z で収束確認・反復

σ 値は DataCube が所有する。VelocityField はモデルの「式」だけを持つ。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class VelocityField(ABC):
    """速度場モデルの基底クラス.

    Attributes
    ----------
    k : float
        v→z 変換係数 [km/s / arcsec]。
        `with_k()` または `with_k_from_sigmas()` で設定する。デフォルト: np.nan（未設定）
    """

    k: float = np.nan

    # ------------------------------------------------------------------
    # イミュータブル更新
    # ------------------------------------------------------------------

    def with_k(self, k: float) -> "VelocityField":
        """変換係数 k を設定した新しいインスタンスを返す.

        Parameters
        ----------
        k : float
            v→z 変換係数 [km/s / arcsec]

        Returns
        -------
        VelocityField
            k が設定された同型のサブクラスインスタンス
        """
        return replace(self, k=k)

    def with_k_from_sigmas(self, sigma_v: float, sigma_z: float) -> "VelocityField":
        """σ_v と σ_z から k を計算して設定した新しいインスタンスを返す.

        `compute_k()` をサブクラスの実装に委譲する。

        Parameters
        ----------
        sigma_v : float
            フラックス加重速度分散 [km/s]。`DataCube.sigma_v` の第 2 要素。
        sigma_z : float
            深度方向の空間分散 [arcsec]。`DataCube.sigma_z` から取得。

        Returns
        -------
        VelocityField
            k が設定された同型のサブクラスインスタンス
        """
        return self.with_k(self.compute_k(sigma_v, sigma_z))

    # ------------------------------------------------------------------
    # k 計算（サブクラスで実装）
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_k(self, sigma_v: float, sigma_z: float) -> float:
        """σ_v と σ_z から変換係数 k を計算する.

        速度場モデルの仮定によって計算式が異なるため、サブクラスで実装する。

        Parameters
        ----------
        sigma_v : float
            フラックス加重速度分散 [km/s]
        sigma_z : float
            深度方向の空間分散 [arcsec]

        Returns
        -------
        float
            変換係数 k [km/s / arcsec]
        """
        ...

    # ------------------------------------------------------------------
    # v→z 変換（サブクラスで実装）
    # ------------------------------------------------------------------

    @abstractmethod
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
        """
        ...


@dataclass(frozen=True)
class LinearVelocityField(VelocityField):
    """線形速度場モデル: v = k · z.

    k の計算: k = σ_v / σ_z
    v → z 変換: z = v / k
    """

    def compute_k(self, sigma_v: float, sigma_z: float) -> float:
        """k = σ_v / σ_z を計算する.

        Parameters
        ----------
        sigma_v : float
            フラックス加重速度分散 [km/s]
        sigma_z : float
            深度方向の空間分散 [arcsec]

        Returns
        -------
        float
            変換係数 k [km/s / arcsec]
        """
        return sigma_v / sigma_z

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
                "k が未設定です。`with_k_from_sigmas()` または `with_k()` で"
                "変換係数を設定してから呼び出してください。"
            )
        return v / self.k


@dataclass(frozen=True)
class PowerLawVelocityField(VelocityField):
    """べき乗則速度場モデル: v = k · z^α.

    k の計算: k = σ_v / σ_z^α
    v → z 変換: z = sign(v) · |v / k|^(1/α)

    Attributes
    ----------
    alpha : float
        べき乗則の指数 α。デフォルト: 1.0（LinearVelocityField と同等）
    """

    alpha: float = 1.0

    def compute_k(self, sigma_v: float, sigma_z: float) -> float:
        """k = σ_v / σ_z^α を計算する.

        Parameters
        ----------
        sigma_v : float
            フラックス加重速度分散 [km/s]
        sigma_z : float
            深度方向の空間分散 [arcsec]

        Returns
        -------
        float
            変換係数 k [km/s / arcsec^α]
        """
        return sigma_v / sigma_z**self.alpha

    def velocity_to_depth(self, v: np.ndarray) -> np.ndarray:
        """v = k · z^α から z = sign(v) · |v/k|^(1/α) で深度に変換する.

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
                "k が未設定です。`with_k_from_sigmas()` または `with_k()` で"
                "変換係数を設定してから呼び出してください。"
            )
        return np.sign(v) * np.abs(v / self.k) ** (1.0 / self.alpha)
