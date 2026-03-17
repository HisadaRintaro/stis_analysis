"""ReconstructPipeline / ReconstructResult — 3D 再構成パイプライン.

ProcessingPipeline / ProcessingResult パターンを踏襲する:
  1. InstrumentModel でファイル探索
  2. DataCube.from_proc_files() — raw cube 構築
  3. DataCube.interpolate()     — x 方向補間
  4. DataCube.compute_sigma_v() — σ_v マップ計算
  5. VelocityField.with_k(k)    — 変換係数 k を設定
  6. DataCube.reconstruct(vf)   — 3D 再構成
  7. save_picture=True なら確認プロット保存
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stis_analysis.core.instrument import InstrumentModel
from stis_analysis.core.wave_constants import oiii5007_stp
from stis_analysis.reconstruct.cube import DataCube
from stis_analysis.reconstruct.velocity_field import (
    LinearVelocityField,
    PowerLawVelocityField,
    VelocityField,
)


@dataclass(frozen=True)
class ReconstructResult:
    """3D 再構成の結果を保持するデータクラス.

    Attributes
    ----------
    raw_cube : DataCube
        from_proc_files() で構築した raw ステージの DataCube
    interpolated_cube : DataCube
        interpolate() 後の DataCube
    velocity_field : VelocityField
        k 設定済みの VelocityField
    reconstructed_cube : DataCube
        reconstruct() 後の DataCube
    """

    raw_cube: DataCube
    interpolated_cube: DataCube
    velocity_field: VelocityField
    reconstructed_cube: DataCube

    # ------------------------------------------------------------------
    # 確認プロット
    # ------------------------------------------------------------------

    def plot_sigma_v_map(self, ax=None, save_dir: Path | None = None):
        """σ_v マップを 2D imshow で表示する.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            描画先 Axes。None の場合は新規作成。
        save_dir : Path, optional
            保存先ディレクトリ。None の場合は保存しない。
        """
        raise NotImplementedError

    def plot_channel_map(self, v_index: int, ax=None, save_dir: Path | None = None):
        """指定速度インデックスのチャンネルマップを表示する.

        Parameters
        ----------
        v_index : int
            速度軸のインデックス
        ax : matplotlib.axes.Axes, optional
            描画先 Axes。None の場合は新規作成。
        save_dir : Path, optional
            保存先ディレクトリ。None の場合は保存しない。
        """
        raise NotImplementedError

    def plot_reconstructed_slice(
        self, z_index: int, ax=None, save_dir: Path | None = None
    ):
        """指定深度インデックスの reconstructed cube スライスを表示する.

        Parameters
        ----------
        z_index : int
            深度軸のインデックス
        ax : matplotlib.axes.Axes, optional
            描画先 Axes。None の場合は新規作成。
        save_dir : Path, optional
            保存先ディレクトリ。None の場合は保存しない。
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ReconstructPipeline:
    """_proc.fits (×6) → 3D 再構成を一括実行するパイプライン.

    Attributes
    ----------
    slit_positions : list[float]
        各スリットの x 位置 [arcsec]。FITSヘッダーから取得せず外部設定する。
    k : float
        σ_v = k · σ_z の変換係数 [km/s / arcsec]。事前フィット済みの値を設定する。
        デフォルト: np.nan（run() 実行前に必ず設定すること）
    velocity_field_model : str
        速度場モデル。"linear" または "power_law"。デフォルト: "linear"
    alpha : float
        `velocity_field_model="power_law"` のときのべき乗指数 α。デフォルト: 1.0
    recession_velocity : float
        銀河後退速度 [km/s]。デフォルト: 1148.0（NGC1068）
    rest_wavelength : float
        速度 v=0 の基準静止波長 [Å]。デフォルト: oiii5007_stp
    pixel_scale_arcsec : float
        DataCube.interpolate() の x グリッド間隔 [arcsec/pix]。デフォルト: 0.05
    suffix : str
        処理済みファイルの接尾辞。デフォルト: "_proc"
    depth : int
        InstrumentModel のサブディレクトリ探索深度。デフォルト: 0
    exclude_files : tuple[str, ...]
        除外するファイル名のパターン（部分一致）。デフォルト: ()
    """

    slit_positions: list[float]
    k: float = np.nan
    velocity_field_model: str = "linear"
    alpha: float = 1.0
    recession_velocity: float = 1148.0
    rest_wavelength: float = oiii5007_stp
    pixel_scale_arcsec: float = 0.05
    suffix: str = "_proc"
    depth: int = 0
    exclude_files: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # パイプライン実行
    # ------------------------------------------------------------------

    def run(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        save_picture: bool = False,
    ) -> ReconstructResult:
        """パイプラインを実行する.

        Parameters
        ----------
        input_dir : Path | str
            `_proc.fits` が格納されたディレクトリ
        output_dir : Path | str
            確認プロット等の出力先ディレクトリ
        save_picture : bool, optional
            True の場合、output_dir に確認用画像を保存する。デフォルト: False

        Returns
        -------
        ReconstructResult

        Raises
        ------
        ValueError
            `self.k` が np.nan の場合（事前に with_k() または k= で設定が必要）
        NotImplementedError
            未実装
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _build_velocity_field(self, sigma_v: np.ndarray, x_grid: np.ndarray) -> VelocityField:
        """velocity_field_model に応じた VelocityField を構築する."""
        if self.velocity_field_model == "linear":
            return LinearVelocityField(sigma_v=sigma_v, k=self.k, x_grid=x_grid)
        elif self.velocity_field_model == "power_law":
            return PowerLawVelocityField(
                sigma_v=sigma_v, k=self.k, x_grid=x_grid, alpha=self.alpha
            )
        else:
            raise ValueError(
                f"velocity_field_model には 'linear' か 'power_law' を指定してください: "
                f"{self.velocity_field_model!r}"
            )
