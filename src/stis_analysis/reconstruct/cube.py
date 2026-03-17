"""DataCube — 3D スペクトルキューブモデル.

raw / interpolated / reconstructed の全ステージを単一クラスで統一する。
ステージはオプションフィールドの有無で判定する:
  - raw          : x_positions が設定され x_grid が None
  - interpolated : x_grid が設定され z_array が None
  - reconstructed: z_array が設定済み

使用方法:
    # 1. _proc.fits を読み込み raw cube を構築
    cube = DataCube.from_proc_files(
        paths=proc_paths,
        slit_positions=slit_positions,
        recession_velocity=1148.0,
        rest_wavelength=oiii5007_stp,
    )

    # 2. x 方向を等間隔グリッドに補間
    interp_cube = cube.interpolate(pixel_scale_arcsec=0.05)

    # 3. フラックス加重速度分散 σ_v マップを計算
    velocity_field = interp_cube.compute_sigma_v()

    # 4. k を設定して 3D 再構成
    vf = velocity_field.with_k(k)
    recon_cube = interp_cube.reconstruct(vf)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from stis_analysis.core.fits_reader import ReaderCollection
from stis_analysis.core.wave_constants import oiii5007_stp
from stis_analysis.lacosmic.image import ImageCollection


@dataclass(frozen=True)
class DataCube:
    """3D スペクトルキューブ.

    全ステージ（raw / interpolated / reconstructed）を統一したデータクラス。

    Attributes
    ----------
    data : np.ndarray
        スペクトルキューブ。shape は以下のステージによって異なる:
        - raw          : (n_slit, n_y, n_v)
        - interpolated : (n_x, n_y, n_v)
        - reconstructed: (n_x, n_y, n_z)
    velocity_array : np.ndarray
        速度軸 [km/s]。shape: (n_v,)
    recession_velocity : float
        銀河後退速度 [km/s]
    rest_wavelength : float
        速度 v=0 の基準静止波長 [Å]。デフォルト: oiii5007_stp
    x_positions : np.ndarray | None
        raw ステージ: 各スリットの x 位置 [arcsec]。shape: (n_slit,)
    x_grid : np.ndarray | None
        interpolated ステージ: 等間隔 x 軸 [arcsec]。shape: (n_x,)
    z_array : np.ndarray | None
        reconstructed ステージ: 深度軸 [arcsec]。shape: (n_z,)
    source_paths : tuple[Path, ...] | None
        読み込み元 FITS ファイルパスのリスト
    """

    data: np.ndarray
    velocity_array: np.ndarray
    recession_velocity: float
    rest_wavelength: float = oiii5007_stp
    x_positions: np.ndarray | None = None
    x_grid: np.ndarray | None = None
    z_array: np.ndarray | None = None
    source_paths: tuple[Path, ...] | None = None

    # ------------------------------------------------------------------
    # ステージ判定プロパティ
    # ------------------------------------------------------------------

    @property
    def is_raw(self) -> bool:
        """raw ステージ（スリット位置は離散的）かどうか."""
        return self.x_positions is not None and self.x_grid is None

    @property
    def is_interpolated(self) -> bool:
        """interpolated ステージ（x 軸が等間隔グリッド）かどうか."""
        return self.x_grid is not None and self.z_array is None

    @property
    def is_reconstructed(self) -> bool:
        """reconstructed ステージ（z 軸が設定済み）かどうか."""
        return self.z_array is not None

    # ------------------------------------------------------------------
    # コンストラクタ
    # ------------------------------------------------------------------

    @classmethod
    def from_proc_files(
        cls,
        paths: list[Path],
        slit_positions: list[float],
        recession_velocity: float,
        rest_wavelength: float = oiii5007_stp,
    ) -> "DataCube":
        """_proc.fits ファイルリストから raw DataCube を構築する.

        Parameters
        ----------
        paths : list[Path]
            `_proc.fits` ファイルパスのリスト（スリット数分）
        slit_positions : list[float]
            各スリットの x 位置 [arcsec]。`paths` と同じ順番で指定する
        recession_velocity : float
            銀河の後退速度 [km/s]（NGC1068: 1148）
        rest_wavelength : float, optional
            速度 v=0 の基準静止波長 [Å]。デフォルト: oiii5007_stp

        Returns
        -------
        DataCube
            raw ステージの DataCube（is_raw == True）

        Raises
        ------
        ValueError
            paths と slit_positions の長さが一致しない場合、
            paths が空の場合、またはスリット間でデータ形状が一致しない場合
        """
        if len(paths) == 0:
            raise ValueError("paths が空です。少なくとも 1 ファイルを指定してください。")
        if len(paths) != len(slit_positions):
            raise ValueError(
                f"paths の長さ ({len(paths)}) と "
                f"slit_positions の長さ ({len(slit_positions)}) が一致しません。"
            )

        readers = ReaderCollection.from_paths(paths)
        collection = ImageCollection.from_readers(readers)

        # 全スリットで共通の velocity_array を最初のモデルから計算
        velocity_array = collection[0].sci.velocity_array(recession_velocity, rest_wavelength)

        # 各スリットの SCI データを収集
        slit_data_list = [img.sci.data for img in collection]

        # スリット間でデータ形状が一致するか確認
        shapes = [d.shape for d in slit_data_list]
        if len(set(shapes)) != 1:
            raise ValueError(
                f"スリット間でデータ形状が一致しません: "
                + ", ".join(f"{p.name}: {s}" for p, s in zip(paths, shapes))
            )

        # shape: (n_slit, n_y, n_v)
        data = np.stack(slit_data_list, axis=0)

        return cls(
            data=data,
            velocity_array=velocity_array,
            recession_velocity=recession_velocity,
            rest_wavelength=rest_wavelength,
            x_positions=np.array(slit_positions),
            x_grid=None,
            z_array=None,
            source_paths=tuple(paths),
        )

    # ------------------------------------------------------------------
    # 処理メソッド
    # ------------------------------------------------------------------

    def interpolate(
        self,
        pixel_scale_arcsec: float = 0.05,
        kind: str = "linear",
    ) -> "DataCube":
        """x 方向を等間隔グリッドに補間する.

        `x_positions` の範囲を `pixel_scale_arcsec` 間隔で等間隔化し、
        `scipy.interpolate.interp1d(axis=0)` で x 軸方向を一括補間する。
        y 軸・v 軸はすでに等間隔グリッドのため補間不要。

        Parameters
        ----------
        pixel_scale_arcsec : float, optional
            出力 x グリッドの間隔 [arcsec/pix]。デフォルト: 0.05
        kind : str, optional
            補間方法。'linear', 'cubic', 'quadratic' 等。デフォルト: 'linear'

        Returns
        -------
        DataCube
            interpolated ステージの DataCube（is_interpolated == True）

        Raises
        ------
        ValueError
            raw ステージでない場合
        """
        from scipy.interpolate import interp1d

        if not self.is_raw:
            raise ValueError(
                "interpolate() は raw ステージの DataCube でのみ使用できます。"
            )

        x_positions = self.x_positions  # shape: (n_slit,)
        assert x_positions is not None  # guaranteed by is_raw check above

        # 等間隔 x グリッドを生成
        x_min, x_max = float(x_positions.min()), float(x_positions.max())
        n_x = round((x_max - x_min) / pixel_scale_arcsec) + 1
        x_grid = np.linspace(x_min, x_max, n_x)  # shape: (n_x,)

        # axis=0 (x 軸) に沿って一括補間
        f = interp1d(x_positions, self.data, axis=0, kind=kind)
        interpolated = f(x_grid)  # shape: (n_x, n_y, n_v)

        return replace(
            self,
            data=interpolated,
            x_grid=x_grid,
            x_positions=None,
        )

    def compute_sigma_v(self) -> "VelocityField":  # type: ignore[name-defined]  # noqa: F821
        """フラックス加重速度分散 σ_v マップを計算する.

        各 (x, y) ピクセルについてフラックス加重平均速度と速度分散を計算し、
        `LinearVelocityField` として返す。

        Returns
        -------
        VelocityField
            σ_v マップを持つ VelocityField（k は未設定 = np.nan）

        Raises
        ------
        ValueError
            `is_interpolated` でない場合
        """
        raise NotImplementedError

    def reconstruct(self, velocity_field: "VelocityField") -> "DataCube":  # type: ignore[name-defined]  # noqa: F821
        """速度場モデルを用いて velocity 軸を depth 軸に変換する.

        `velocity_field.velocity_to_depth(v)` で各ピクセルの v → z 変換を行い、
        共通 z グリッドに補間した reconstructed cube を返す。

        Parameters
        ----------
        velocity_field : VelocityField
            k が設定済みの VelocityField（`with_k()` 適用後）

        Returns
        -------
        DataCube
            reconstructed ステージの DataCube（is_reconstructed == True）

        Raises
        ------
        ValueError
            `is_interpolated` でない場合、または `velocity_field.k` が nan の場合
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 可視化メソッド
    # ------------------------------------------------------------------

    def imshow_channel(self, v_index: int, ax=None, save_dir: Path | None = None):
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

    def plot_spectrum(self, ix: int, iy: int, ax=None):
        """指定 (ix, iy) ピクセルのスペクトルをプロットする.

        Parameters
        ----------
        ix : int
            x 軸インデックス
        iy : int
            y 軸（空間）インデックス
        ax : matplotlib.axes.Axes, optional
            描画先 Axes。None の場合は新規作成。
        """
        raise NotImplementedError

    def imshow_integrated(self, ax=None, save_dir: Path | None = None):
        """velocity 軸を積分した 2D マップを表示する.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            描画先 Axes。None の場合は新規作成。
        save_dir : Path, optional
            保存先ディレクトリ。None の場合は保存しない。
        """
        raise NotImplementedError


# 循環インポートを避けるため VelocityField のアノテーション解決は実行時に行う
from stis_analysis.reconstruct.velocity_field import VelocityField  # noqa: E402
