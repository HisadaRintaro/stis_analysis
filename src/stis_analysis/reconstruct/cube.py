"""DataCube — 3D スペクトルキューブモデル.

raw / interpolated / reconstructed の全ステージを単一クラスで統一する。
ステージはオプションフィールドの有無で判定する:
  - raw          : x_positions が設定され x_array が None
  - interpolated : x_array が設定され z_array が None
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

    # 3. フラックス加重速度分散 σ_v と深度分散 σ_z を取得
    _, sigma_v = interp_cube.sigma_v
    sigma_z = 0.3  # ユーザーが幾何から推定 [arcsec]

    # 4. k をモデルから計算して 3D 再構成
    vf = LinearVelocityField().with_k_from_sigmas(sigma_v, sigma_z)
    recon_cube = interp_cube.reconstruct(vf)

    # 5. 再構成後の σ_z で収束確認
    sigma_z_actual = recon_cube.sigma_z
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
    x_array : np.ndarray | None
        interpolated ステージ: 等間隔 x 軸 [arcsec]。shape: (n_x,)
    y_array : np.ndarray | None
        空間 y 軸 [arcsec]。FITS ヘッダーの CRVAL2/CDELT2/CRPIX2 から計算。shape: (n_y,)
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
    x_array: np.ndarray | None = None
    y_array: np.ndarray | None = None
    z_array: np.ndarray | None = None
    source_paths: tuple[Path, ...] | None = None

    # ------------------------------------------------------------------
    # ステージ判定プロパティ
    # ------------------------------------------------------------------

    @property
    def is_raw(self) -> bool:
        """raw ステージ（x_positions あり・x_array なし）かどうか."""
        return self.x_positions is not None and self.x_array is None

    @property
    def is_interpolated(self) -> bool:
        """interpolated ステージ（x_array あり・z_array なし）かどうか."""
        return self.x_array is not None and self.z_array is None

    @property
    def is_reconstructed(self) -> bool:
        """reconstructed ステージ（z 軸が設定済み）かどうか."""
        return self.z_array is not None

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.is_raw:
            stage = "raw"
            shape_label = "(n_slit, n_y, n_v)"
        elif self.is_interpolated:
            stage = "interpolated"
            shape_label = "(n_x, n_y, n_v)"
        else:
            stage = "reconstructed"
            shape_label = "(n_x, n_y, n_z)"

        def _arr_summary(arr: np.ndarray, unit: str) -> str:
            if len(arr) <= 6:
                vals = ", ".join(f"{v:.3f}" for v in arr)
                return f"[{vals}] {unit}  ({len(arr)} pts)"
            return f"[{arr[0]:.3f}, ..., {arr[-1]:.3f}] {unit}  ({len(arr)} pts)"

        lines = [
            "DataCube(",
            f"  stage    : {stage}",
            f"  shape    : {self.data.shape}  {shape_label}",
        ]
        if self.is_raw and self.x_positions is not None:
            lines.append(f"  x_pos    : {_arr_summary(self.x_positions, 'arcsec')}")
        elif self.x_array is not None:
            lines.append(f"  x_array  : {_arr_summary(self.x_array, 'arcsec')}")
        if self.y_array is not None:
            lines.append(f"  y_array  : {_arr_summary(self.y_array, 'arcsec')}")
        lines.append(f"  velocity : {_arr_summary(self.velocity_array, 'km/s')}")
        if self.z_array is not None:
            lines.append(f"  z_array  : {_arr_summary(self.z_array, 'arcsec')}")
        lines += [
            f"  v_rec    : {self.recession_velocity:.1f} km/s",
            f"  λ_rest   : {self.rest_wavelength:.3f} Å",
            ")",
        ]
        return "\n".join(lines)

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

        # 全スリットで共通の velocity_array / y_array を最初のモデルから計算
        first_sci = collection[0].sci
        velocity_array = first_sci.velocity_array(recession_velocity, rest_wavelength)

        y_array = first_sci.spatial_array

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
            x_array=None,
            y_array=y_array,
            z_array=None,
            source_paths=tuple(paths),
        )

    # ------------------------------------------------------------------
    # 統計ヘルパー
    # ------------------------------------------------------------------

    @staticmethod
    def _flux_weighted_stats(
        flux: np.ndarray,
        array: np.ndarray,
    ) -> tuple[float, float]:
        """フラックス加重平均と加重標準偏差を計算する（全ピクセル総和）.

        速度軸（v）だけでなく x, y, z 軸にも再利用可能な汎用ヘルパー。
        `array` は `flux` にブロードキャスト可能な shape で渡す。

        Parameters
        ----------
        flux : np.ndarray
            フラックス配列。shape は任意。
        array : np.ndarray
            集計対象の軸の値（速度・位置など）。`flux` にブロードキャスト可能な shape。

        Returns
        -------
        tuple[float, float]
            (weighted_mean, weighted_sigma): 全ピクセルを対象としたスカラー値 [同単位]
            総フラックスが 0 の場合は (np.nan, np.nan) を返す
        """
        w = np.where(flux > 0, flux, 0.0)  # 負フラックスをクリップ
        total = float(w.sum())

        if total == 0.0:
            return float(np.nan), float(np.nan)

        weighted_mean = float(np.sum(w * array) / total)
        weighted_sigma = float(
            np.sqrt(np.sum(w * (array - weighted_mean) ** 2) / total)
        )
        return weighted_mean, weighted_sigma

    @property
    def sigma_v(self) -> tuple[float, float]:
        """フラックス加重平均速度と速度分散 σ_v.

        interpolated cube の全ピクセルにわたるフラックス加重統計をスカラーで返す。

        Returns
        -------
        tuple[float, float]
            (v_mean, sigma_v): 全ピクセルを対象としたスカラー [km/s]
            フラックスが 0 の場合は (np.nan, np.nan) を返す

        Raises
        ------
        ValueError
            `is_interpolated` でない場合
        """
        if not self.is_interpolated:
            raise ValueError(
                "sigma_v は interpolated ステージの DataCube でのみ使用できます。"
            )
        # velocity_array shape (n_v,) → (n_x, n_y, n_v) に自動ブロードキャスト
        return self._flux_weighted_stats(self.data, self.velocity_array)

    @property
    def sigma_x(self) -> tuple[float, float]:
        """フラックス加重平均 x 位置と x 方向空間分散 σ_x.

        interpolated / reconstructed ステージで使用可能。
        視線方向に積分された輝線の空間的広がりを表す。

        Returns
        -------
        tuple[float, float]
            (x_mean, sigma_x): 全ピクセルを対象としたスカラー [arcsec]

        Raises
        ------
        ValueError
            `x_array` が未設定（raw ステージ）の場合
        """
        if self.x_array is None:
            raise ValueError(
                "sigma_x は x_array が設定された DataCube でのみ使用できます（interpolated 以降）。"
            )
        # x_grid shape (n_x,) → (n_x, 1, 1) に reshape してブロードキャスト
        x = self.x_array[:, np.newaxis, np.newaxis]
        return self._flux_weighted_stats(self.data, x)

    @property
    def sigma_y(self) -> tuple[float, float]:
        """フラックス加重平均 y 位置と y 方向空間分散 σ_y.

        interpolated / reconstructed ステージで使用可能。
        視線方向に積分された輝線の空間的広がりを表す。

        Returns
        -------
        tuple[float, float]
            (y_mean, sigma_y): 全ピクセルを対象としたスカラー [arcsec]

        Raises
        ------
        ValueError
            `y_array` が未設定の場合
        """
        if self.y_array is None:
            raise ValueError(
                "y_array が未設定です。FITS ヘッダーから y_array を設定してください。"
            )
        # y_array shape (n_y,) → (1, n_y, 1) に reshape してブロードキャスト
        y = self.y_array[np.newaxis, :, np.newaxis]
        return self._flux_weighted_stats(self.data, y)

    @property
    def sigma_z(self) -> float:
        """深度方向の空間分散 σ_z の推定値.

        球対称を仮定し、σ_z = sqrt(0.5 * (σ_x² + σ_y²)) で導出する。
        interpolated ステージで使用することで reconstruct 前に k を計算できる。

        Returns
        -------
        float
            深度分散 σ_z [arcsec]

        Raises
        ------
        ValueError
            `x_grid` または `y_array` が未設定の場合
        """
        _, sx = self.sigma_x
        _, sy = self.sigma_y
        return float(np.sqrt(0.5 * (sx**2 + sy**2)))

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
        x_array = np.linspace(x_min, x_max, n_x)  # shape: (n_x,)

        # axis=0 (x 軸) に沿って一括補間
        f = interp1d(x_positions, self.data, axis=0, kind=kind)
        interpolated = f(x_array)  # shape: (n_x, n_y, n_v)

        return replace(
            self,
            data=interpolated,
            x_array=x_array,
            x_positions=None,
        )

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
        from scipy.interpolate import interp1d

        if not self.is_interpolated:
            raise ValueError(
                "reconstruct() は interpolated ステージの DataCube でのみ使用できます。"
            )
        if np.isnan(velocity_field.k):
            raise ValueError(
                "velocity_field.k が未設定です。"
                "`with_k_from_sigmas()` または `with_k()` で設定してから呼び出してください。"
            )

        # 1. velocity → z 変換（power law では非等間隔になりうる）
        z_values = velocity_field.velocity_to_depth(self.velocity_array)  # (n_v,)

        # 2. 単調増加に並べ替え
        sort_idx = np.argsort(z_values)
        z_sorted = z_values[sort_idx]
        data_sorted = self.data[:, :, sort_idx]  # (n_x, n_y, n_v)

        # 3. 等間隔 z_grid に補間（点数は velocity_array と同じに維持）
        z_grid = np.linspace(z_sorted[0], z_sorted[-1], len(z_sorted))
        f = interp1d(z_sorted, data_sorted, axis=2, kind="linear",
                     bounds_error=False, fill_value=0.0)
        data_z = f(z_grid)  # (n_x, n_y, n_z)

        # 4. reconstructed ステージへ遷移
        return replace(self, data=data_z, z_array=z_grid)

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
