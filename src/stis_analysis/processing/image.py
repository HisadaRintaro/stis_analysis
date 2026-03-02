"""処理済み画像モデル.

LA-Cosmic 後の FITS ファイルに対して以下を順番に適用するモジュール:
  - 連続光差し引き (subtract_continuum)
  - OIII λ4959 除去 (remove_o3_4959)
  - velocity range clipping (clip_velocity_range)
  - 連続光フィット確認プロット (plot_continuum_fit)

stistools.x2d による幾何補正はファイルベースのため ProcessingPipeline 側で実行する。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self, cast

import numpy as np
import matplotlib.pyplot as plt

from stis_analysis.lacosmic.image import ImageModel, ImageCollection
from stis_analysis.core.fits_reader import ReaderCollection
from stis_analysis.processing.wave_constants import (
    c_kms,
    oiii4959_stp,
    oiii5007_stp,
    oiii5007_oiii4959,
)


def _velocity_array(
    wavelength: np.ndarray,
    recession_velocity: float,
    rest_wavelength: float,
) -> np.ndarray:
    """波長配列を銀河フレームでの速度配列に変換する.

    Parameters
    ----------
    wavelength : np.ndarray
        波長配列 [Å]
    recession_velocity : float
        銀河の後退速度 [km/s]
    rest_wavelength : float
        静止系基準波長 [Å]

    Returns
    -------
    np.ndarray
        速度配列 [km/s]。正値が赤方偏移方向。
    """
    z = recession_velocity / c_kms
    lambda_ref = rest_wavelength * (1.0 + z)
    return c_kms * (wavelength / lambda_ref - 1.0)


@dataclass(frozen=True)
class ProcessingImageModel(ImageModel):
    """LA-Cosmic 後の STIS 画像に対するスペクトル処理モデル.

    lacosmic.ImageModel を継承し、以下のメソッドを追加する:
      - subtract_continuum: 連続光差し引き
      - remove_o3_4959: OIII λ4959 除去
      - clip_velocity_range: velocity range clipping
      - plot_continuum_fit: 連続光フィット確認プロット
    """

    # 追加フィールドなし（lacosmic.ImageModel の全フィールドを継承）

    def __repr__(self) -> str:
        dq_mask_info = (
            f"dq_mask_count={self.dq_mask.sum()}"
            if self.dq_mask is not None
            else "dq_mask=None"
        )
        contsub = self.primary_header.get("CONTSUB", False)
        o3corr = self.primary_header.get("O3CORR", False)
        vclip = self.primary_header.get("VCLIP", False)
        return (
            f"ProcessingImageModel(\n"
            f"  filename={self.filename},\n"
            f"  sci={self.sci.data.shape},\n"
            f"  err={self.err is not None},\n"
            f"  dq={self.dq is not None},\n"
            f"  {dq_mask_info},\n"
            f"  source_path={self.source_path},\n"
            f"  contsub={contsub}, o3corr={o3corr}, vclip={vclip},\n"
            f")"
        )

    # ------------------------------------------------------------------
    # 連続光差し引き
    # ------------------------------------------------------------------

    def subtract_continuum(
        self,
        continuum_windows_kms: list[tuple[float, float]],
        recession_velocity: float,
        rest_wavelength: float = oiii5007_stp,
        degree: int = 1,
    ) -> Self:
        """連続光を多項式フィットして差し引く.

        OIII λ5007（赤方偏移補正後）を速度 v=0 とした相対速度で
        連続光ウィンドウを指定する。各空間行に対して個別にフィットする。

        Parameters
        ----------
        continuum_windows_kms : list[tuple[float, float]]
            連続光として使用する速度ウィンドウ [km/s] のリスト。
            例: [(-4000, -3200), (3000, 4000)]
            OIII λ5007 観測波長を v=0 とした相対速度で指定する。
        recession_velocity : float
            銀河の後退速度 [km/s]（NGC1068: 1148）
        rest_wavelength : float, optional
            基準静止波長 [Å]（速度 v=0 の定義）。デフォルト: oiii5007_stp
        degree : int, optional
            多項式次数（デフォルト: 1 = 直線フィット）

        Returns
        -------
        ProcessingImageModel
            連続光差し引き済みの新しいインスタンス
        """
        wavelength = self.sci.wavelength
        if wavelength is None:
            raise ValueError("SCI ヘッダーに WCS 情報（CRVAL1/CDELT1）がありません。")

        velocity = _velocity_array(wavelength, recession_velocity, rest_wavelength)

        # 連続光ウィンドウのマスク
        cont_mask = np.zeros(len(wavelength), dtype=bool)
        for v_lo, v_hi in continuum_windows_kms:
            cont_mask |= (velocity >= v_lo) & (velocity <= v_hi)

        if cont_mask.sum() < degree + 1:
            raise ValueError(
                f"連続光ウィンドウ内のピクセル数（{cont_mask.sum()}）が"
                f"多項式次数 {degree} に対して不足しています。"
            )

        sci_data = self.sci.data.copy()
        pixel_indices = np.arange(len(wavelength), dtype=float)

        for row in range(sci_data.shape[0]):
            coeffs = np.polyfit(
                pixel_indices[cont_mask], sci_data[row, cont_mask], deg=degree
            )
            continuum = np.polyval(coeffs, pixel_indices)
            sci_data[row] -= continuum

        new_primary = self.primary_header.copy()
        new_primary["CONTSUB"] = (True, "Continuum subtraction applied")
        new_primary["CONTDEG"] = (degree, "Polynomial degree for continuum fit")
        for k, (v_lo, v_hi) in enumerate(continuum_windows_kms):
            new_primary[f"CWIN{k}LO"] = (v_lo, f"[km/s] Continuum window {k} lower bound")
            new_primary[f"CWIN{k}HI"] = (v_hi, f"[km/s] Continuum window {k} upper bound")
        new_primary.add_history(
            f"Continuum subtraction: degree={degree}, "
            f"windows={continuum_windows_kms} km/s "
            f"(ref {rest_wavelength:.3f} Ang, v_reces={recession_velocity} km/s)"
        )

        return replace(
            self,
            sci=replace(self.sci, data=sci_data),
            primary_header=new_primary,
        )

    # ------------------------------------------------------------------
    # OIII λ4959 除去
    # ------------------------------------------------------------------

    def remove_o3_4959(
        self,
        recession_velocity: float,
        scale: float | None = None,
        half_width_aa: float = 30.0,
    ) -> Self:
        """OIII λ4959 輝線を λ5007 プロファイルのスケールコピーで差し引く.

        OIII λ4959 の観測波長近傍のピクセルに対して、
        対応する λ5007 ピクセル（Δpix シフト分）を
        強度比 1/oiii5007_oiii4959 でスケーリングして差し引く。

        Parameters
        ----------
        recession_velocity : float
            銀河の後退速度 [km/s]（NGC1068: 1148）
        scale : float, optional
            スケーリング係数。デフォルト: 1 / oiii5007_oiii4959 = 1/2.98
        half_width_aa : float, optional
            4959 Å 近傍の処理対象半幅 [Å]（デフォルト: 30.0）

        Returns
        -------
        ProcessingImageModel
            OIII λ4959 除去済みの新しいインスタンス
        """
        wavelength = self.sci.wavelength
        if wavelength is None:
            raise ValueError("SCI ヘッダーに WCS 情報がありません。")

        scale_factor = scale if scale is not None else (1.0 / oiii5007_oiii4959)

        z = recession_velocity / c_kms
        lambda_4959_obs = oiii4959_stp * (1.0 + z)
        lambda_5007_obs = oiii5007_stp * (1.0 + z)

        # CDELT1 または CD1_1 からピクセルスケールを取得
        _cdelt1_raw = self.sci.header.get("CDELT1", self.sci.header.get("CD1_1"))
        if _cdelt1_raw is None or _cdelt1_raw == 0:
            raise ValueError("SCI ヘッダーに CDELT1 / CD1_1 がありません。")
        cdelt1 = float(_cdelt1_raw)  # type: ignore[arg-type]

        delta_lam = lambda_5007_obs - lambda_4959_obs  # [Å]
        delta_pix = int(round(delta_lam / cdelt1))

        # 4959 近傍のピクセルマスク
        target_mask = np.abs(wavelength - lambda_4959_obs) < half_width_aa

        sci_data = self.sci.data.copy()
        n_wave = sci_data.shape[1]

        for col_idx in np.where(target_mask)[0]:
            src_idx = col_idx + delta_pix
            if 0 <= src_idx < n_wave:
                sci_data[:, col_idx] -= sci_data[:, src_idx] * scale_factor

        new_primary = self.primary_header.copy()
        new_primary["O3CORR"] = (True, "OIII 4959 removal applied")
        new_primary["O3SCALE"] = (scale_factor, "Scale factor applied (1/F5007_F4959)")
        new_primary.add_history(
            f"OIII 4959 removal: scale={scale_factor:.4f}, "
            f"lambda_4959_obs={lambda_4959_obs:.3f} Ang "
            f"(v_reces={recession_velocity} km/s)"
        )

        return replace(
            self,
            sci=replace(self.sci, data=sci_data),
            primary_header=new_primary,
        )

    # ------------------------------------------------------------------
    # Velocity range clipping
    # ------------------------------------------------------------------

    def clip_velocity_range(
        self,
        v_min: float,
        v_max: float,
        recession_velocity: float,
        rest_wavelength: float = oiii5007_stp,
    ) -> Self:
        """速度範囲に対応する波長ピクセルだけを切り出す.

        OIII λ5007（赤方偏移補正後）を速度 v=0 として、
        指定した速度範囲 [v_min, v_max] に含まれる列を切り出す。
        WCS キーワード（CRVAL1/CRPIX1/NAXIS1）を更新し、
        後から確認できるヘッダーキーワードを primary_header に追加する。

        Parameters
        ----------
        v_min : float
            下限速度 [km/s]
        v_max : float
            上限速度 [km/s]
        recession_velocity : float
            銀河の後退速度 [km/s]（NGC1068: 1148）
        rest_wavelength : float, optional
            基準静止波長 [Å]。デフォルト: oiii5007_stp

        Returns
        -------
        ProcessingImageModel
            velocity clipping 済みの新しいインスタンス

        Raises
        ------
        ValueError
            指定速度範囲にピクセルが存在しない場合
        """
        wavelength = self.sci.wavelength
        if wavelength is None:
            raise ValueError("SCI ヘッダーに WCS 情報がありません。")

        velocity = _velocity_array(wavelength, recession_velocity, rest_wavelength)
        indices = np.where((velocity >= v_min) & (velocity <= v_max))[0]

        if len(indices) == 0:
            raise ValueError(
                f"速度範囲 [{v_min}, {v_max}] km/s にピクセルが存在しません。"
            )

        start_idx = int(indices[0])
        end_idx = int(indices[-1])

        # --- データをスライス ---
        clipped_sci = self.sci.data[:, start_idx : end_idx + 1]
        clipped_err = (
            replace(self.err, data=self.err.data[:, start_idx : end_idx + 1])
            if self.err is not None
            else None
        )
        clipped_dq = (
            replace(self.dq, data=self.dq.data[:, start_idx : end_idx + 1])
            if self.dq is not None
            else None
        )

        # --- WCS 更新 ---
        old_crval1 = float(self.sci.header.get("CRVAL1", wavelength[0]))  # type: ignore[arg-type]
        _cdelt1_raw = self.sci.header.get("CDELT1", self.sci.header.get("CD1_1", 1.0))
        old_cdelt1 = float(_cdelt1_raw)  # type: ignore[arg-type]
        old_crpix1 = float(self.sci.header.get("CRPIX1", 1.0))  # type: ignore[arg-type]
        new_crval1 = old_crval1 + old_cdelt1 * (start_idx - (old_crpix1 - 1.0))

        new_sci_header = self.sci.header.copy()
        try:
            crval1_comment = self.sci.header.comments["CRVAL1"]
        except (KeyError, IndexError):
            crval1_comment = "Reference value"
        new_sci_header["CRVAL1"] = (new_crval1, crval1_comment)
        new_sci_header["CRPIX1"] = (1.0, "Reference pixel")
        new_sci_header["NAXIS1"] = end_idx - start_idx + 1

        # --- 後から確認しやすいヘッダーキーワード ---
        z = recession_velocity / c_kms
        new_primary = self.primary_header.copy()
        new_primary["VCLIP"] = (True, "Velocity range clipping applied")
        new_primary["VCLIPMIN"] = (v_min, "[km/s] Lower velocity clip bound")
        new_primary["VCLIPMAX"] = (v_max, "[km/s] Upper velocity clip bound")
        new_primary["VCLIPREF"] = (rest_wavelength, "[Angstrom] Rest wavelength for v=0")
        new_primary["VRECES"] = (recession_velocity, "[km/s] Galaxy recession velocity used")
        new_primary["VCLIPZ"] = (z, "Redshift used (v_reces/c)")
        new_primary.add_history(
            f"Velocity clipping: {v_min} to {v_max} km/s"
        )
        new_primary.add_history(
            f"  Reference: {rest_wavelength:.3f} Ang "
            f"(v_reces={recession_velocity} km/s, z={z:.6f})"
        )

        return replace(
            self,
            sci=replace(self.sci, data=clipped_sci, header=new_sci_header),
            err=clipped_err,
            dq=clipped_dq,
            primary_header=new_primary,
        )

    # ------------------------------------------------------------------
    # 連続光フィット確認プロット
    # ------------------------------------------------------------------

    def plot_continuum_fit(
        self,
        slit_index: int,
        continuum_windows_kms: list[tuple[float, float]],
        recession_velocity: float,
        rest_wavelength: float = oiii5007_stp,
        degree: int = 1,
        ax=None,
    ) -> plt.Axes:  # pyright: ignore
        """連続光フィットウィンドウとフィット結果を可視化する.

        subtract_continuum() を実行する前に、ウィンドウ設定が
        適切かどうかを目視確認するためのプロット。

        Parameters
        ----------
        slit_index : int
            確認したいスリット位置（空間方向のインデックス）
        continuum_windows_kms : list[tuple[float, float]]
            subtract_continuum() に渡す予定の連続光ウィンドウ [km/s]
        recession_velocity : float
            銀河の後退速度 [km/s]
        rest_wavelength : float, optional
            基準静止波長 [Å]。デフォルト: oiii5007_stp
        degree : int, optional
            多項式次数（デフォルト: 1）
        ax : matplotlib.axes.Axes, optional
            描画先 Axes。None の場合は新規作成。

        Returns
        -------
        matplotlib.axes.Axes
            描画に使用した Axes オブジェクト
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        wavelength = self.sci.wavelength
        if wavelength is None:
            raise ValueError("SCI ヘッダーに WCS 情報がありません。")

        velocity = _velocity_array(wavelength, recession_velocity, rest_wavelength)
        spectrum = self.sci.data[slit_index, :]

        # スペクトルをプロット
        ax.plot(velocity, spectrum, color="steelblue", lw=0.8, label="spectrum")

        # 連続光ウィンドウをハイライト
        cont_mask = np.zeros(len(wavelength), dtype=bool)
        for k, (v_lo, v_hi) in enumerate(continuum_windows_kms):
            cont_mask |= (velocity >= v_lo) & (velocity <= v_hi)
            ax.axvspan(v_lo, v_hi, alpha=0.2, color="orange",
                       label=f"cont window {k}" if k == 0 else None)

        # 連続光フィット
        pixel_indices = np.arange(len(wavelength), dtype=float)
        if cont_mask.sum() >= degree + 1:
            coeffs = np.polyfit(
                pixel_indices[cont_mask], spectrum[cont_mask], deg=degree
            )
            continuum = np.polyval(coeffs, pixel_indices)
            ax.plot(velocity, continuum, color="red", lw=1.2,
                    linestyle="--", label=f"continuum fit (deg={degree})")

        # OIII λ4959 と λ5007 の位置をマーク
        z = recession_velocity / c_kms
        for lam, label in [
            (oiii4959_stp, "OIII 4959"),
            (oiii5007_stp, "OIII 5007"),
        ]:
            lam_obs = lam * (1.0 + z)
            v_line = c_kms * (lam_obs / (rest_wavelength * (1.0 + z)) - 1.0)
            ax.axvline(v_line, color="gray", linestyle=":", lw=1.0, label=label)

        ax.set_xlabel("Velocity [km/s]")
        ax.set_ylabel("Counts")
        ax.set_title(
            f"{self.filename} (slit={slit_index}, "
            f"v_reces={recession_velocity} km/s)"
        )
        ax.legend(fontsize="small")
        return ax

    # ------------------------------------------------------------------
    # write_fits オーバーライド（処理キーワード対応）
    # ------------------------------------------------------------------

    def write_fits(
        self,
        output_suffix: str = "_proc",
        output_dir: Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """処理済み画像を FITS ファイルとして出力する.

        parent の write_fits と異なり、デフォルト suffix が '_proc'。
        primary_header には各処理ステップのキーワードが既に格納済み
        （subtract_continuum / remove_o3_4959 / clip_velocity_range を
        適用した際に追加されている）。

        Parameters
        ----------
        output_suffix : str, optional
            出力ファイルの接尾辞（デフォルト: "_proc"）
        output_dir : Path | None, optional
            出力先ディレクトリ。None の場合は source_path を使用する
        overwrite : bool, optional
            既存ファイルの上書きを許可するか（デフォルト: False）

        Returns
        -------
        Path
            出力した FITS ファイルのパス
        """
        return super().write_fits(
            output_suffix=output_suffix,
            output_dir=output_dir,
            overwrite=overwrite,
        )


@dataclass(frozen=True)
class ProcessingImageCollection(ImageCollection):
    """複数の ProcessingImageModel をまとめて管理するコレクション.

    lacosmic.ImageCollection を継承し、各処理を一括適用するメソッドを追加する。
    """

    images: list[ProcessingImageModel]  # type: ignore[assignment]

    def __repr__(self) -> str:
        filenames = [img.filename for img in self.images]
        return (
            f"ProcessingImageCollection(\n"
            f"  n_images={len(self.images)},\n"
            f"  files={filenames},\n"
            f")"
        )

    @classmethod
    def from_readers(  # type: ignore[override]
        cls,
        readers: ReaderCollection,
        dq_flags: int = 16,
        **kwargs,
    ) -> "ProcessingImageCollection":
        """ReaderCollection から ProcessingImageCollection を生成する.

        Parameters
        ----------
        readers : ReaderCollection
            読み込み済みの FITS Reader コレクション
        dq_flags : int, optional
            マスク対象の DQ ビットフラグ（デフォルト: 16）

        Returns
        -------
        ProcessingImageCollection
        """
        images = [
            ProcessingImageModel.from_reader(reader, dq_flags=dq_flags)
            for reader in readers
        ]
        return cls(images=images, **kwargs)  # type: ignore[arg-type]

    def subtract_continuum(
        self,
        continuum_windows_kms: list[tuple[float, float]],
        recession_velocity: float,
        rest_wavelength: float = oiii5007_stp,
        degree: int = 1,
    ) -> "ProcessingImageCollection":
        """全画像に連続光差し引きを一括適用する."""
        images = [
            cast(ProcessingImageModel, image).subtract_continuum(
                continuum_windows_kms=continuum_windows_kms,
                recession_velocity=recession_velocity,
                rest_wavelength=rest_wavelength,
                degree=degree,
            )
            for image in self.images
        ]
        return replace(self, images=images)  # type: ignore[arg-type]

    def remove_o3_4959(
        self,
        recession_velocity: float,
        scale: float | None = None,
        half_width_aa: float = 30.0,
    ) -> "ProcessingImageCollection":
        """全画像に OIII λ4959 除去を一括適用する."""
        images = [
            cast(ProcessingImageModel, image).remove_o3_4959(
                recession_velocity=recession_velocity,
                scale=scale,
                half_width_aa=half_width_aa,
            )
            for image in self.images
        ]
        return replace(self, images=images)  # type: ignore[arg-type]

    def clip_velocity_range(
        self,
        v_min: float,
        v_max: float,
        recession_velocity: float,
        rest_wavelength: float = oiii5007_stp,
    ) -> "ProcessingImageCollection":
        """全画像に velocity range clipping を一括適用する."""
        images = [
            cast(ProcessingImageModel, image).clip_velocity_range(
                v_min=v_min,
                v_max=v_max,
                recession_velocity=recession_velocity,
                rest_wavelength=rest_wavelength,
            )
            for image in self.images
        ]
        return replace(self, images=images)  # type: ignore[arg-type]

    def write_fits(
        self,
        output_suffix: str = "_proc",
        output_dir: Path | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        """全画像を FITS ファイルとして一括出力する."""
        return [
            image.write_fits(
                output_suffix=output_suffix,
                output_dir=output_dir,
                overwrite=overwrite,
            )
            for image in self.images
        ]
