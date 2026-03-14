"""処理済み画像モデル.

LA-Cosmic 後の FITS ファイルに対して以下を順番に適用するモジュール:
  - 連続光差し引き (subtract_continuum)
  - OIII λ4959 除去 (remove_o3_4959)
  - velocity range clipping (clip_velocity_range)
  - 連続光フィット確認プロット (plot_continuum_fit)

stistools.x2d による幾何補正はファイルベースのため ProcessingPipeline 側で実行する。

使用方法:
    # 1. データ読み込み
    reader = STISFitsReader.open(path)
    model = ImageModel.from_reader(reader)

    # 2. 処理パラメータを設定して ProcessingImageModel を生成
    proc = ProcessingImageModel.setup(
        model,
        recession_velocity=1148.0,
        continuum_windows_kms=[(-4000, -3200), (3000, 4000)],
    )

    # 3. 処理（各メソッドは引数不要）
    result = proc.subtract_continuum().remove_o3_4959().clip_velocity_range(-2500, 2500)
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


@dataclass(frozen=True, kw_only=True)
class ProcessingImageModel(ImageModel):
    """LA-Cosmic 後の STIS 画像に対するスペクトル処理モデル.

    lacosmic.ImageModel を継承し、以下のメソッドを追加する:
      - subtract_continuum: 連続光差し引き
      - remove_o3_4959: OIII λ4959 除去
      - clip_velocity_range: velocity range clipping
      - plot_continuum_fit: 連続光フィット確認プロット

    インスタンス生成は setup() クラスメソッドを使用する。

    Attributes
    ----------
    recession_velocity : float
        銀河後退速度 [km/s]。全処理メソッドで使用する。
    continuum_windows_kms : tuple[tuple[float, float], ...]
        subtract_continuum() で使用する連続光ウィンドウ [km/s]。
    rest_wavelength : float
        速度 v=0 の基準静止波長 [Å]。デフォルト: oiii5007_stp
    o3_half_width_aa : float
        remove_o3_4959() で使用する除去領域半幅 [Å]。デフォルト: 30.0
    continuum : np.ndarray | None
        subtract_continuum() で計算した連続光モデル（shape: rows × cols）。
    """

    # 処理パラメータ（setup() で設定する）
    recession_velocity: float
    continuum_windows_kms: tuple[tuple[float, float], ...]
    rest_wavelength: float = oiii5007_stp
    o3_half_width_aa: float = 30.0

    # 処理結果フィールド（subtract_continuum() が設定する）
    continuum: np.ndarray | None = None

    # ------------------------------------------------------------------
    # インスタンス生成
    # ------------------------------------------------------------------

    @classmethod
    def setup(
        cls,
        model: ImageModel,
        recession_velocity: float,
        continuum_windows_kms: list[tuple[float, float]],
        rest_wavelength: float = oiii5007_stp,
        o3_half_width_aa: float = 30.0,
    ) -> "ProcessingImageModel":
        """ImageModel と処理パラメータから ProcessingImageModel を生成する.

        Parameters
        ----------
        model : ImageModel
            LA-Cosmic 後の ImageModel（from_reader() で生成済みのもの）
        recession_velocity : float
            銀河の後退速度 [km/s]（NGC1068: 1148）
        continuum_windows_kms : list[tuple[float, float]]
            連続光ウィンドウ [km/s]（例: [(-4000, -3200), (3000, 4000)]）
        rest_wavelength : float, optional
            速度 v=0 の基準静止波長 [Å]。デフォルト: oiii5007_stp
        o3_half_width_aa : float, optional
            OIII λ4959 除去領域半幅 [Å]。デフォルト: 30.0

        Returns
        -------
        ProcessingImageModel
        """
        return cls(
            primary_header=model.primary_header,
            sci=model.sci,
            err=model.err,
            dq=model.dq,
            dq_mask=model.dq_mask,
            cr_mask=model.cr_mask,
            source_path=model.source_path,
            dq_flags=model.dq_flags,
            recession_velocity=recession_velocity,
            continuum_windows_kms=tuple(continuum_windows_kms),
            rest_wavelength=rest_wavelength,
            o3_half_width_aa=o3_half_width_aa,
        )

    # ------------------------------------------------------------------
    # プロパティ（recession_velocity から逆算）
    # ------------------------------------------------------------------

    @property
    def oiii4959_obs(self) -> float:
        """OIII λ4959 観測波長 [Å]（recession_velocity から逆算）."""
        return oiii4959_stp * (1.0 + self.recession_velocity / c_kms)

    @property
    def oiii5007_obs(self) -> float:
        """OIII λ5007 観測波長 [Å]（recession_velocity から逆算）."""
        return oiii5007_stp * (1.0 + self.recession_velocity / c_kms)

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
            f"  recession_velocity={self.recession_velocity},\n"
            f"  contsub={contsub}, o3corr={o3corr}, vclip={vclip},\n"
            f")"
        )

    # ------------------------------------------------------------------
    # 連続光差し引き
    # ------------------------------------------------------------------

    @staticmethod
    def _continuum_mask(
        velocity: np.ndarray,
        continuum_windows_kms: tuple[tuple[float, float], ...],
        degree: int,
    ) -> np.ndarray:
        """連続光ウィンドウに対応する boolean マスクを生成する.

        Parameters
        ----------
        velocity : np.ndarray
            速度配列 [km/s]
        continuum_windows_kms : tuple[tuple[float, float], ...]
            連続光として使用する速度ウィンドウ [km/s] のリスト
        degree : int
            多項式次数（ピクセル数の下限チェックに使用）

        Returns
        -------
        np.ndarray
            連続光ウィンドウ内を True とする boolean マスク

        Raises
        ------
        ValueError
            ウィンドウ内のピクセル数が多項式次数に対して不足している場合
        """
        mask = np.zeros(len(velocity), dtype=bool)
        for v_lo, v_hi in continuum_windows_kms:
            mask |= (velocity >= v_lo) & (velocity <= v_hi)

        if mask.sum() < degree + 1:
            raise ValueError(
                f"連続光ウィンドウ内のピクセル数（{mask.sum()}）が"
                f"多項式次数 {degree} に対して不足しています。"
            )
        return mask

    def subtract_continuum(self, degree: int = 1) -> Self:
        """連続光を多項式フィットして差し引く.

        self.continuum_windows_kms / recession_velocity / rest_wavelength を使用する。
        各空間行に対して個別にフィットする。
        フィットした連続光モデルは self.continuum に保存される。

        Parameters
        ----------
        degree : int, optional
            多項式次数（デフォルト: 1 = 直線フィット）

        Returns
        -------
        ProcessingImageModel
            連続光差し引き済みの新しいインスタンス
        """
        windows = self.continuum_windows_kms
        velocity = self.sci.velocity_array(self.recession_velocity, self.rest_wavelength)
        cont_mask = self._continuum_mask(velocity, windows, degree)

        sci_data = self.sci.data.copy()
        continuum_data = np.zeros_like(sci_data)
        pixel_indices = np.arange(len(velocity), dtype=float)

        for row in range(sci_data.shape[0]):
            coeffs = np.polyfit(
                pixel_indices[cont_mask], sci_data[row, cont_mask], deg=degree
            )
            continuum_row = np.polyval(coeffs, pixel_indices)
            continuum_data[row] = continuum_row
            sci_data[row] -= continuum_row

        new_primary = self.primary_header.copy()
        new_primary["CONTSUB"] = (True, "Continuum subtraction applied")
        new_primary["CONTDEG"] = (degree, "Polynomial degree for continuum fit")
        for k, (v_lo, v_hi) in enumerate(windows):
            new_primary[f"CWIN{k}LO"] = (v_lo, f"[km/s] Continuum window {k} lower bound")
            new_primary[f"CWIN{k}HI"] = (v_hi, f"[km/s] Continuum window {k} upper bound")
        new_primary.add_history(
            f"Continuum subtraction: degree={degree}, "
            f"windows={list(windows)} km/s "
            f"(ref {self.rest_wavelength:.3f} Ang, v_reces={self.recession_velocity} km/s)"
        )

        return replace(
            self,
            sci=replace(self.sci, data=sci_data),
            primary_header=new_primary,
            continuum=continuum_data,
        )

    # ------------------------------------------------------------------
    # OIII λ4959 除去
    # ------------------------------------------------------------------

    def remove_o3_4959(
        self,
        scale: float = 1.0 / oiii5007_oiii4959,
    ) -> Self:
        """OIII λ4959 輝線を λ5007 プロファイルのスケールコピーで差し引く.

        self.recession_velocity / o3_half_width_aa を使用する。
        OIII λ4959 の観測波長近傍のピクセルに対して、
        対応する λ5007 ピクセル（Δpix シフト分）を
        強度比 1/oiii5007_oiii4959 でスケーリングして差し引く。

        Parameters
        ----------
        scale : float, optional
            スケーリング係数。デフォルト: 1 / oiii5007_oiii4959 ≈ 1/2.98

        Returns
        -------
        ProcessingImageModel
            OIII λ4959 除去済みの新しいインスタンス
        """
        wavelength = self.sci.wavelength
        z = self.recession_velocity / c_kms
        lambda_4959_obs = oiii4959_stp * (1.0 + z)
        lambda_5007_obs = oiii5007_stp * (1.0 + z)

        delta_lam = lambda_5007_obs - lambda_4959_obs  # [Å]
        delta_pix = int(round(delta_lam / self.sci.cdelt1))

        target_mask = np.abs(wavelength - lambda_4959_obs) < self.o3_half_width_aa

        sci_data = self.sci.data.copy()
        n_wave = sci_data.shape[1]
        target_cols = np.where(target_mask)[0]
        src_cols = target_cols + delta_pix
        valid = (src_cols >= 0) & (src_cols < n_wave)
        sci_data[:, target_cols[valid]] -= sci_data[:, src_cols[valid]] * scale

        new_primary = self.primary_header.copy()
        new_primary["O3CORR"] = (True, "OIII 4959 removal applied")
        new_primary["O3SCALE"] = (scale, "Scale factor applied (1/F5007_F4959)")
        new_primary.add_history(
            f"OIII 4959 removal: scale={scale:.4f}, "
            f"lambda_4959_obs={lambda_4959_obs:.3f} Ang "
            f"(v_reces={self.recession_velocity} km/s)"
        )

        return replace(
            self,
            sci=replace(self.sci, data=sci_data),
            primary_header=new_primary,
        )

    # ------------------------------------------------------------------
    # Velocity range clipping
    # ------------------------------------------------------------------

    def clip_velocity_range(self, v_min: float, v_max: float) -> Self:
        """速度範囲に対応する波長ピクセルだけを切り出す.

        self.recession_velocity / rest_wavelength を使用する。
        OIII λ5007（赤方偏移補正後）を速度 v=0 として、
        指定した速度範囲 [v_min, v_max] に含まれる列を切り出す。
        格納済みの continuum も同様にスライスする。

        Parameters
        ----------
        v_min : float
            下限速度 [km/s]
        v_max : float
            上限速度 [km/s]

        Returns
        -------
        ProcessingImageModel
            velocity clipping 済みの新しいインスタンス

        Raises
        ------
        ValueError
            指定速度範囲にピクセルが存在しない場合
        """
        velocity = self.sci.velocity_array(self.recession_velocity, self.rest_wavelength)
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
        clipped_continuum = (
            self.continuum[:, start_idx : end_idx + 1]
            if self.continuum is not None
            else None
        )

        # --- WCS 更新 ---
        old_crval1 = float(self.sci.header.get("CRVAL1", self.sci.wavelength[0]))  # type: ignore[arg-type]
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

        # --- ヘッダーキーワード ---
        z = self.recession_velocity / c_kms
        new_primary = self.primary_header.copy()
        new_primary["VCLIP"] = (True, "Velocity range clipping applied")
        new_primary["VCLIPMIN"] = (v_min, "[km/s] Lower velocity clip bound")
        new_primary["VCLIPMAX"] = (v_max, "[km/s] Upper velocity clip bound")
        new_primary["VCLIPREF"] = (self.rest_wavelength, "[Angstrom] Rest wavelength for v=0")
        new_primary["VRECES"] = (self.recession_velocity, "[km/s] Galaxy recession velocity used")
        new_primary["VCLIPZ"] = (z, "Redshift used (v_reces/c)")
        new_primary.add_history(f"Velocity clipping: {v_min} to {v_max} km/s")
        new_primary.add_history(
            f"  Reference: {self.rest_wavelength:.3f} Ang "
            f"(v_reces={self.recession_velocity} km/s, z={z:.6f})"
        )

        return replace(
            self,
            sci=replace(self.sci, data=clipped_sci, header=new_sci_header),
            err=clipped_err,
            dq=clipped_dq,
            primary_header=new_primary,
            continuum=clipped_continuum,
        )

    # ------------------------------------------------------------------
    # 連続光フィット確認プロット
    # ------------------------------------------------------------------

    def plot_continuum_fit(
        self,
        slit_index: int,
        ax=None,
    ) -> plt.Axes:  # pyright: ignore
        """連続光フィット結果を ax にアノテーションとして描画する.

        subtract_continuum() 実行後に呼び出す。スペクトル自体は描画しない。
        既存の ax に重ねることで「before スペクトル + 連続光フィット」を
        一枚のプロットに表示できる。

        描画内容:
          - 連続光フィット線（破線、赤）
          - 連続光ウィンドウ（橙シェード）
          - OIII λ4959 / λ5007 観測波長（グレー縦線）
          - OIII λ4959 除去領域（紫シェード）

        Parameters
        ----------
        slit_index : int
            確認したいスリット位置（空間方向のインデックス）
        ax : matplotlib.axes.Axes, optional
            描画先 Axes。None の場合は新規作成。

        Returns
        -------
        matplotlib.axes.Axes
            描画に使用した Axes オブジェクト

        Raises
        ------
        ValueError
            continuum が未設定（subtract_continuum() 未実行）の場合
        """
        if self.continuum is None:
            raise ValueError(
                "continuum が未設定です。subtract_continuum() を先に実行してください。"
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        wavelength = self.sci.wavelength

        # 連続光フィット線
        ax.plot(wavelength, self.continuum[slit_index], color="red", lw=1.2,
                linestyle="--", label="continuum fit")

        # 連続光ウィンドウをハイライト（km/s → Å 変換）
        z = self.recession_velocity / c_kms
        lambda_ref = self.rest_wavelength * (1.0 + z)
        for k, (v_lo, v_hi) in enumerate(self.continuum_windows_kms):
            lam_lo = lambda_ref * (1.0 + v_lo / c_kms)
            lam_hi = lambda_ref * (1.0 + v_hi / c_kms)
            ax.axvspan(lam_lo, lam_hi, alpha=0.2, color="orange",
                       label="cont window" if k == 0 else None)

        # OIII 輝線位置
        ax.axvline(self.oiii4959_obs, color="gray", linestyle=":", lw=1.0,
                   label="OIII 4959")
        ax.axvline(self.oiii5007_obs, color="gray", linestyle="-.", lw=1.0,
                   label="OIII 5007")

        # OIII λ4959 除去領域
        ax.axvspan(
            self.oiii4959_obs - self.o3_half_width_aa,
            self.oiii4959_obs + self.o3_half_width_aa,
            alpha=0.15, color="purple",
            label=f"OIII 4959 removal (±{self.o3_half_width_aa:.0f} Å)",
        )

        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel(f"flux [{self.sci.unit}]")
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
        """処理済み画像を FITS ファイルとして出力する."""
        return super().write_fits(
            output_suffix=output_suffix,
            output_dir=output_dir,
            overwrite=overwrite,
        )


@dataclass(frozen=True)
class ProcessingImageCollection(ImageCollection):
    """複数の ProcessingImageModel をまとめて管理するコレクション."""

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
    def setup(
        cls,
        source: ReaderCollection,
        recession_velocity: float,
        continuum_windows_kms: list[tuple[float, float]],
        rest_wavelength: float = oiii5007_stp,
        o3_half_width_aa: float = 30.0,
        dq_flags: int = 16,
    ) -> "ProcessingImageCollection":
        """ReaderCollection と処理パラメータから ProcessingImageCollection を生成する.

        Parameters
        ----------
        source : ReaderCollection
            読み込み済みの FITS Reader コレクション
        recession_velocity : float
            銀河の後退速度 [km/s]（NGC1068: 1148）
        continuum_windows_kms : list[tuple[float, float]]
            連続光ウィンドウ [km/s]（例: [(-4000, -3200), (3000, 4000)]）
        rest_wavelength : float, optional
            速度 v=0 の基準静止波長 [Å]。デフォルト: oiii5007_stp
        o3_half_width_aa : float, optional
            OIII λ4959 除去領域半幅 [Å]。デフォルト: 30.0
        dq_flags : int, optional
            マスク対象の DQ ビットフラグ（デフォルト: 16 = hot pixel）

        Returns
        -------
        ProcessingImageCollection
        """
        images = [
            ProcessingImageModel.setup(
                ImageModel.from_reader(reader, dq_flags=dq_flags),
                recession_velocity=recession_velocity,
                continuum_windows_kms=continuum_windows_kms,
                rest_wavelength=rest_wavelength,
                o3_half_width_aa=o3_half_width_aa,
            )
            for reader in source
        ]
        return cls(images=images)  # type: ignore[arg-type]

    def subtract_continuum(self, degree: int = 1) -> "ProcessingImageCollection":
        """全画像に連続光差し引きを一括適用する."""
        images = [
            cast(ProcessingImageModel, image).subtract_continuum(degree=degree)
            for image in self.images
        ]
        return replace(self, images=images)  # type: ignore[arg-type]

    def remove_o3_4959(
        self,
        scale: float = 1.0 / oiii5007_oiii4959,
    ) -> "ProcessingImageCollection":
        """全画像に OIII λ4959 除去を一括適用する."""
        images = [
            cast(ProcessingImageModel, image).remove_o3_4959(scale=scale)
            for image in self.images
        ]
        return replace(self, images=images)  # type: ignore[arg-type]

    def clip_velocity_range(
        self,
        v_min: float,
        v_max: float,
    ) -> "ProcessingImageCollection":
        """全画像に velocity range clipping を一括適用する."""
        images = [
            cast(ProcessingImageModel, image).clip_velocity_range(v_min=v_min, v_max=v_max)
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
