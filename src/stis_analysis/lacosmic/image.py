"""画像モデル.

STIS FITS ファイルから読み込んだ画像データを管理し、
宇宙線除去 (LA-Cosmic) および可視化機能を提供するモジュール。

移植元: stis_la_cosmic/image.py
変更点:
- ImageUnit を stis_analysis.core.image からインポート
- STISFitsReader, ReaderCollection を stis_analysis.core.fits_reader からインポート
- ImageModel.imshow_mask(): self.dq_mask.data → self.dq_mask (バグ修正)
- ImageCollection.imshow_mask(): dq_mask=None の場合の fallback を追加 (バグ修正)
"""

import math
from typing import Literal
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Self
from matplotlib.colors import AsinhNorm, Normalize
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from lacosmic import remove_cosmics  # type: ignore
from scipy.ndimage import median_filter  # type: ignore

from stis_analysis.core.image import ImageUnit
from stis_analysis.core.fits_reader import ReaderCollection, STISFitsReader
from stis_analysis.core.wave_constants import c_kms, oiii5007_stp


@dataclass(frozen=True)
class ImageModel:
    """STIS 画像データの単一フレームモデル.

    1つの FITS ファイルに対応する科学画像・誤差・DQ・マスクを
    ImageUnit として保持し、宇宙線除去および画像表示の機能を提供する。

    Attributes
    ----------
    primary_header : fits.Header
        Primary Header（HDU 0）
    sci : ImageUnit
        科学画像（SCI, HDU 1）の data / header ペア
    err : ImageUnit | None
        誤差（ERR, HDU 2）の data / header ペア。None の場合は存在しない
    dq : ImageUnit | None
        DQ（HDU 3）の data / header ペア。None の場合は存在しない
    cr_mask : ImageUnit | None
        宇宙線マスク。LA-Cosmic 後 HDU に追加される（EXTNAME='LACOSMIC'）
    source_path : Path | None
        元の FITS ファイルが格納されているディレクトリパス
    dq_flags : int
        マスク対象の DQ ビットフラグ（デフォルト: 16 = hot pixel）
    dq_mask : np.ndarray | None
        DQ ビットフラグに基づいて生成された bad pixel マスク（True = bad pixel）
    """

    primary_header: fits.Header
    sci: ImageUnit
    err: ImageUnit | None = None
    dq: ImageUnit | None = None
    cr_mask: ImageUnit | None = None
    source_path: Path | None = None
    dq_flags: int = 16
    dq_mask: np.ndarray | None = None

    def __repr__(self) -> str:
        dq_mask_info = (
            f"dq_mask_count={self.dq_mask.sum()}"
            if self.dq_mask is not None
            else "dq_mask=None"
        )
        cr_mask_info = (
            f"cr_mask_count={self.cr_mask.data.sum()}"
            if self.cr_mask is not None
            else "cr_mask=None"
        )
        return (
            f"ImageModel(\n"
            f"  sci={self.sci.data.shape},\n"
            f"  err={self.err is not None},\n"
            f"  dq={self.dq is not None},\n"
            f"  {cr_mask_info},\n"
            f"  source_path={self.source_path}\n"
            f"  dq_flags={self.dq_flags}\n"
            f"  {dq_mask_info},\n"
            f"  gain={self.gain}\n"
            f"  read_noise={self.read_noise}\n"
            f")"
        )

    @classmethod
    def from_reader(
        cls,
        reader: STISFitsReader,
        dq_flags: int = 16,
    ) -> Self:
        """STISFitsReader から ImageModel を生成する.

        Parameters
        ----------
        reader : STISFitsReader
            読み込み済みの FITS Reader
        dq_flags : int, optional
            マスク対象の DQ ビットフラグ（デフォルト: 16 = hot pixel）。
            複数フラグはビット OR で指定（例: 16 | 256）

        Returns
        -------
        ImageModel
            生成されたモデル
        """
        try:
            err = ImageUnit(
                data=reader.image_data(2),
                header=reader.header(2),
            )
        except KeyError:
            err = None
        try:
            dq_data = reader.image_data(3)
            dq = ImageUnit(
                data=dq_data,
                header=reader.header(3),
            )
            dq_mask = (dq_data & dq_flags).astype(bool)
        except KeyError:
            dq = None
            dq_mask = None

        # LACOSMIC 拡張（HDU 4）があれば読み込む
        cr_mask = None
        for hdu_idx, hdr in reader.headers.items():
            if hdr.get("EXTNAME") == "LACOSMIC" and hdu_idx in reader.data:
                cr_mask = ImageUnit(data=reader.data[hdu_idx], header=hdr)
                break

        return cls(
            primary_header=reader.header(0),
            sci=ImageUnit(
                data=reader.image_data(1),
                header=reader.header(1),
            ),
            err=err,
            dq=dq,
            dq_mask=dq_mask,
            cr_mask=cr_mask,
            source_path=reader.filename.parent,
            dq_flags=dq_flags,
        )

    @staticmethod
    def median_interpolate(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """マスク対象ピクセルを隣接8ピクセルの中央値で補間する.

        各 bad pixel に対して、周囲8近傍のうちマスクされていない
        ピクセルの中央値で値を置換する。有効な近傍がない場合は
        元の値を保持する。

        Parameters
        ----------
        image : np.ndarray
            2次元の画像データ配列
        mask : np.ndarray
            boolean マスク配列（True = bad pixel）

        Returns
        -------
        np.ndarray
            補間済みの画像データ配列（コピー）
        """
        result = image.copy()
        median = median_filter(image, size=3)
        result[mask] = median[mask]
        return result

    def interpolate_bad_pixels(
        self,
        mask_negative: bool = True,
    ) -> Self:
        """DQ マスクおよび負値ピクセルを中央値補間した ImageModel を返す.

        DQ フラグに基づく bad pixel マスクと、オプションで負の値を持つ
        ピクセルのマスクを合成し、該当ピクセルを周囲8近傍の中央値で補間する。

        Parameters
        ----------
        mask_negative : bool, optional
            True の場合、負の値を持つピクセルもマスク対象にする（デフォルト: True）

        Returns
        -------
        ImageModel
            補間済み ImageModel
        """
        dq_mask = (
            self.dq_mask
            if self.dq_mask is not None
            else np.zeros(self.shape, dtype=bool)
        )
        neg_mask = (
            (self.sci.data < 0)
            if mask_negative
            else np.zeros(self.shape, dtype=bool)
        )
        combined_mask = dq_mask | neg_mask

        if combined_mask.any():
            interpolated = self.median_interpolate(self.sci.data, combined_mask)
        else:
            interpolated = self.sci.data

        return replace(
            self,
            sci=replace(self.sci, data=interpolated),
            dq_mask=combined_mask,
        )

    @property
    def gain(self) -> float:
        """ゲイン [e-/ADU]（Primary Header の ATODGAIN）."""
        val = self.primary_header.get("ATODGAIN")
        if val is None:
            warnings.warn(f"{self.filename}: ATODGAIN not found in header, using 1.0")
            return 1.0
        return float(val)  # type: ignore[arg-type]

    @property
    def read_noise(self) -> float:
        """リードノイズ [e-]（Primary Header の READNSE）."""
        val = self.primary_header.get("READNSE")
        if val is None:
            warnings.warn(f"{self.filename}: READNSE not found in header, using 2.5")
            return 2.5
        return float(val)  # type: ignore[arg-type]

    def remove_cosmic_ray(
        self,
        contrast: float = 5.0,
        cr_threshold: float = 5,
        neighbor_threshold: float = 5,
        mask_negative: bool = True,
        **kwargs,
    ) -> Self:
        """LA-Cosmic アルゴリズムにより宇宙線を除去する.

        interpolate_bad_pixels で前処理を行った後、
        lacosmic.remove_cosmics で宇宙線ヒットを検出・除去する。
        宇宙線マスクは戻り値の .cr_mask 属性に格納される。

        Parameters
        ----------
        contrast : float, optional
            ラプラシアン/ノイズ比のコントラスト閾値（デフォルト: 5.0）
        cr_threshold : float, optional
            宇宙線検出のシグマクリッピング閾値（デフォルト: 5）
        neighbor_threshold : float, optional
            近傍ピクセルの検出閾値（デフォルト: 5）
        mask_negative : bool, optional
            True の場合、負の値を持つピクセルもマスク対象にする（デフォルト: True）
        **kwargs
            lacosmic.remove_cosmics に渡す追加キーワード引数

        Returns
        -------
        ImageModel
            宇宙線除去済み画像を持つ新しいインスタンス。
            .cr_mask に宇宙線マスクが格納される
        """
        preprocessed = self.interpolate_bad_pixels(mask_negative=mask_negative)

        clean_data, cr_mask = remove_cosmics(
            preprocessed.sci.data,
            contrast,
            cr_threshold,
            neighbor_threshold,
            mask=preprocessed.dq_mask,
            effective_gain=self.gain,
            readnoise=self.read_noise,
            **kwargs,
        )

        cr_mask_header = fits.Header()
        cr_mask_header["EXTNAME"] = "LACOSMIC"
        cr_mask_unit = ImageUnit(data=cr_mask, header=cr_mask_header)

        return replace(
            self,
            sci=replace(self.sci, data=clean_data),
            cr_mask=cr_mask_unit,
        )

    @staticmethod
    def _resolve_output_path(
        source_path: Path | None,
        output_dir: Path | None,
        filename: str,
        output_suffix: str,
        overwrite: bool,
    ) -> Path:
        """出力パスを決定し、上書き防止チェックを行う.

        Parameters
        ----------
        source_path : Path | None
            元ファイルのディレクトリパス
        output_dir : Path | None
            出力先ディレクトリ。None の場合は source_path を使用する
        filename : str
            ファイルのルートネーム（ROOTNAME）
        output_suffix : str
            出力ファイルの接尾辞
        overwrite : bool
            既存ファイルの上書きを許可するか

        Returns
        -------
        Path
            出力先のフルパス

        Raises
        ------
        ValueError
            source_path が未設定かつ output_dir も指定されていない場合
        FileExistsError
            出力ファイルが既に存在し overwrite=False の場合
        """
        dest_dir = output_dir or source_path
        if dest_dir is None:
            raise ValueError(
                "出力先を決定できません。source_path が未設定のため、"
                "output_dir を指定してください。"
            )
        output_path = dest_dir / f"{filename}{output_suffix}.fits"
        if not overwrite and output_path.exists():
            raise FileExistsError(
                f"出力ファイルが既に存在します: {output_path}"
            )
        return output_path

    @staticmethod
    def _build_primary_header(
        primary_header: fits.Header | None,
        lacorr_applied: bool = True,
    ) -> fits.Header:
        """Primary Header を準備し、必要に応じて LACORR キーワードを挿入する.

        lacorr_applied=True の場合は CALIBRATION SWITCHES セクション末尾に
        LACORR カードと history を追加する。False の場合は追加しない。

        Parameters
        ----------
        primary_header : fits.Header | None
            元の Primary Header。None の場合は空ヘッダーを新規作成する
        lacorr_applied : bool, optional
            True の場合のみ LACORR=True キーワードと history を追加する
            （デフォルト: True）

        Returns
        -------
        fits.Header
            処理済みの Primary Header
        """
        header = primary_header.copy() if primary_header is not None else fits.Header()

        if not lacorr_applied:
            return header

        # CAL SWITCHES セクション末尾の空白カードインデックスを探す
        section_end = None
        in_cal_section = False
        cards = list(header.cards)
        for i, card in enumerate(cards):
            if card.keyword == '' and 'CALIBRATION SWITCHES' in card.comment:
                in_cal_section = True
                continue
            if in_cal_section and card.keyword == '' and cards[i - 1].keyword != '':
                section_end = i
                break

        lacorr_card = fits.Card('LACORR', True, 'LA-Cosmic correction applied')
        if section_end is not None:
            header.insert(section_end, lacorr_card)
        else:
            header.append(lacorr_card)

        header.add_history('LA-Cosmic cosmic ray rejection applied (stis_analysis.lacosmic)')
        return header

    def write_fits(
        self,
        output_suffix: str = "_lac",
        output_dir: Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """画像を FITS ファイルとして出力する.

        LA-Cosmic が適用済み（cr_mask の EXTNAME が 'LACOSMIC'）の場合は
        PrimaryHDU に LACORR=True キーワードと history を追加する。
        未適用のまま '_lac' suffix で書き出そうとした場合は UserWarning を発行する。

        Parameters
        ----------
        output_suffix : str, optional
            出力ファイルの接尾辞（デフォルト: "_lac"）
        output_dir : Path | None, optional
            出力先ディレクトリ。None の場合は source_path を使用する
        overwrite : bool, optional
            既存ファイルの上書きを許可するか（デフォルト: False）

        Returns
        -------
        Path
            出力した FITS ファイルのパス

        Raises
        ------
        ValueError
            source_path が設定されておらず output_dir も指定されていない場合
        FileExistsError
            出力ファイルが既に存在し overwrite=False の場合
        """
        lacorr_applied = (
            self.cr_mask is not None
            and self.cr_mask.header.get("EXTNAME") == "LACOSMIC"
        )
        if not lacorr_applied and output_suffix == "_lac":
            warnings.warn(
                f"{self.filename}: LA-Cosmic が未適用ですが '"
                f"{output_suffix}' suffix で書き出しています。"
                " remove_cosmic_ray() を実行済みか確認してください。",
                UserWarning,
                stacklevel=2,
            )
        output_path = self._resolve_output_path(
            self.source_path, output_dir, self.filename, output_suffix, overwrite
        )
        primary_header = self._build_primary_header(
            self.primary_header, lacorr_applied=lacorr_applied
        )
        hdu_list: list[fits.PrimaryHDU | fits.ImageHDU] = [
            fits.PrimaryHDU(header=primary_header),
            self.sci.to_hdu(),
        ]
        if self.err is not None:
            hdu_list.append(self.err.to_hdu())
        if self.dq is not None:
            hdu_list.append(self.dq.to_hdu())
        if self.cr_mask is not None:
            hdu_list.append(self.cr_mask.to_hdu())
        fits.HDUList(hdu_list).writeto(output_path, overwrite=overwrite)
        return output_path

    def plot_lacosmic_residual(
        self,
        other: "ImageModel",
        slit_index: int,
        recession_velocity: float,
        rest_wavelength: float = oiii5007_stp,
        v_range: tuple[float, float] = (-3000.0, 3000.0),
        labels: tuple[str, str] = ("before", "after (LA-Cosmic)"),
        axes=None,
    ) -> tuple:  # pyright: ignore
        """LA-Cosmic 除去残差スペクトルを 2 段プロットで確認する.

        上段: self (before) と other (after) スペクトルを重ね描き。
        下段: 残差 = before − after（LA-Cosmic が除去した分）。
              CR マスクされたピクセルを赤い点でマーク。

        Parameters
        ----------
        other : ImageModel
            LA-Cosmic 処理後の画像（after）
        slit_index : int
            確認するスリット行のインデックス
        recession_velocity : float
            銀河の後退速度 [km/s]
        rest_wavelength : float, optional
            速度 v=0 の基準静止波長 [Å]（デフォルト: oiii5007_stp）
        v_range : tuple[float, float], optional
            速度表示範囲 [km/s]（デフォルト: (-3000, 3000)）
        labels : tuple[str, str], optional
            凡例ラベル（before, after の順）
        axes : tuple | None, optional
            (ax_top, ax_bottom) の Axes タプル。None の場合は新規作成。

        Returns
        -------
        tuple
            (ax_top, ax_bottom) の Axes タプル
        """
        z = recession_velocity / c_kms
        lambda_ref = rest_wavelength * (1.0 + z)

        wl = self.sci.wavelength
        if wl is None:
            raise ValueError("波長情報が存在しません（WCS が設定されていません）")

        vel = c_kms * (wl / lambda_ref - 1.0)
        spec_before = self.sci.data[slit_index, :]
        spec_after = other.sci.data[slit_index, :]
        residual = spec_before - spec_after

        v_lo, v_hi = v_range
        mask_v = (vel >= v_lo) & (vel <= v_hi)

        cr_row = (
            other.cr_mask.data[slit_index, :].astype(bool)
            if other.cr_mask is not None
            else np.zeros(len(spec_before), dtype=bool)
        )

        if axes is None:
            _, (ax_top, ax_bottom) = plt.subplots(
                2, 1, figsize=(8, 6), sharex=True,
                gridspec_kw={"height_ratios": [2, 1]},
            )
        else:
            ax_top, ax_bottom = axes

        # 上段: before vs after
        ax_top.plot(vel[mask_v], spec_before[mask_v],
                    color="steelblue", lw=0.8, alpha=0.9, label=labels[0])
        ax_top.plot(vel[mask_v], spec_after[mask_v],
                    color="tomato", lw=0.8, alpha=0.9, label=labels[1])
        ax_top.set_ylabel("Counts")
        ax_top.set_title(f"{self.filename}  slit={slit_index}")
        ax_top.legend(fontsize="small")

        # 下段: 残差
        ax_bottom.plot(vel[mask_v], residual[mask_v],
                       color="dimgray", lw=0.8, label="residual (before − after)")
        cr_in_range = cr_row & mask_v
        if cr_in_range.any():
            ax_bottom.scatter(
                vel[cr_in_range], residual[cr_in_range],
                color="red", s=12, zorder=5, label="CR-flagged pixels",
            )
        ax_bottom.axhline(0, color="k", lw=0.5, ls="--")
        ax_bottom.set_xlabel("Velocity [km/s]")
        ax_bottom.set_ylabel("Residual")
        ax_bottom.legend(fontsize="small")

        return ax_top, ax_bottom

    def plot_spectrum(
        self,
        slit_index: int,
        ax=None,
        **kwargs,
    ) -> plt.Axes:  # pyright: ignore
        """指定スリット位置での波長方向スペクトルをプロットする.

        sci の data から slit_index 行を取り出し、横軸を波長（またはピクセル）、
        縦軸をカウントとしてプロットする。

        Parameters
        ----------
        slit_index : int
            スリット方向（y 軸）のインデックス
        ax : matplotlib.axes.Axes, optional
            描画先の Axes オブジェクト。None の場合は新規作成する
        **kwargs
            matplotlib.axes.Axes.plot に渡す追加キーワード引数

        Returns
        -------
        matplotlib.axes.Axes
            描画に使用した Axes オブジェクト
        """
        if ax is None:
            _, ax = plt.subplots()

        spectrum = self.sci.data[slit_index, :]
        wavelength = self.sci.wavelength

        if wavelength is not None:
            ax.plot(wavelength, spectrum, **kwargs)
            ax.set_xlabel("Wavelength [Å]")
        else:
            ax.plot(spectrum, **kwargs)
            ax.set_xlabel("Pixel")

        ax.set_ylabel("Counts")
        ax.set_title(f"{self.filename} (slit index = {slit_index})")
        return ax

    def imshow(self, ax=None, **kwargs) -> plt.Axes:  # pyright: ignore
        """画像データを matplotlib で表示する.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            描画先の Axes オブジェクト。None の場合は新規作成する
        **kwargs
            matplotlib.axes.Axes.imshow に渡す追加キーワード引数

        Returns
        -------
        matplotlib.axes.Axes
            描画に使用した Axes オブジェクト
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(self.sci.data, **kwargs)
        return ax

    def imshow_mask(self, ax=None, **kwargs) -> plt.Axes:  # pyright: ignore
        """DQ マスク画像を matplotlib で表示する.

        DQ フラグから生成された bad pixel マスク（dq_mask）を可視化する。
        dq_mask が設定されていない場合は全ゼロの画像を表示する。

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            描画先の Axes オブジェクト。None の場合は新規作成する
        **kwargs
            matplotlib.axes.Axes.imshow に渡す追加キーワード引数

        Returns
        -------
        matplotlib.axes.Axes
            描画に使用した Axes オブジェクト
        """
        if ax is None:
            _, ax = plt.subplots()
        # 修正: dq_mask は np.ndarray | None であり .data は不要
        mask_data = (
            self.dq_mask
            if self.dq_mask is not None
            else np.zeros(self.shape, dtype=bool)
        )
        ax.imshow(mask_data, cmap="gray", **kwargs)
        ax.set_title(f"{self.filename} (DQ Flag = {self.dq_flags})")
        return ax

    def imshow_cr_mask(self, ax=None, **kwargs) -> plt.Axes:  # pyright: ignore
        """LA-Cosmic マスク画像を matplotlib で表示する.

        remove_cosmic_ray() 後の宇宙線マスク（cr_mask）を可視化する。
        cr_mask が設定されていない場合は全ゼロの画像を表示する。

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            描画先の Axes オブジェクト。None の場合は新規作成する
        **kwargs
            matplotlib.axes.Axes.imshow に渡す追加キーワード引数

        Returns
        -------
        matplotlib.axes.Axes
            描画に使用した Axes オブジェクト
        """
        if ax is None:
            _, ax = plt.subplots()
        mask_data = (
            self.cr_mask.data
            if self.cr_mask is not None
            else np.zeros(self.shape, dtype=bool)
        )
        ax.imshow(mask_data, cmap="gray", **kwargs)
        ax.set_title(f"{self.filename} (LA-Cosmic mask)")
        return ax

    @property
    def shape(self) -> tuple[int, int]:
        return self.sci.data.shape  # type: ignore[return-value]

    @property
    def filename(self) -> str:
        try:
            return self.sci.header["ROOTNAME"]  # pyright: ignore
        except KeyError:
            return "UNKNOWN"


@dataclass(frozen=True)
class ImageCollection:
    """複数の ImageModel をまとめて管理するコレクション.

    複数フレームの画像に対して一括での宇宙線除去と
    グリッド表示を提供する。LA-Cosmic パラメータを
    コレクション全体で共有する。

    Attributes
    ----------
    images : list[ImageModel]
        管理対象の ImageModel リスト
    contrast : float
        ラプラシアン/ノイズ比のコントラスト閾値（デフォルト: 5.0）
    cr_threshold : float
        宇宙線検出のシグマクリッピング閾値（デフォルト: 5）
    neighbor_threshold : float
        近傍ピクセルの検出閾値（デフォルト: 5）
    """

    images: list[ImageModel]
    contrast: float = 5.0
    cr_threshold: float = 5
    neighbor_threshold: float = 5

    def __repr__(self) -> str:
        return (
            f"ImageCollection({len(self.images)} images, \n"
            + f"contrast={self.contrast}, \n"
            + f"cr_threshold={self.cr_threshold}, \n"
            + f"neighbor_threshold={self.neighbor_threshold}, \n"
        )

    @classmethod
    def from_readers(
        cls,
        readers: ReaderCollection,
        dq_flags: int = 16,
        contrast: float = 5.0,
        cr_threshold: float = 5,
        neighbor_threshold: float = 5,
    ) -> Self:
        """ReaderCollection から ImageCollection を生成する.

        Parameters
        ----------
        readers : ReaderCollection
            読み込み済みの FITS Reader コレクション
        dq_flags : int, optional
            マスク対象の DQ ビットフラグ（デフォルト: 16 = hot pixel）
        contrast : float, optional
            ラプラシアン/ノイズ比のコントラスト閾値（デフォルト: 5.0）
        cr_threshold : float, optional
            宇宙線検出のシグマクリッピング閾値（デフォルト: 5）
        neighbor_threshold : float, optional
            近傍ピクセルの検出閾値（デフォルト: 5）

        Returns
        -------
        ImageCollection
            生成されたコレクション
        """
        images = [
            ImageModel.from_reader(reader, dq_flags=dq_flags)
            for reader in readers
        ]
        return cls(
            images=images,
            contrast=contrast,
            cr_threshold=cr_threshold,
            neighbor_threshold=neighbor_threshold,
        )

    def interpolate_bad_pixels(self, **kwargs) -> Self:
        """全画像の hot pixel と negative pixel を中央値補間する.

        各 ImageModel に対して interpolate_bad_pixels を呼び出し、
        補間後の画像を生成する。

        Parameters
        ----------
        **kwargs
            ImageModel.interpolate_bad_pixels に渡す追加キーワード引数

        Returns
        -------
        ImageCollection
            補間済み画像を持つ新しいコレクション
        """
        images = [image.interpolate_bad_pixels(**kwargs) for image in self.images]
        return replace(self, images=images)

    def remove_cosmic_ray(self, **kwargs) -> Self:
        """全画像から LA-Cosmic で宇宙線を一括除去する.

        コレクションが保持する LA-Cosmic パラメータを使用して、
        各 ImageModel に対して宇宙線の除去を行う。
        宇宙線マスクは各 ImageModel の .cr_mask 属性に格納される。

        Parameters
        ----------
        **kwargs
            lacosmic.remove_cosmics に渡す追加キーワード引数

        Returns
        -------
        ImageCollection
            宇宙線除去済み画像を持つ新しいコレクション
        """
        images = [
            image.remove_cosmic_ray(
                contrast=self.contrast,
                cr_threshold=self.cr_threshold,
                neighbor_threshold=self.neighbor_threshold,
                **kwargs,
            )
            for image in self.images
        ]
        return replace(self, images=images)

    def write_fits(
        self,
        output_suffix: str = "_lac",
        output_dir: Path | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        """全画像を FITS ファイルとして一括出力する.

        Parameters
        ----------
        output_suffix : str, optional
            出力ファイルの接尾辞（デフォルト: "_lac"）
        output_dir : Path | None, optional
            出力先ディレクトリ。None の場合は各画像の source_path を使用する
        overwrite : bool, optional
            既存ファイルの上書きを許可するか（デフォルト: False）

        Returns
        -------
        list[Path]
            出力した FITS ファイルパスのリスト
        """
        return [
            image.write_fits(
                output_suffix=output_suffix,
                output_dir=output_dir,
                overwrite=overwrite,
            )
            for image in self.images
        ]

    @staticmethod
    def save_fig(ax: np.ndarray, save_path: Path | str, title: str | None = None) -> None:
        """Axes 配列から Figure を取得し、タイトルを設定して画像を保存・クローズする."""
        fig = ax.flat[0].figure
        if title:
            fig.suptitle(title)
        fig.savefig(save_path)
        print(f"saved {save_path}")
        plt.close(fig)

    def imshow(
        self,
        ax=None,
        area: bool = False,
        x_center: int = 330,
        y_center: int = 550,
        half_width: int = 100,
        save_dir: Path | str | None = None,
        title: str | None = None,
        cmap: str = "coolwarm",
        norm: Normalize | None = AsinhNorm(),
        **kwargs,
    ) -> plt.Axes:  # pyright: ignore
        """全画像をサブプロットのグリッドで一覧表示する.

        2行3列のグリッドに各フレームを並べて表示する。

        Parameters
        ----------
        ax : np.ndarray of matplotlib.axes.Axes, optional
            描画先の Axes 配列。None の場合は 2×3 グリッドを新規作成する
        area : bool, optional
            True の場合、表示範囲を中心座標周辺に制限する（デフォルト: False）
        x_center : int, optional
            表示範囲の中心 x 座標（デフォルト: 330）
        y_center : int, optional
            表示範囲の中心 y 座標（デフォルト: 550）
        half_width : int, optional
            中心からの表示半幅（デフォルト: 100）
        save_dir : Path | str, optional
            保存先ディレクトリ。指定すると `imshow.png` として保存する
        title : str, optional
            Figure 全体のタイトル
        **kwargs
            matplotlib.axes.Axes.imshow に渡す追加キーワード引数

        Returns
        -------
        np.ndarray of matplotlib.axes.Axes
            描画に使用した Axes 配列
        """
        if ax is None:
            _, ax = plt.subplots(2, 3, figsize=(10, 8))
        for i, image in enumerate(self.images):
            cs = ax[i // 3, i % 3].imshow(image.sci.data, cmap=cmap, norm=norm, **kwargs)
            ax[i // 3, i % 3].set_title(image.filename)
            if area:
                ax[i // 3, i % 3].set_xlim(x_center - half_width, x_center + half_width)
                ax[i // 3, i % 3].set_ylim(y_center - half_width, y_center + half_width)
            plt.colorbar(cs)
        if save_dir:
            self.save_fig(ax, Path(save_dir) / f"{title}_imshow.png", title)
        return ax

    def imshow_mask(
        self,
        ax=None,
        mask_type: Literal["dq", "cr"] = "dq",
        area: bool = False,
        x_center: int = 330,
        y_center: int = 550,
        half_width: int = 100,
        save_dir: Path | str | None = None,
        title: str | None = None,
        **kwargs,
    ) -> plt.Axes:  # pyright: ignore
        """全画像のマスクをサブプロットのグリッドで一覧表示する.

        Parameters
        ----------
        ax : np.ndarray of matplotlib.axes.Axes, optional
            描画先の Axes 配列。None の場合は 2×3 グリッドを新規作成する
        mask_type : {"dq", "cr"}, optional
            表示するマスクの種類（デフォルト: "dq"）
        area : bool, optional
            True の場合、表示範囲を中心座標周辺に制限する（デフォルト: False）
        x_center : int, optional
            表示範囲の中心 x 座標（デフォルト: 330）
        y_center : int, optional
            表示範囲の中心 y 座標（デフォルト: 550）
        half_width : int, optional
            中心からの表示半幅（デフォルト: 100）
        save_dir : Path | str, optional
            保存先ディレクトリ。指定すると `imshow_mask_{mask_type}.png` として保存する
        title : str, optional
            Figure 全体のタイトル
        **kwargs
            matplotlib.axes.Axes.imshow に渡す追加キーワード引数

        Returns
        -------
        np.ndarray of matplotlib.axes.Axes
            描画に使用した Axes 配列
        """
        if mask_type not in ("dq", "cr"):
            raise ValueError("mask_type must be 'dq' or 'cr'")
        if ax is None:
            _, ax = plt.subplots(2, 3, figsize=(10, 8))
        for i, image in enumerate(self.images):
            if mask_type == "dq":
                # 修正: dq_mask は np.ndarray | None のため None の場合は fallback
                mask_data = (
                    image.dq_mask
                    if image.dq_mask is not None
                    else np.zeros(image.shape, dtype=bool)
                )
                title_suffix = "DQ mask"
            else:  # "cr"
                mask_data = (
                    image.cr_mask.data
                    if image.cr_mask is not None
                    else np.zeros(image.shape, dtype=bool)
                )
                title_suffix = "CR mask"
            ax[i // 3, i % 3].imshow(mask_data, cmap="gray", **kwargs)
            ax[i // 3, i % 3].set_title(f"{image.filename} ({title_suffix})")
            if area:
                ax[i // 3, i % 3].set_xlim(x_center - half_width, x_center + half_width)
                ax[i // 3, i % 3].set_ylim(y_center - half_width, y_center + half_width)
        if save_dir:
            self.save_fig(ax, Path(save_dir) / f"imshow_mask_{mask_type}.png", title)
        return ax

    def plot_spectrum_comparison(
        self,
        other: "ImageCollection",
        slit_index: int,
        labels: tuple[str, str] = ("before CR removal", "after CR removal"),
        image_source: Literal["self", "other"] = "other",
        area: bool = False,
        x_center: int = 330,
        y_center: int = 550,
        half_width: int = 100,
        imshow_kwargs: dict = {"cmap": "coolwarm", "norm": AsinhNorm()},
        save_dir: Path | str | None = None,
        title: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """2つのコレクション間でスペクトルを比較プロットする.

        Parameters
        ----------
        other : ImageCollection
            比較対象のコレクション（例: 宇宙線除去後）
        slit_index : int
            スリット方向（y 軸）のインデックス
        labels : tuple[str, str], optional
            凡例ラベル（self, other の順）
        image_source : {"self", "other"}, optional
            右側パネルに表示する画像の選択（デフォルト: "other"）
        imshow_kwargs : dict, optional
            右側パネルの ImageModel.imshow に渡すキーワード引数
        save_dir : Path | str | None, optional
            保存先ディレクトリ。指定すると `spectrum_comparison_slit{slit_index}.png` として保存する
        title : str | None, optional
            Figure 全体のタイトル
        **kwargs
            スペクトルプロットに渡す追加キーワード引数

        Returns
        -------
        np.ndarray of matplotlib.axes.Axes
            描画に使用した Axes 配列

        Raises
        ------
        ValueError
            2つのコレクションの画像数が一致しない場合
        """
        if len(self) != len(other):
            raise ValueError(
                f"画像数が一致しません: self={len(self)}, other={len(other)}"
            )

        fig, gs, spec_axes = self._create_comparison_figure()

        # --- スペクトル比較プロット ---
        for ax, img_self, img_other in zip(spec_axes.flat, self.images, other.images):
            img_self.plot_spectrum(slit_index, ax=ax, label=labels[0], **kwargs)
            img_other.plot_spectrum(slit_index, ax=ax, label=labels[1], **kwargs)
            ax.legend(fontsize="small")
        for ax in spec_axes.flat[len(self.images):]:
            ax.set_visible(False)

        # --- 右側イメージパネル ---
        ax_image = fig.add_subplot(gs[:, 3])
        source = other if image_source == "other" else self
        display_image = source.images[0]
        display_image.imshow(ax=ax_image, **(imshow_kwargs or {}))
        self._configure_image_panel(
            ax_image, display_image, slit_index, area,
            x_center, y_center, half_width,
        )

        self._add_param_text(fig)

        if save_dir:
            self.save_fig(spec_axes, Path(save_dir) / f"spectrum_comparison_slit{slit_index}.png", title)

        return spec_axes

    def _create_comparison_figure(self):
        """比較プロット用の Figure / GridSpec / Axes(2×3) を作成する."""
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1])
        spec_axes = np.empty((2, 3), dtype=object)
        for row in range(2):
            for col in range(3):
                spec_axes[row, col] = fig.add_subplot(gs[row, col])
        return fig, gs, spec_axes

    @staticmethod
    def _configure_image_panel(
        ax_image,
        display_image: "ImageModel",
        slit_index: int,
        area: bool = False,
        x_center: int = 330,
        y_center: int = 550,
        half_width: int = 100,
    ) -> None:
        """右側イメージパネルのラベル・slit 線・表示範囲を設定する."""
        ax_image.axhline(
            y=slit_index, color="red", linestyle="--", linewidth=1.5,
            label=f"slit index = {slit_index}",
        )
        ax_image.legend(fontsize="small", loc="upper right")
        ax_image.set_title(f"{display_image.filename}")
        ax_image.set_xlabel("Pixel (dispersion)")
        ax_image.set_ylabel("Pixel (spatial)")
        if area:
            ax_image.set_xlim(x_center - half_width, x_center + half_width)
            ax_image.set_ylim(y_center - half_width, y_center + half_width)

    def _add_param_text(self, fig) -> None:
        """LA-Cosmic パラメータのテキストを Figure 下部に追加しレイアウト調整する."""
        param_text = (
            f"contrast={self.contrast}  "
            f"cr_threshold={self.cr_threshold}  "
            f"neighbor_threshold={self.neighbor_threshold}"
        )
        fig.text(0.5, 0.01, param_text, ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.93))

    def plot_lacosmic_residual(
        self,
        other: "ImageCollection",
        slit_index: int,
        recession_velocity: float,
        rest_wavelength: float = oiii5007_stp,
        v_range: tuple[float, float] = (-3000.0, 3000.0),
        labels: tuple[str, str] = ("before", "after (LA-Cosmic)"),
        save_dir: Path | str | None = None,
    ) -> np.ndarray:
        """LA-Cosmic 残差を全画像で 2 段 × n_images タイル表示する.

        各列が 1 枚の画像に対応し、上段行にスペクトル比較、
        下段行に残差スペクトルを配置する。

        Parameters
        ----------
        other : ImageCollection
            LA-Cosmic 処理後のコレクション（after）
        slit_index : int
            確認するスリット行のインデックス
        recession_velocity : float
            銀河の後退速度 [km/s]
        rest_wavelength : float, optional
            速度 v=0 の基準静止波長 [Å]（デフォルト: oiii5007_stp）
        v_range : tuple[float, float], optional
            速度表示範囲 [km/s]（デフォルト: (-3000, 3000)）
        labels : tuple[str, str], optional
            凡例ラベル（before, after の順）
        save_dir : Path | str | None, optional
            保存先のディレクトリ。None の場合は保存しない。

        Returns
        -------
        np.ndarray
            Axes の 2D 配列（shape: (2 * nrows, ncols)）
        """
        n = len(self.images)
        if n == 0:
            raise ValueError("画像が存在しません。")
        if len(other.images) != n:
            raise ValueError(
                f"画像数が一致しません: self={n}, other={len(other.images)}"
            )

        ncols = min(n, 7)
        nrows = math.ceil(n / ncols)
        total_rows = 2 * nrows

        fig, axes_2d = plt.subplots(
            total_rows, ncols,
            figsize=(2.5 * ncols, 2.5 * total_rows),
            squeeze=False,
        )

        for i, (img_before, img_after) in enumerate(zip(self.images, other.images)):
            row_base = (i // ncols) * 2
            col = i % ncols
            ax_top = axes_2d[row_base, col]
            ax_bottom = axes_2d[row_base + 1, col]
            img_before.plot_lacosmic_residual(
                img_after,
                slit_index=slit_index,
                recession_velocity=recession_velocity,
                rest_wavelength=rest_wavelength,
                v_range=v_range,
                labels=labels,
                axes=(ax_top, ax_bottom),
            )

        # 余ったサブプロットを非表示
        for i in range(n, nrows * ncols):
            row_base = (i // ncols) * 2
            col = i % ncols
            axes_2d[row_base, col].set_visible(False)
            axes_2d[row_base + 1, col].set_visible(False)

        plt.tight_layout()

        if save_dir is not None:
            save_path = (
                Path(save_dir)
                / f"residual_figure(slit_index_{slit_index},recession_velocity_{recession_velocity}).png"
            )
            self.save_fig(axes_2d, save_path)
        else:
            plt.show()

        return axes_2d

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> ImageModel:
        return self.images[index]

    def __iter__(self):
        return iter(self.images)
