"""共通画像ユニット.

FITS HDU の data と header のペアを型安全に管理する ImageUnit を提供する。

ImageUnit は spectrum_package と stis_la_cosmic の両実装をマージしたもの:

- naxis1 / naxis2 プロパティ（spectrum_package 由来）
- wavelength プロパティ：CRVAL1/CDELT1 から波長配列を生成（stis_la_cosmic 由来）
- to_hdu()：bool 型 data を uint8 に変換して ImageHDU を返す（stis_la_cosmic 由来）
- __repr__（stis_la_cosmic 由来）
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from astropy.io import fits  # type: ignore
from scipy.sparse.csgraph import laplacian
from .wave_constants import c_kms



@dataclass(frozen=True)
class ImageUnit:
    """data と header のペア（astropy.io.fits.ImageHDU の型安全なラッパー）.

    Attributes
    ----------
    data : np.ndarray
        画像データ配列
    header : fits.Header
        対応する FITS ヘッダー
    """

    data: np.ndarray
    header: fits.Header

    def __repr__(self) -> str:
        return (
            f"ImageUnit(data={self.data.shape}, "
            f"header={'fits.Header' if self.header else None})"
        )

    @property
    def naxis1(self) -> int:
        """列数（NAXIS1）."""
        return int(self.header.get("NAXIS1", 0))  # type: ignore[arg-type]

    @property
    def naxis2(self) -> int:
        """行数（NAXIS2）."""
        return int(self.header.get("NAXIS2", 0))  # type: ignore[arg-type]

    @property
    def cdelt1(self) -> float:
        """波長/pixel（CDELT1 または CD1_1）."""
        return cast(float, self.header.get("CDELT1", self.header.get("CD1_1")))

    @property
    def crval1(self) -> float:
        """参照波長（CRVAL1）."""
        return cast(float, self.header.get("CRVAL1"))

    @property
    def crpix1(self) -> float:
        """参照ピクセル（CRPIX1）."""
        return cast(float, self.header.get("CRPIX1", 1.0))

    @property
    def crval2(self) -> float:
        """参照位置（CRVAL2）[arcsec]."""
        return cast(float, self.header.get("CRVAL2", 0.0))

    @property
    def cdelt2(self) -> float:
        """空間スケール/pixel（CDELT2 または CD2_2）[arcsec/pix]."""
        return cast(float, self.header.get("CDELT2", self.header.get("CD2_2", 1.0)))

    @property
    def crpix2(self) -> float:
        """参照ピクセル（CRPIX2）."""
        return cast(float, self.header.get("CRPIX2", 1.0))

    @property
    def spatial_array(self) -> np.ndarray:
        """ヘッダーの WCS キーワードから空間 y 軸配列を生成する.

        CRVAL2（参照ピクセルの位置）と CDELT2（arcsec/pix）から
        各ピクセルの空間位置を計算する。

        Returns
        -------
        np.ndarray
            空間位置配列 [arcsec]。shape: (naxis2,)
        """
        return self.crval2 + self.cdelt2 * (np.arange(self.naxis2) - (self.crpix2 - 1))
    
    @property
    def unit(self) -> str:
        """単位（BUNIT）."""
        return cast(str, self.header.get("BUNIT", "counts"))

    @property
    def wavelength(self) -> np.ndarray:
        """ヘッダーの WCS キーワードから波長配列を生成する.

        CRVAL1（参照ピクセルの波長）と CDELT1 または CD1_1（波長/pixel）から
        各ピクセルの波長を計算する。必要なキーワードが存在しない場合は None を返す。

        Returns
        -------
        np.ndarray
            波長配列 [Å]。WCS 情報が不足している場合は ValueError を投げる。
        """
        crval1 = self.crval1
        cdelt1 = self.cdelt1
        if crval1 is None or cdelt1 is None:
            raise ValueError("SCI ヘッダーに WCS 情報（CRVAL1/CDELT1）がありません。")
        crpix1 = self.crpix1
        n_pixels = self.naxis1
        return crval1 + cdelt1 * (np.arange(n_pixels) - (crpix1 - 1))

    def velocity_array(
        self,
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
        return c_kms * (self.wavelength / lambda_ref - 1.0)

    def to_hdu(self) -> fits.ImageHDU:
        """fits.ImageHDU に変換する.

        bool 型の data は FITS で扱えないため uint8 に変換する。

        Returns
        -------
        fits.ImageHDU
            data と header を格納した ImageHDU
        """
        data = self.data.astype(np.uint8) if self.data.dtype == bool else self.data
        return fits.ImageHDU(data=data, header=self.header)

    def plot_spectrum(self, slit_index: int, ax=None, **kwargs):
        """指定スリット行のスペクトルを波長軸でプロットする.

        Parameters
        ----------
        slit_index : int
            描画するスリット行のインデックス（空間方向）
        ax : matplotlib.axes.Axes, optional
            描画先 Axes。None の場合は新規作成。
        **kwargs
            ax.plot() に渡す追加キーワード引数

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        wavelength = self.wavelength
        ax.plot(wavelength, self.data[slit_index, :], **kwargs)
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel(self.unit)
        return ax

    def imshow(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        
        wavelength = self.wavelength
        if type(wavelength) != type(None):
            wavelength = cast(np.ndarray, wavelength)
            extent = (wavelength[0], wavelength[-1], 0, self.naxis1)
        else:
            extent = None
        cs = ax.imshow(self.data, **kwargs, extent = extent)
        plt.colorbar(cs, ax=ax)
        ax.set_xlabel(r'wavelength [$\AA$]')
        ax.set_ylabel('spatial [pixel]')
        return ax