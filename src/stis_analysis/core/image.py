"""共通画像ユニット.

FITS HDU の data と header のペアを型安全に管理する ImageUnit を提供する。

ImageUnit は spectrum_package と stis_la_cosmic の両実装をマージしたもの:

- naxis1 / naxis2 プロパティ（spectrum_package 由来）
- wavelength プロパティ：CRVAL1/CDELT1 から波長配列を生成（stis_la_cosmic 由来）
- to_hdu()：bool 型 data を uint8 に変換して ImageHDU を返す（stis_la_cosmic 由来）
- __repr__（stis_la_cosmic 由来）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from astropy.io import fits  # type: ignore


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
    def wavelength(self) -> np.ndarray | None:
        """ヘッダーの WCS キーワードから波長配列を生成する.

        CRVAL1（参照ピクセルの波長）と CDELT1 または CD1_1（波長/pixel）から
        各ピクセルの波長を計算する。必要なキーワードが存在しない場合は None を返す。

        Returns
        -------
        np.ndarray | None
            波長配列 [Å]。WCS 情報が不足している場合は None
        """
        crval1 = cast(float | None, self.header.get("CRVAL1"))
        cdelt1 = cast(float | None, self.header.get("CDELT1", self.header.get("CD1_1")))
        if crval1 is None or cdelt1 is None:
            return None
        crpix1 = cast(float, self.header.get("CRPIX1", 1.0))
        n_pixels = self.data.shape[1]
        return crval1 + cdelt1 * (np.arange(n_pixels) - (crpix1 - 1))

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
