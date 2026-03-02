"""stis_analysis.processing - HST/STIS スペクトル処理サブパッケージ.

_lac.fits (LA-Cosmic 済み) に対して以下を順番に適用する:
  1. stistools.x2d による 2D 幾何補正
  2. 連続光差し引き
  3. OIII λ4959 除去
  4. velocity range clipping
"""

from stis_analysis.processing.image import ProcessingImageModel, ProcessingImageCollection
from stis_analysis.processing.pipeline import ProcessingPipeline, ProcessingResult

__all__ = [
    "ProcessingImageModel",
    "ProcessingImageCollection",
    "ProcessingPipeline",
    "ProcessingResult",
]
