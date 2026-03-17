"""stis_analysis.reconstruct — Stage 3: 3D 再構成サブパッケージ.

_proc.fits (×6) から 3D スペクトルキューブを構築し、速度場推定・3D 再構成を行う:
  1. DataCube.from_proc_files() — 6スリットを読み込み raw cube を構築
  2. DataCube.interpolate()     — x方向を等間隔グリッドに補間
  3. DataCube.compute_sigma_v() — フラックス加重速度分散 σ_v マップを計算
  4. VelocityField.with_k(k)    — 変換係数 k を設定
  5. DataCube.reconstruct(vf)   — velocity → depth 軸変換で 3D 再構成
"""

from stis_analysis.reconstruct.cube import DataCube
from stis_analysis.reconstruct.velocity_field import (
    VelocityField,
    LinearVelocityField,
    PowerLawVelocityField,
)
from stis_analysis.reconstruct.pipeline import ReconstructPipeline, ReconstructResult

__all__ = [
    "DataCube",
    "VelocityField",
    "LinearVelocityField",
    "PowerLawVelocityField",
    "ReconstructPipeline",
    "ReconstructResult",
]
