"""03_reconstruct_pipeline.py  ─  ReconstructPipeline で _proc.fits を一括処理して 3D 再構成を行う.

処理順:
  1. InstrumentModel でファイル探索
  2. DataCube.from_proc_files() — raw cube 構築
  3. DataCube.interpolate()     — x 方向補間
  4. DataCube.sigma_v / sigma_z — σ 値を計算（球対称仮定）
  5. VelocityField.with_k_from_sigmas() — k を自動計算
  6. DataCube.reconstruct()     — 3D 再構成

各ステップを手動で確認したい場合は run_reconstruct.py を使用してください。

ipython での実行:
    %run scripts/run_reconstruct_pipeline.py

または:
    poetry run ipython -i scripts/run_reconstruct_pipeline.py
"""

from pathlib import Path

from stis_analysis.reconstruct.pipeline import ReconstructPipeline

# ------------------------------------------------------------------ #
# 設定（必要に応じて変更）
# ------------------------------------------------------------------ #

PROC_DIR   = Path("../data/output/proc")   # _proc.fits があるディレクトリ
OUTPUT_DIR = Path("../data/output/recon")  # 出力先

RECESSION_VELOCITY = 1148.0   # NGC1068 後退速度 [km/s]

# 各スリットの x 位置 [arcsec]（ファイル順に対応）
SLIT_POSITIONS: list[float] = [
    # TODO: 実際のスリット位置に書き換えてください
    -0.125, -0.075, -0.025, 0.025, 0.075, 0.125,
]

# 速度場モデル: "linear" or "power_law"
VELOCITY_FIELD_MODEL = "linear"

# x 方向補間グリッド間隔 [arcsec/pix]
PIXEL_SCALE_ARCSEC = 0.05

# 確認用画像の保存（可視化メソッド実装後に有効）
SAVE_PICTURE = False

# ------------------------------------------------------------------ #

pipeline = ReconstructPipeline(
    slit_positions=SLIT_POSITIONS,
    velocity_field_model=VELOCITY_FIELD_MODEL,
    recession_velocity=RECESSION_VELOCITY,
    pixel_scale_arcsec=PIXEL_SCALE_ARCSEC,
    suffix="_proc",
    depth=0,
)

result = pipeline.run(
    input_dir=PROC_DIR,
    output_dir=OUTPUT_DIR,
    save_picture=SAVE_PICTURE,
)

print(f"\nraw cube shape      : {result.raw_cube.data.shape}")
print(f"interpolated shape  : {result.interpolated_cube.data.shape}")
print(f"reconstructed shape : {result.reconstructed_cube.data.shape}")
print(f"velocity field k    : {result.velocity_field.k:.4f} km/s/arcsec")
