"""Stage 3: 3D 再構成パイプライン実行スクリプト.

IPython で `%run scripts/run_reconstruct.py` として実行する。

使用前に以下の設定値を実際の値に変更してください:
  - SLIT_POSITIONS: 各スリットの x 位置 [arcsec]（FITSヘッダーから取得しない）
  - K_VALUE       : σ_v = k · σ_z の変換係数 [km/s / arcsec]（事前フィット済み）
"""

from pathlib import Path

from stis_analysis.reconstruct.pipeline import ReconstructPipeline

# ------------------------------------------------------------------ #
# 設定（必要に応じて変更）
# ------------------------------------------------------------------ #

PROC_DIR           = Path("../data/output/proc")
OUTPUT_DIR         = Path("../data/output/recon")

RECESSION_VELOCITY = 1148.0   # NGC1068 後退速度 [km/s]

# 各スリットの x 位置 [arcsec]（6 スリット分）
SLIT_POSITIONS: list[float] = [
    # TODO: 実際のスリット位置に書き換えてください
    -0.125, -0.075, -0.025, 0.025, 0.075, 0.125,
]

# σ_v = k · σ_z の変換係数 [km/s / arcsec]
# TODO: 幾何モデルから事前にフィットした値に書き換えてください
K_VALUE: float = float("nan")

# 速度場モデル: "linear" or "power_law"
VELOCITY_FIELD_MODEL = "linear"

# x 方向補間グリッド間隔 [arcsec/pix]
PIXEL_SCALE_ARCSEC = 0.05

# 確認用画像の保存
SAVE_PICTURE = True

# ------------------------------------------------------------------ #

pipeline = ReconstructPipeline(
    slit_positions=SLIT_POSITIONS,
    k=K_VALUE,
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
print(f"sigma_v median      : {result.velocity_field.sigma_v_median:.1f} km/s")
