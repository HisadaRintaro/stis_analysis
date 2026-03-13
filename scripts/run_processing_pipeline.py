"""02_processing_pipeline.py  ─  ProcessingPipeline で _lac.fits を一括処理して _proc.fits を生成する.

処理順:
  1. stistools.x2d による 2D 幾何補正（run_x2d=True の場合）
  2. 連続光差し引き
  3. OIII λ4959 除去
  4. velocity range clipping

各ステップを手動で確認したい場合は run_processing.py を使用してください。

ipython での実行:
    %run scripts/run_processing_pipeline.py

または:
    poetry run ipython -i scripts/run_processing_pipeline.py
"""

from pathlib import Path

from stis_analysis.processing import ProcessingPipeline

# ------------------------------------------------------------------ #
# 設定（必要に応じて変更）
# ------------------------------------------------------------------ #

LAC_DIR = Path("../data/output/lac")     # _lac.fits があるディレクトリ
OUTPUT_DIR = Path("../data/output/proc") # _proc.fits の出力先

RECESSION_VELOCITY = 1148.0         # NGC1068 後退速度 [km/s]

# OIII λ5007 観測波長を v=0 とした相対速度 [km/s] でウィンドウを指定
# → 輝線（λ4959, λ5007）が重ならない領域を選ぶ
CONTINUUM_WINDOWS_KMS = [
    (-4000.0, -3200.0),
    (3000.0, 4000.0),
]

# velocity clipping の範囲 [km/s]
V_MIN = -2500.0
V_MAX = 2500.0

# x2d（2D 幾何補正）を実行するか
# _flt.fits を直接渡す場合は True、すでに x2d 済みなら False
RUN_X2D = True

# ------------------------------------------------------------------ #

pipeline = ProcessingPipeline(
    continuum_windows_kms=CONTINUUM_WINDOWS_KMS,
    continuum_degree=1,
    recession_velocity=RECESSION_VELOCITY,
    v_min=V_MIN,
    v_max=V_MAX,
    suffix="_lac",
    depth=0,
)

result = pipeline.run(
    input_dir=LAC_DIR,
    output_dir=OUTPUT_DIR,
    run_x2d=RUN_X2D,
    overwrite=True,
)

print(f"Output ({len(result.output_paths)} files):")
for p in result.output_paths:
    print(f"  {p}")

# ------------------------------------------------------------------ #
# 確認用プロット（必要に応じて slit_index を変更）
# ------------------------------------------------------------------ #

SLIT_INDEX = 10  # 空間方向の確認スリット行

result.plot_continuum_fit(
    slit_index=SLIT_INDEX,
    continuum_windows_kms=CONTINUUM_WINDOWS_KMS,
    recession_velocity=RECESSION_VELOCITY,
)
