"""02_processing.py  ─  _lac.fits の各処理ステップを手動で実行して確認する.

ProcessingPipeline を使わず各ステップを個別に実行することで、
ステップごとにスペクトルを確認しながら処理を進めることができる。

処理順:
  Step 1. ファイル探索
  Step 2. stistools.x2d による 2D 幾何補正
  Step 3. ProcessingImageCollection として読み込み
  Step 4. 連続光差し引き  ← plot_continuum_fit() で確認
  Step 5. OIII λ4959 除去
  Step 6. velocity range clipping
  Step 7. _proc.fits として書き出し
  Step 8. 処理前後の比較プロット

ipython での実行:
    %run scripts/run_processing.py

または:
    poetry run ipython -i scripts/run_processing.py
"""

from pathlib import Path

from stis_analysis.core.fits_reader import ReaderCollection
from stis_analysis.core.instrument import InstrumentModel
from stis_analysis.processing.image import ProcessingImageCollection
from stis_analysis.processing.pipeline import ProcessingPipeline

# ------------------------------------------------------------------ #
# 設定（必要に応じて変更）
# ------------------------------------------------------------------ #

LAC_DIR = Path("../data/output/lac")     # _lac.fits があるディレクトリ
OUTPUT_DIR = Path("../data/output/proc") # _proc.fits の出力先

SUFFIX = "_lac"
DEPTH = 0   # lac/ 直下にファイルがある場合は 0

DQ_FLAGS: int = 16

RECESSION_VELOCITY = 1148.0         # NGC1068 後退速度 [km/s]

# OIII λ5007 観測波長を v=0 とした相対速度 [km/s] でウィンドウを指定
# → 輝線（λ4959, λ5007）が重ならない領域を選ぶ
CONTINUUM_WINDOWS_KMS = [
    (-4000.0, -3200.0),
    (3000.0, 4000.0),
]
CONTINUUM_DEGREE = 1

# OIII λ4959 除去パラメータ
O3_SCALE: float = 1.0 / 2.98   # OIII λ5007/λ4959 強度比
O3_HALF_WIDTH_AA: float = 30.0

# velocity clipping の範囲 [km/s]
V_MIN = -2500.0
V_MAX = 2500.0

# x2d（2D 幾何補正）を実行するか
# _lac.fits を直接渡す場合は True、すでに x2d 済みなら False
RUN_X2D = True

# 確認用スリット行インデックス
SLIT_INDEX = 10

# ------------------------------------------------------------------ #

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# Step 1. ファイル探索
# ------------------------------------------------------------------ #

instrument = InstrumentModel.load(
    file_directory=str(LAC_DIR),
    suffix=SUFFIX,
    extension=".fits",
    depth=DEPTH,
)
lac_paths = instrument.path_list
print(f"Found {len(lac_paths)} files:")
for p in lac_paths:
    print(f"  {p}")

# ------------------------------------------------------------------ #
# Step 2. stistools.x2d による 2D 幾何補正
# ------------------------------------------------------------------ #

if RUN_X2D:
    ProcessingPipeline._check_reference_files(lac_paths)
    x2d_paths = ProcessingPipeline._run_x2d_batch(lac_paths, OUTPUT_DIR)
    print(f"\nStep 2: x2d 完了 ({len(x2d_paths)} files)")
else:
    x2d_paths = lac_paths
    print("\nStep 2: x2d スキップ（RUN_X2D=False）")

# ------------------------------------------------------------------ #
# Step 3. ProcessingImageCollection として読み込み
# ------------------------------------------------------------------ #

readers = ReaderCollection.from_paths(x2d_paths)
collection = ProcessingImageCollection.from_readers(readers, dq_flags=DQ_FLAGS)
print(f"\nStep 3: 読み込み完了 ({len(collection.images)} images)")

# ------------------------------------------------------------------ #
# Step 4. 連続光差し引き
# ------------------------------------------------------------------ #

continuum_subtracted = collection.subtract_continuum(
    continuum_windows_kms=CONTINUUM_WINDOWS_KMS,
    recession_velocity=RECESSION_VELOCITY,
    degree=CONTINUUM_DEGREE,
)
print("\nStep 4: 連続光差し引き完了")

# 確認プロット: 連続光フィット（差し引き前）
for image in collection.images:
    image.plot_continuum_fit(
        slit_index=SLIT_INDEX,
        continuum_windows_kms=CONTINUUM_WINDOWS_KMS,
        recession_velocity=RECESSION_VELOCITY,
        degree=CONTINUUM_DEGREE,
    )

# ------------------------------------------------------------------ #
# Step 5. OIII λ4959 除去
# ------------------------------------------------------------------ #

o3_removed = continuum_subtracted.remove_o3_4959(
    recession_velocity=RECESSION_VELOCITY,
    scale=O3_SCALE,
    half_width_aa=O3_HALF_WIDTH_AA,
)
print("\nStep 5: OIII λ4959 除去完了")

# ------------------------------------------------------------------ #
# Step 6. velocity range clipping
# ------------------------------------------------------------------ #

clipped = o3_removed.clip_velocity_range(
    v_min=V_MIN,
    v_max=V_MAX,
    recession_velocity=RECESSION_VELOCITY,
)
print(f"\nStep 6: velocity clipping 完了 ({V_MIN} ~ {V_MAX} km/s)")

# ------------------------------------------------------------------ #
# Step 7. _proc.fits として書き出し
# ------------------------------------------------------------------ #

#output_paths = clipped.write_fits(
    #output_suffix="_proc",
    #output_dir=OUTPUT_DIR,
    #overwrite=False,
#)
#print(f"\nStep 7: 書き出し完了 ({len(output_paths)} files):")
#for p in output_paths:
    #print(f"  {p}")

# ------------------------------------------------------------------ #
# Step 8. 処理前後の比較プロット
# ------------------------------------------------------------------ #

print("\nStep 8: 確認用プロット")

# 連続光フィット確認（差し引き前スペクトルとフィット曲線）
for image in collection.images:
    image.plot_continuum_fit(
        slit_index=SLIT_INDEX,
        continuum_windows_kms=CONTINUUM_WINDOWS_KMS,
        recession_velocity=RECESSION_VELOCITY,
        degree=CONTINUUM_DEGREE,
        o3_half_width_aa=O3_HALF_WIDTH_AA,
    )
