"""check_lacosmic_residual.py  ─  LA-Cosmic の残差を確認する.

FLT（処理前）と LAC（LA-Cosmic 適用後）を読み込み、
差分スペクトルで輝線フラックスの損失量を可視化する。

ipython での実行:
    %run scripts/check_lacosmic_residual.py

セッション内で run_lacosmic.py を実行済みの場合は、
ファイルを再読み込みせず in-memory オブジェクトを直接使うことも可能:

    collection.plot_lacosmic_residual(
        lac_collection, slit_index=SLIT_INDEX,
        recession_velocity=RECESSION_VELOCITY,
    )
"""

from pathlib import Path

from stis_analysis.core.fits_reader import ReaderCollection
from stis_analysis.core.instrument import InstrumentModel
from stis_analysis.lacosmic import ImageCollection

# ------------------------------------------------------------------ #
# 設定（必要に応じて変更）
# ------------------------------------------------------------------ #

CRJ_DIR = Path("../../HST")           # _crj.fits があるルートディレクトリ
LAC_DIR = Path("../../output/lac")    # _lac.fits があるディレクトリ

CRJ_SUFFIX = "_crj"
CRJ_DEPTH = 1    # HST/o56502010/o56502010_crj.fits の 1 階層下

RECESSION_VELOCITY = 1148.0   # NGC1068 後退速度 [km/s]

SLIT_INDEX = 10               # 確認するスリット行
V_RANGE = (-3000.0, 3000.0)   # 速度表示範囲 [km/s]

# ------------------------------------------------------------------ #

# --- CRJ (処理前) の読み込み ---
crj_instrument = InstrumentModel.load(
    file_directory=str(CRJ_DIR),
    suffix=CRJ_SUFFIX,
    extension=".fits",
    depth=CRJ_DEPTH,
)
crj_readers = ReaderCollection.from_paths(crj_instrument.path_list)
crj_collection = ImageCollection.from_readers(crj_readers)

print(f"CRJ files ({len(crj_collection.images)}):")
for img in crj_collection.images:
    print(f"  {img.filename}")

# --- LAC (LA-Cosmic 適用後) の読み込み ---
lac_instrument = InstrumentModel.load(
    file_directory=str(LAC_DIR),
    suffix="_lac",
    extension=".fits",
    depth=0,
)
lac_readers = ReaderCollection.from_paths(lac_instrument.path_list)
lac_collection = ImageCollection.from_readers(lac_readers)

print(f"\nLAC files ({len(lac_collection.images)}):")
for img in lac_collection.images:
    cr_count = img.cr_mask.data.sum() if img.cr_mask is not None else "N/A"
    print(f"  {img.filename}  CR-flagged pixels: {cr_count}")

# ------------------------------------------------------------------ #
# 残差確認プロット
# ------------------------------------------------------------------ #

crj_collection.plot_lacosmic_residual(
    lac_collection,
    slit_index=SLIT_INDEX,
    recession_velocity=RECESSION_VELOCITY,
    v_range=V_RANGE,
)
