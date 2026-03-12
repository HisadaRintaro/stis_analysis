"""01_lacosmic.py  ─  _flt.fits に LA-Cosmic を適用して _lac.fits を生成する.

ipython での実行:
    %run scripts/01_lacosmic.py

または:
    poetry run ipython -i scripts/01_lacosmic.py
"""

from matplotlib.colors import AsinhNorm
from pathlib import Path
import matplotlib.pyplot as plt

from stis_analysis.core import InstrumentModel, ReaderCollection
from stis_analysis.lacosmic import ImageCollection

# ------------------------------------------------------------------ #
# 設定（必要に応じて変更）
# ------------------------------------------------------------------ #

HST_DIR = Path("../data/HST")            # _flt.fits があるルートディレクトリ
OUTPUT_DIR = Path("../data/output/lac")  # _lac.fits の出力先

SUFFIX = "_crj"
DEPTH = 1  # HST/o56502010/o56502010_crj.fits の 1 階層下

DQ_FLAGS: int = 16

LA_COSMIC_PARAMS = dict(
    contrast=5.0,
    cr_threshold=5.0,
    neighbor_threshold=5.0,
)

# 除外したいファイルの stem を列挙（不要なら空のまま）
EXCLUDE_FILES: tuple[str, ...] = ('o56503010_crj',)

# --- NGC1068 ---

RECESSION_VELOCITY = 1148.0
INDEX = 572

# ------------------------------------------------------------------ #

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- ファイル探索 ---
instrument = InstrumentModel.load(
    file_directory=str(HST_DIR),
    suffix=SUFFIX,
    extension=".fits",
    depth=DEPTH,
    exclude_files=EXCLUDE_FILES,
)
print(f"Found {len(instrument.path_list)} files:")
for p in instrument.path_list:
    print(f"  {p}")

# --- 読み込み ---
readers = ReaderCollection.from_paths(instrument.path_list)
collection = ImageCollection.from_readers(readers, dq_flags=DQ_FLAGS, **LA_COSMIC_PARAMS)

# --- LA-Cosmic 適用 ---
lac_collection = collection.remove_cosmic_ray(maxiter=1)

# --- _lac.fits として書き出し ---
output_paths = lac_collection.write_fits(output_dir=OUTPUT_DIR, overwrite=True)

print(f"\nOutput ({len(output_paths)} files):")
for p in output_paths:
    print(f"  {p}")

# --- 画像ファイル出力 ---
# imshow.png             : CRJ 全画像一覧（処理前）
# imshow_mask_dq.png     : DQ マスク一覧
# imshow_mask_cr.png     : LA-Cosmic 検出マスク一覧
# spectrum_comparison_slit{INDEX}.png : スペクトル比較（処理前後）
# residual_figure(...)   : LA-Cosmic 残差プロット

lac_collection.imshow(save_dir=OUTPUT_DIR, norm=AsinhNorm(), cmap="coolwarm")
lac_collection.imshow_mask(mask_type="dq", save_dir=OUTPUT_DIR)
lac_collection.imshow_mask(mask_type="cr", save_dir=OUTPUT_DIR)
collection.plot_spectrum_comparison(lac_collection, INDEX, save_dir=OUTPUT_DIR)
collection.plot_lacosmic_residual(lac_collection, INDEX, RECESSION_VELOCITY, save_dir=OUTPUT_DIR)

# ---　lacosmic image sample ---

#from lacosmic.utils import make_cosmic_rays, make_gaussian_sources
#shape = (512, 512)
#data, error = make_gaussian_sources(shape, seed=0)
#cr_img = make_cosmic_rays(shape, n_cosmics=200, seed=0)
#data2 = data + cr_img  