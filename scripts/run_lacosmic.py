"""01_lacosmic.py  ─  _flt.fits に LA-Cosmic を適用して _lac.fits を生成する.

ipython での実行:
    %run scripts/01_lacosmic.py

または:
    poetry run ipython -i scripts/01_lacosmic.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
from astropy.io import fits

from stis_analysis.core import InstrumentModel, ReaderCollection, ImageUnit
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
EXCLUDE_FILES: tuple[str, ...] = ()

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
#output_paths = lac_collection.write_fits(output_dir=OUTPUT_DIR, overwrite=True)

#print(f"\nOutput ({len(output_paths)} files):")
#for p in output_paths:
#    print(f"  {p}")

# ---　lacosmic image sample ---

from lacosmic.utils import make_cosmic_rays, make_gaussian_sources
shape = (512, 512)
data, error = make_gaussian_sources(shape, seed=0)
cr_img = make_cosmic_rays(shape, n_cosmics=200, seed=0)
data2 = data + cr_img  