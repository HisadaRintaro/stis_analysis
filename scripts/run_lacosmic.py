"""01_lacosmic.py  ─  _flt.fits に LA-Cosmic を適用して _lac.fits を生成する.

ipython での実行:
    %run scripts/01_lacosmic.py

または:
    poetry run ipython -i scripts/01_lacosmic.py
"""

from pathlib import Path

from stis_analysis.core.fits_reader import ReaderCollection
from stis_analysis.core.instrument import InstrumentModel
from stis_analysis.lacosmic import ImageCollection

# ------------------------------------------------------------------ #
# 設定（必要に応じて変更）
# ------------------------------------------------------------------ #

HST_DIR = Path("../HST")            # _flt.fits があるルートディレクトリ
OUTPUT_DIR = Path("../output/lac")  # _lac.fits の出力先

SUFFIX = "_flt"
DEPTH = 1  # HST/o56502010/o56502010_flt.fits の 1 階層下

LA_COSMIC_PARAMS = dict(
    contrast=5.0,
    cr_threshold=5.0,
    neighbor_threshold=5.0,
    error=5.0,
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
collection = ImageCollection.from_readers(readers, **LA_COSMIC_PARAMS)

# --- LA-Cosmic 適用 ---
lac_collection = collection.remove_cosmic_ray()

# --- _lac.fits として書き出し ---
output_paths = lac_collection.write_fits(output_dir=OUTPUT_DIR, overwrite=True)

print(f"\nOutput ({len(output_paths)} files):")
for p in output_paths:
    print(f"  {p}")
