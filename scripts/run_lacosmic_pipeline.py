"""01_lacosmic_pipeline.py  ─  LaCosmicPipeline で _crj.fits に LA-Cosmic を適用して _lac.fits を生成する.

ipython での実行:
    %run scripts/01_lacosmic_pipeline.py

または:
    poetry run ipython -i scripts/01_lacosmic_pipeline.py
"""

from pathlib import Path

from stis_analysis.lacosmic import LaCosmicPipeline

# ------------------------------------------------------------------ #
# 設定（必要に応じて変更）
# ------------------------------------------------------------------ #

HST_DIR = Path("../data/HST")            # _crj.fits があるルートディレクトリ
OUTPUT_DIR = Path("../data/output/lac")  # _lac.fits の出力先（既存ファイルがあれば lac-2 等に退避）

SUFFIX = "_crj"
DEPTH = 1  # HST/o56502010/o56502010_crj.fits の 1 階層下

DQ_FLAGS: int = 16

# 除外したいファイルの stem を列挙（不要なら空のまま）
EXCLUDE_FILES: tuple[str, ...] = ('o56503010_crj',)

# --- NGC1068 ---
RECESSION_VELOCITY = 1148.0
SLIT_INDEX = 572

# ------------------------------------------------------------------ #

pipeline = LaCosmicPipeline(
    contrast=5.0,
    cr_threshold=5.0,
    neighbor_threshold=5.0,
    maxiter=1,
    dq_flags=DQ_FLAGS,
    suffix=SUFFIX,
    depth=DEPTH,
    exclude_files=EXCLUDE_FILES,
)

result = pipeline.run(
    input_dir=HST_DIR,
    output_dir=OUTPUT_DIR,
    save=True,
    slit_index=SLIT_INDEX,
    recession_velocity=RECESSION_VELOCITY,
)

print(f"\nOutput ({len(result.output_paths)} files):")
for p in result.output_paths:
    print(f"  {p}")
