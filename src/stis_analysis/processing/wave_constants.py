# 後方互換性のため stis_analysis.core.wave_constants を再エクスポートする。
# 新規コードは stis_analysis.core.wave_constants から直接インポートすること。
from stis_analysis.core.wave_constants import *  # noqa: F401, F403
