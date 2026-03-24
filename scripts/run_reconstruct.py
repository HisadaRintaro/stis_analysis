"""03_reconstruct.py  ─  _proc.fits の各処理ステップを手動で実行して確認する.

ReconstructPipeline を使わず各ステップを個別に実行することで、
ステップごとに DataCube の状態を確認しながら 3D 再構成を進めることができる。

処理順:
  Step 1. ファイル探索
  Step 2. raw cube 構築
  Step 3. x 方向補間（interpolated cube）
  Step 4. σ_v・σ_z の計算
  Step 5. VelocityField 構築（k を自動計算）
  Step 6. 3D 再構成（reconstructed cube）

ipython での実行:
    %run scripts/run_reconstruct.py

または:
    poetry run ipython -i scripts/run_reconstruct.py
"""

from pathlib import Path

from stis_analysis.core.instrument import InstrumentModel
from stis_analysis.reconstruct.cube import DataCube
from stis_analysis.reconstruct.velocity_field import LinearVelocityField

# ------------------------------------------------------------------ #
# 設定（必要に応じて変更）
# ------------------------------------------------------------------ #

PROC_DIR   = Path("../data/output/proc")   # _proc.fits があるディレクトリ
OUTPUT_DIR = Path("../data/output/recon")  # 出力先

SUFFIX = "_proc"
DEPTH  = 0

RECESSION_VELOCITY = 1148.0   # NGC1068 後退速度 [km/s]

# 各スリットの x 位置 [arcsec]（ファイル順に対応）
SLIT_POSITIONS: list[float] = [
    # TODO: 実際のスリット位置に書き換えてください
    -0.125, -0.075, -0.025, 0.025, 0.075, 0.125,
]

# x 方向補間グリッド間隔 [arcsec/pix]
PIXEL_SCALE_ARCSEC = 0.05

# 速度場モデル
VELOCITY_FIELD_MODEL = "linear"   # "linear" or "power_law"

# ------------------------------------------------------------------ #

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# Step 1. ファイル探索
# ------------------------------------------------------------------ #

instrument = InstrumentModel.load(
    file_directory=str(PROC_DIR),
    suffix=SUFFIX,
    extension=".fits",
    depth=DEPTH,
)
proc_paths = instrument.path_list
print(f"Step 1: Found {len(proc_paths)} files:")
for p in proc_paths:
    print(f"  {p}")

# ------------------------------------------------------------------ #
# Step 2. raw cube 構築
# ------------------------------------------------------------------ #

raw_cube = DataCube.from_proc_files(
    paths=proc_paths,
    slit_positions=SLIT_POSITIONS,
    recession_velocity=RECESSION_VELOCITY,
)
print(f"\nStep 2: raw cube 構築完了")
print(f"  shape        : {raw_cube.data.shape}  (n_slit, n_y, n_v)")
print(f"  velocity_array: [{raw_cube.velocity_array[0]:.1f}, ..., {raw_cube.velocity_array[-1]:.1f}] km/s")

# ------------------------------------------------------------------ #
# Step 3. x 方向補間（interpolated cube）
# ------------------------------------------------------------------ #

interp_cube = raw_cube.interpolate(pixel_scale_arcsec=PIXEL_SCALE_ARCSEC)
print(f"\nStep 3: interpolate 完了")
print(f"  shape  : {interp_cube.data.shape}  (n_x, n_y, n_v)")
assert interp_cube.x_array is not None
print(f"  x_array: [{interp_cube.x_array[0]:.3f}, ..., {interp_cube.x_array[-1]:.3f}] arcsec")

# ------------------------------------------------------------------ #
# Step 4. σ_v・σ_z の計算
# ------------------------------------------------------------------ #

v_mean, sigma_v = interp_cube.sigma_v
sigma_z = interp_cube.sigma_z
_, sigma_x = interp_cube.sigma_x
_, sigma_y = interp_cube.sigma_y

print(f"\nStep 4: σ 計算完了")
print(f"  v_mean  : {v_mean:.2f} km/s")
print(f"  sigma_v : {sigma_v:.2f} km/s")
print(f"  sigma_x : {sigma_x:.4f} arcsec")
print(f"  sigma_y : {sigma_y:.4f} arcsec")
print(f"  sigma_z : {sigma_z:.4f} arcsec  (= sqrt(0.5*(σx²+σy²)))")

# ------------------------------------------------------------------ #
# Step 5. VelocityField 構築（k を自動計算）
# ------------------------------------------------------------------ #

if VELOCITY_FIELD_MODEL == "linear":
    velocity_field = LinearVelocityField().with_k_from_sigmas(sigma_v, sigma_z)
else:
    from stis_analysis.reconstruct.velocity_field import PowerLawVelocityField
    velocity_field = PowerLawVelocityField().with_k_from_sigmas(sigma_v, sigma_z)

print(f"\nStep 5: VelocityField 構築完了")
print(f"  model   : {VELOCITY_FIELD_MODEL}")
print(f"  k       : {velocity_field.k:.4f} km/s/arcsec")

# ------------------------------------------------------------------ #
# Step 6. 3D 再構成（reconstructed cube）
# ------------------------------------------------------------------ #

recon_cube = interp_cube.reconstruct(velocity_field)
print(f"\nStep 6: reconstruct 完了")
print(f"  shape   : {recon_cube.data.shape}  (n_x, n_y, n_z)")
assert recon_cube.z_array is not None
print(f"  z_array : [{recon_cube.z_array[0]:.3f}, ..., {recon_cube.z_array[-1]:.3f}] arcsec")
