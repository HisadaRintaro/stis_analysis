# CLAUDE.md

## プロジェクト概要

HST/STIS スリット分光データから銀河の3次元構造（x, y, z）を再構成する Python パッケージ。
複数スリット位置で取得した 2D スペクトル画像（x-λ）を積み重ねて 3D キューブ（x-y-v）を作り、
速度場モデルを用いて速度軸を深度軸（z）に変換することで立体構造を得る。

## パッケージ構成

```
src/stis_analysis/
  core/          基盤クラス（ImageUnit, InstrumentModel, FitsReader）
  lacosmic/      宇宙線除去（L.A.Cosmic ラッパー）
  processing/    STIS パイプライン処理（calstis ラッパー）
  reconstruct/   3D 再構成（DataCube, VelocityField, ReconstructPipeline）

scripts/
  run_lacosmic.py              宇宙線除去（ステップ確認版）
  run_lacosmic_pipeline.py     宇宙線除去（ワンショット版）
  run_processing.py            STIS 処理（ステップ確認版）
  run_processing_pipeline.py   STIS 処理（ワンショット版）
  run_reconstruct.py           3D 再構成（ステップ確認版）
  run_reconstruct_pipeline.py  3D 再構成（ワンショット版）
  check_lacosmic_residual.py   LA-Cosmic 残差確認
  convolve2d_reference.py      convolve2d 参考実装

tests/
  test_core/ test_lacosmic/ test_processing/ test_reconstruct/
```

## 設計原則

### 不変オブジェクト + ステージ管理
- `@dataclass(frozen=True)` + `dataclasses.replace()` でステージ遷移を表現する
- 破壊的変更は行わず、常に新オブジェクトを返す

### DataCube のステージ
| ステージ | 条件 | 説明 |
|---|---|---|
| raw | `x_positions is not None and x_array is None` | from_proc_files() 直後 |
| interpolated | `x_array is not None and z_array is None` | interpolate() 後 |
| reconstructed | `z_array is not None` | reconstruct() 後 |

データ shape は常に `(n_x, n_y, n_v)` — ステージ間で軸の意味は変わらない。

### DataCube のフィールド（主要）

| フィールド | 型 | 説明 |
|---|---|---|
| `data` | `np.ndarray` | スペクトルキューブ |
| `velocity_array` | `np.ndarray` | 速度軸 [km/s] |
| `x_positions` | `np.ndarray \| None` | raw: スリット x 位置 [arcsec] |
| `x_array` | `np.ndarray \| None` | interpolated: 等間隔 x 軸 [arcsec] |
| `y_array` | `np.ndarray \| None` | 空間 y 軸 [arcsec]（全ステージ） |
| `z_array` | `np.ndarray \| None` | reconstructed: 深度軸 [arcsec] |
| `source_paths` | `tuple[Path,...] \| None` | 読み込み元 FITS パス |
| `reader_collection` | `ReaderCollection \| None` | FITS ヘッダー確認用（repr=False） |
| `image_collection` | `ImageCollection \| None` | SCI データ確認用（repr=False） |

`reader_collection` / `image_collection` は `from_proc_files()` で設定され、`replace()` で各ステージに引き継がれる。

### DataCube のメソッド
| メソッド | ステージ | 説明 |
|---|---|---|
| `from_proc_files()` | → raw | _proc.fits から構築 |
| `interpolate()` | raw → interpolated | x 方向を等間隔補間 |
| `reconstruct(vf)` | interpolated → reconstructed | v→z 変換 |
| `trim_y(y_min, y_max)` | 全ステージ | y 軸を arcsec 単位でトリミング |
| `view_3d(colormap, contrast_limits, save_dir)` | reconstructed | napari で 3D 表示・PNG 保存 |
| `sigma_v` / `sigma_x` / `sigma_y` / `sigma_z` | interpolated 以降 | フラックス加重統計 |

### VelocityField（ABC）
- `compute_k(sigma_v, sigma_z) -> float` と `velocity_to_depth(v) -> ndarray` を抽象メソッドとして持つ
- `with_k(k)` / `with_k_from_sigmas(sigma_v, sigma_z)` は基底クラスに実装済み
- サブクラス: `LinearVelocityField`（k = σ_v / σ_z）、`PowerLawVelocityField(alpha)`（k = σ_v / σ_z^α）
- Protocol ではなく ABC を選択した理由: `with_k()` 等の共有実装が必要なため

### σ 統計プロパティ
- `_flux_weighted_stats(flux, array)` 静的ヘルパーを全軸で共用
- `sigma_v` は interpolated ステージ以降で使用可能
- `sigma_x`, `sigma_y`, `sigma_z` は x_array / y_array が設定済みであれば使用可能（interpolated 以降）
- `sigma_z = sqrt(0.5 * (σ_x² + σ_y²))` — 球対称仮定、reconstruct 前に計算可能

### y_array の単位（重要）
- `ImageUnit.spatial_array` は FITS ヘッダーの CRVAL2/CDELT2（または CD2_2）から生成する
- STIS `_x2d.fits` では `CUNIT2` が省略されており `CD2_2` が degrees 単位で入っている
- `spatial_array` は `CUNIT2="deg"` または `CD2_2` が存在する場合に自動で `× 3600` して arcsec に変換する
- `cunit1` / `cunit2` プロパティで単位を確認可能
- この変換漏れにより `sigma_y` / `sigma_z` / `k` が約 3600 倍ズレていた（issue #25 で修正済み）
- **実データで検証必須**: `raw_cube.y_array` の値が arcsec スケール（〜0.05 程度）であることを確認すること

### 可視化（napari）
- `DataCube.view_3d(colormap, contrast_limits, save_dir)` で reconstructed cube を napari で表示
- `save_dir` 指定時はオフスクリーンで `view_3d.png` として保存、GUI は開かない
- napari は optional extras のためメソッド内でローカルインポート
- `trim_y()` でトリミングしてから `view_3d()` に渡すのが標準的な使い方

## コーディング規約

- Python 3.13、型ヒントを全メソッドに記述
- Pyright 準拠（`assert x is not None` でナローイング、スクリプト内で使用）
- クラス内セクション順: ステージ判定 → `__repr__` → コンストラクタ → 統計ヘルパー → 処理メソッド → 可視化メソッド
- テストは pytest、`unittest.mock` を使用。ファイル I/O を伴うものは `InstrumentModel.load` と `DataCube.from_proc_files` をモック
- ネガティブフラックスは統計計算前に 0 クリップ（`np.where(flux > 0, flux, 0.0)`）
- optional extras（napari, scipy）はメソッド内でローカルインポートする
- `field(repr=False)` で大きなオブジェクトを `__repr__` から除外する（`reader_collection`, `image_collection` 等）

## 依存関係

```toml
# 必須
numpy>=1.26, astropy>=6.0, matplotlib>=3.8, pandas>=3.0

# optional extras
lacosmic   = [lacosmic, scipy]
processing = [stistools, scipy, crds]
reconstruct = [scipy, napari>=0.7.0, PyQt6>=6.6]
```

`poetry install --extras reconstruct` で reconstruct グループをインストール。
napari を extras に追加する際は先に `poetry add napari PyQt6` でロックファイルに登録してから `pyproject.toml` の extras に手書きする（issue #22 参照）。

## 未実装（保留中）

- `DataCube.imshow_channel()` / `plot_spectrum()` / `imshow_integrated()` — `raise NotImplementedError`（matplotlib 2D）
- `ReconstructResult.plot_channel_map()` / `plot_reconstructed_slice()` — `raise NotImplementedError`

## よく使うコマンド

```bash
# テスト実行
poetry run pytest

# 型チェック
poetry run pyright src/

# 特定サブパッケージのテストのみ
poetry run pytest tests/test_reconstruct/ -v
```

## 実データ確認時のチェックリスト（issue #25）

```python
# y_array の単位確認（arcsec スケールになっているか）
raw_cube.y_array[1] - raw_cube.y_array[0]   # 〜0.05 なら正常

# ヘッダー確認
raw_cube.reader_collection[0].header(1).get("CUNIT2")
raw_cube.reader_collection[0].header(1).get("CD2_2")

# sigma 値の確認
_, sigma_v = interp_cube.sigma_v
print(f"sigma_v : {sigma_v:.3f} km/s")
print(f"sigma_z : {interp_cube.sigma_z:.4f} arcsec")
print(f"k       : {sigma_v / interp_cube.sigma_z:.3f} km/s/arcsec")
```
