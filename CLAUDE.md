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
  run_processing.py            STIS 処理（ステップ確認版）
  run_processing_pipeline.py   STIS 処理（ワンショット版）
  run_reconstruct.py           3D 再構成（ステップ確認版）
  run_reconstruct_pipeline.py  3D 再構成（ワンショット版）

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

## コーディング規約

- Python 3.13、型ヒントを全メソッドに記述
- Pyright 準拠（`assert x is not None` でナローイング、スクリプト内で使用）
- クラス内セクション順: ステージ判定 → `__repr__` → コンストラクタ → 統計ヘルパー → 処理メソッド → 可視化メソッド
- テストは pytest、`unittest.mock` を使用。ファイル I/O を伴うものは `InstrumentModel.load` と `DataCube.from_proc_files` をモック
- ネガティブフラックスは統計計算前に 0 クリップ（`np.where(flux > 0, flux, 0.0)`）

## 依存関係

```toml
# 必須
numpy>=1.26, astropy>=6.0, matplotlib>=3.8, pandas>=3.0

# optional extras
lacosmic  = [lacosmic, scipy]
processing = [stistools, scipy, crds]
reconstruct = [scipy, pyvista]  # mayavi は Python 3.13 + 新 VTK でビルド不可
```

`poetry install --extras reconstruct` で reconstruct グループをインストール。

## 未実装（保留中）

- `DataCube.imshow_channel()` / `plot_spectrum()` / `imshow_integrated()` — `raise NotImplementedError`
- `ReconstructResult.plot_channel_map()` / `plot_reconstructed_slice()` — `raise NotImplementedError`
- 可視化は pyvista（`pyvista>=0.44`）を使う予定。mayavi はビルド不可のため使わない

## よく使うコマンド

```bash
# テスト実行
poetry run pytest

# 型チェック
poetry run pyright src/

# 特定サブパッケージのテストのみ
poetry run pytest tests/test_reconstruct/ -v
```
