# stis_analysis

HST/STIS 分光データの宇宙線除去・処理・解析を行う統合パッケージ。

## 概要

`stis_analysis` は、Hubble Space Telescope (HST) の Space Telescope Imaging Spectrograph (STIS) で取得したデータを処理するモノレポです。以下の 3 つのステージで構成されています。

```mermaid
graph LR
    A["_crj.fits"] -->|"Stage 1: lacosmic"| B["_lac.fits"]
    B -->|"Stage 2: processing"| C["_proc.fits (×6)"]
    C -->|"Stage 3: reconstruct"| D["ReconstructedCube"]
```

| Stage | 処理内容 | 入力 → 出力 |
|---|---|---|
| **1. lacosmic** | LA-Cosmic 宇宙線除去 | `_crj.fits` → `_lac.fits` |
| **2. processing** | stistools pipeline → 連続光差し引き → OIII λ4959 除去 → ±2500 km/s 切り取り | `_lac.fits` → `_proc.fits` (×6) |
| **3. reconstruct** | 6スリット → 3D Cube → x方向補間 → 速度場推定 → 3D 再構成 | `_proc.fits (×6)` → `ReconstructedCube` |

---

## パッケージ構成

```
stis_analysis/
├── src/stis_analysis/
│   ├── core/                          ← 共通基盤
│   │   ├── fits_reader.py             ← STISFitsReader, ReaderCollection
│   │   ├── instrument.py              ← InstrumentModel
│   │   ├── image.py                   ← ImageUnit
│   │   └── wave_constants.py          ← 波長定数・輝線波長 (c_kms, oiii5007_stp 等)
│   │
│   ├── lacosmic/                      ← Stage 1: 宇宙線除去
│   │   ├── image.py                   ← ImageModel, ImageCollection
│   │   └── pipeline.py                ← LaCosmicPipeline, PipelineResult
│   │
│   ├── processing/                    ← Stage 2: スペクトル処理
│   │   ├── image.py                   ← ProcessingImageModel, ProcessingImageCollection
│   │   ├── pipeline.py                ← ProcessingPipeline, ProcessingResult
│   │   └── wave_constants.py          ← core.wave_constants の再エクスポート
│   │
│   └── reconstruct/                   ← Stage 3: 3D 再構成（実装予定）
│       ├── cube.py                    ← DataCube（raw/interpolated/reconstructed を統一）
│       ├── velocity_field.py          ← VelocityField, LinearVelocityField, PowerLawVelocityField
│       └── pipeline.py                ← ReconstructPipeline, ReconstructResult
│
├── scripts/
│   ├── run_lacosmic.py                ← Stage 1 ステップ確認（IPython 対話用）
│   ├── run_lacosmic_pipeline.py       ← Stage 1 パイプライン一括実行
│   ├── run_processing.py              ← Stage 2 ステップ確認（IPython 対話用）
│   ├── run_processing_pipeline.py     ← Stage 2 パイプライン一括実行
│   ├── run_reconstruct.py             ← Stage 3 パイプライン一括実行（実装予定）
│   ├── check_lacosmic_residual.py     ← LA-Cosmic 残差確認
│   └── convolve2d_reference.py        ← convolve2d 参考実装
│
├── tests/
│   ├── test_core/
│   ├── test_lacosmic/
│   ├── test_processing/               ← test_image.py, test_pipeline.py
│   └── test_reconstruct/              ← test_cube.py, test_pipeline.py（実装予定）
│
├── pyproject.toml
└── README.md
```

### 主要クラス

#### `core`

| クラス / 定数 | 説明 |
|---|---|
| `STISFitsReader` | FITS ファイルを SCI / ERR / DQ HDU に分解して読み込む |
| `ReaderCollection` | 複数の `STISFitsReader` をまとめるコレクション |
| `InstrumentModel` | ディレクトリ探索・ファイルリスト管理 |
| `ImageUnit` | data / header のペア。`wavelength`・`plot_spectrum()` を持つ |
| `wave_constants` | `c_kms`, `oiii5007_stp`, `oiii4959_stp` 等の輝線波長定数 |

#### `lacosmic`

| クラス | 説明 |
|---|---|
| `ImageModel` | 1 枚の FITS 画像に対する LA-Cosmic 宇宙線除去モデル |
| `ImageCollection` | 複数 `ImageModel` のコレクション。`remove_cosmic_ray()` を提供 |
| `LaCosmicPipeline` | `_crj.fits` → 宇宙線除去 → `_lac.fits` 書き出しを一括実行 |
| `PipelineResult` | `before` / `after` の `ImageCollection` と出力パスリストを保持 |

#### `processing`

| クラス | 説明 |
|---|---|
| `ProcessingImageModel` | 1 枚の `_lac.fits` に対するスペクトル処理モデル。`setup()` で処理パラメータを設定し、`subtract_continuum()` → `remove_o3_4959()` → `clip_velocity_range()` をチェーンで適用できる |
| `ProcessingImageCollection` | 複数 `ProcessingImageModel` のコレクション |
| `ProcessingPipeline` | `_lac.fits` → x2d 補正 → 連続光差し引き → OIII λ4959 除去 → velocity clipping → `_proc.fits` 書き出しを一括実行 |
| `ProcessingResult` | `before` / `after` の `ProcessingImageCollection` と出力パスリストを保持。`plot_before_after()` / `plot_continuum_fit()` でステップ確認プロットを生成 |

#### `reconstruct`（実装予定）

| クラス | 説明 |
|---|---|
| `DataCube` | raw / interpolated / reconstructed の全ステージを統一した 3D スペクトルキューブ。`from_proc_files()` → `interpolate()` → `compute_sigma_v()` → `reconstruct(velocity_field)` をチェーンで適用できる |
| `VelocityField` | フラックス加重速度分散マップ（σ_v）と変換係数 k を保持する基底クラス。`velocity_to_depth(v)` で v→z 変換を提供 |
| `LinearVelocityField` | v = k·z モデル（デフォルト） |
| `PowerLawVelocityField` | v = k·z^α モデル（べき乗則モデル） |
| `ReconstructPipeline` | `_proc.fits` (×6) → DataCube 構築 → x補間 → σ_v 計算 → 3D 再構成を一括実行 |
| `ReconstructResult` | raw / interpolated / reconstructed の各 DataCube と VelocityField を保持。確認プロット生成メソッドを提供 |

---

## 開発ロードマップ

### Phase 1: 基盤構築 ✅ 完了
- [x] `core/` の実装 — `STISFitsReader`, `ReaderCollection`, `InstrumentModel`, `ImageUnit`
- [x] `lacosmic/` の移植 — `stis_la_cosmic` から import パスを書き換えて移行
- [x] `lacosmic/` のテスト整備

### Phase 2: Stage 2 開発 ✅ 完了
- [x] `processing/` サブパッケージの設計・実装
  - [x] stistools pipeline 連携 (`ProcessingPipeline._run_x2d_batch`・既存 `_x2d.fits` のスキップ)
  - [x] 連続光差し引き (`ProcessingImageModel.subtract_continuum`)
  - [x] OIII λ4959 除去 (`ProcessingImageModel.remove_o3_4959`)
  - [x] ±2500 km/s 切り取り (`ProcessingImageModel.clip_velocity_range`)
  - [x] 処理前後の比較プロット (`ProcessingResult.plot_before_after`, `plot_continuum_fit`)
  - [x] プロット静的メソッド分離 (`ProcessingResult._plot_before_after`, `_plot_continuum_fit`)
  - [x] 出力ディレクトリ自動退避 (`ProcessingPipeline._resolve_output_dir`)
  - [x] `run()` に `save_picture` / `slit_index` パラメータ追加（処理ステップ毎の確認プロット保存）
  - [x] `ProcessingImageModel.setup()` クラスメソッド追加（処理パラメータをフィールド化）
- [x] `processing/` のテスト整備（`ProcessingImageModel` のユニットテスト）
- [x] `ProcessingPipeline` の統合テスト整備（`_resolve_output_dir` + `run()` 退避動作）

> **スコープ**: `processing` の責務は `_lac.fits` → `_proc.fits` 出力まで。
> 3D Cube 結合・空間補間は Stage 3 で扱う。

### Phase 3: Stage 3 開発（`reconstruct/`）
- [ ] **3-1** `DataCube` の設計・実装（`cube.py`）
  - [ ] `from_proc_files()` — `_proc.fits` (×6) 読み込み・velocity軸変換・raw cube 構築
  - [ ] `interpolate()` — x方向をyピクセル間隔（0.05"/pix）に `np.interp` で補間
  - [ ] `compute_sigma_v()` — フラックス加重速度分散 σ_v マップ計算
  - [ ] `reconstruct(velocity_field)` — `velocity_field.velocity_to_depth(v)` で velocity→z 軸変換
  - [ ] 可視化メソッド（`imshow_channel`, `plot_spectrum`, `imshow_integrated`）
- [ ] **3-2** `VelocityField` の設計・実装（`velocity_field.py`）
  - [ ] 基底クラス `VelocityField`（σ_v map, k, `velocity_to_depth()` インターフェース）
  - [ ] `LinearVelocityField` — v = k·z モデル
  - [ ] `PowerLawVelocityField` — v = k·z^α モデル
- [ ] **3-3** `ReconstructPipeline` / `ReconstructResult` の実装（`pipeline.py`）
  - [ ] `ReconstructPipeline.run()` — ファイル探索 → DataCube構築 → 補間 → σ_v → 再構成
  - [ ] `ReconstructResult` 確認プロット（`plot_sigma_v_map`, `plot_channel_map`, `plot_reconstructed_slice`）
- [ ] **3-4** `scripts/run_reconstruct.py` の整備
- [ ] **3-5** `reconstruct/` のテスト整備（`test_cube.py`, `test_pipeline.py`）

### Phase 4: `BaseImageModel` 抽出 (将来)
- [ ] `processing/` 安定後に `core/image.py` へ共通基底クラスを抽出

---

## インストール

### Step 0: conda (Miniconda) のインストール

conda が未インストールの場合は先にインストールしてください。

→ [Miniconda 公式インストールガイド](https://docs.anaconda.com/miniconda/install/)

### Step 1: hstcal のインストール（Stage 2 を使う場合のみ）

Stage 2 の `processing` パイプラインは、STScI が提供する CALSTIS バイナリ (`cs7.e`) に依存しています。
このバイナリは conda 経由でのみ配布されているため、以下の手順で別途インストールしてください。

```bash
# hstcal 専用の conda 環境を作成してインストール
conda create -n calstis_bin
conda install -n calstis_bin -c conda-forge hstcal
```

次に、`~/.zshrc`（または `~/.bashrc`）に以下を追記してください：

```zsh
# CALSTIS binary (cs7.e) for stistools/HST STIS pipeline
# See: https://github.com/HisadaRintaro/stis_analysis
export PATH="$HOME/miniconda3/envs/calstis_bin/bin:$PATH"
```

追記後、シェルを再起動するか `source ~/.zshrc` を実行してください。

> `$(conda info --base)` は shell 起動時に conda が未初期化だとエラーになるため、
> `$HOME/miniconda3`（Miniconda のデフォルトインストール先）をハードコードしています。

> Stage 1 (lacosmic) のみ使用する場合、この手順は不要です。

### Step 1.5: CRDS キャリブレーション参照ファイルのセットアップ（Stage 2 を使う場合のみ）

**CRDS（Calibration Reference Data System）** は STScI が管理するキャリブレーション参照ファイルの配布システムです。
CALSTISが波長較正・感度補正等を行うために必要なファイル群（`oref$...`）をここから取得します。

→ [CRDS 公式ドキュメント](https://hst-crds.stsci.edu/static/users_guide/index.html)

> **研究グループでの利用について**
> 参照ファイルは容量が大きいため、共有サーバーや NAS 等にまとめて配置し、
> メンバー全員が同じパスを `CRDS_PATH` / `oref` に設定して使用することを推奨します。
> 個人環境で使う場合は `$HOME/crds_cache` に置きます。

**① `~/.zshrc` に環境変数を追記**

```zsh
# HST Calibration Reference Data System
export CRDS_SERVER_URL="https://hst-crds.stsci.edu"
export CRDS_PATH="$HOME/crds_cache"           # 共有サーバーの場合はそのパスに変更
export oref="$HOME/crds_cache/references/hst/oref/"
```

**② 参照ファイルをダウンロード（初回のみ・共有サーバーへの配置を推奨）**

```bash
# 必要なファイルを個別に取得（推奨：容量が少ない。ただし必要なファイルは使うデータによって異なります）
crds sync --contexts hst_latest.pmap --fetch-references \
  --files 16j16006o_sdc.fits 16j16005o_apd.fits l2j0137to_dsp.fits \
  h5s11397o_iac.fits qa31608go_1dt.fits 95118575o_pht.fits \
  y2r1559to_apt.fits q541740oo_pct.fits t9a1003so_tds.fits

# または STIS 参照ファイルを全件同期（容量大）
crds sync --contexts hst_latest.pmap --fetch-references
```

> **途中で中断した場合**は再度同じコマンドを実行してください。
> `crds sync` は冪等なのでダウンロード済みのファイルはスキップされます。

必要なファイル名は処理対象の FITS ファイルによって異なります。
`pipeline.run()` 実行時に不足ファイルと取得コマンドが自動表示されます。

### Step 2: パッケージのインストール

```bash
# 基本インストール
pip install stis-analysis

# Stage 1 (LA-Cosmic) を使う場合
pip install "stis-analysis[lacosmic]"

# Stage 2 (processing) を使う場合
pip install "stis-analysis[processing]"

# 全機能
pip install "stis-analysis[all]"
```

> **開発環境:**
> ```bash
> git clone https://github.com/HisadaRintaro/stis_analysis.git
> cd stis_analysis
> poetry install --with dev
> ```

---

## 依存関係

```toml
[project.optional-dependencies]
lacosmic = ["lacosmic>=1.4.0"]
processing = ["stistools>=1.4.7", "scipy>=1.17.0"]
all = ["stis-analysis[lacosmic,processing]"]
```

---

## 既存リポジトリとの関係

| リポジトリ | 方針 |
|---|---|
| `stis_la_cosmic` | `stis_analysis.lacosmic` に統合。元リポジトリはアーカイブ予定 |
| `spectrum_package` | フロー変更のためアーカイブ。再利用可能な部品は `core/` に流用 |

---

## ライセンス

MIT License
