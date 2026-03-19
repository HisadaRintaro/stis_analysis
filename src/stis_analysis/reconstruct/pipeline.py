"""ReconstructPipeline / ReconstructResult — 3D 再構成パイプライン.

ProcessingPipeline / ProcessingResult パターンを踏襲する:
  1. InstrumentModel でファイル探索
  2. DataCube.from_proc_files() — raw cube 構築
  3. DataCube.interpolate()     — x 方向補間
  4. DataCube.sigma_v           — フラックス加重速度分散 σ_v を取得
  5. DataCube.sigma_z           — 空間分散から深度分散 σ_z を推定（球対称仮定）
  6. VelocityField.with_k_from_sigmas(sigma_v, sigma_z) — モデルから k を自動計算
  7. DataCube.reconstruct(vf)   — 3D 再構成
  8. save_picture=True なら確認プロット保存
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stis_analysis.core.instrument import InstrumentModel
from stis_analysis.core.wave_constants import oiii5007_stp
from stis_analysis.reconstruct.cube import DataCube
from stis_analysis.reconstruct.velocity_field import (
    LinearVelocityField,
    PowerLawVelocityField,
    VelocityField,
)


@dataclass(frozen=True)
class ReconstructResult:
    """3D 再構成の結果を保持するデータクラス.

    Attributes
    ----------
    raw_cube : DataCube
        from_proc_files() で構築した raw ステージの DataCube
    interpolated_cube : DataCube
        interpolate() 後の DataCube
    velocity_field : VelocityField
        k 設定済みの VelocityField
    reconstructed_cube : DataCube
        reconstruct() 後の DataCube
    """

    raw_cube: DataCube
    interpolated_cube: DataCube
    velocity_field: VelocityField
    reconstructed_cube: DataCube

    # ------------------------------------------------------------------
    # 確認プロット
    # ------------------------------------------------------------------

    def plot_channel_map(self, v_index: int, ax=None, save_dir: Path | None = None):
        """指定速度インデックスのチャンネルマップを表示する.

        Parameters
        ----------
        v_index : int
            速度軸のインデックス
        ax : matplotlib.axes.Axes, optional
            描画先 Axes。None の場合は新規作成。
        save_dir : Path, optional
            保存先ディレクトリ。None の場合は保存しない。
        """
        raise NotImplementedError

    def plot_reconstructed_slice(
        self, z_index: int, ax=None, save_dir: Path | None = None
    ):
        """指定深度インデックスの reconstructed cube スライスを表示する.

        Parameters
        ----------
        z_index : int
            深度軸のインデックス
        ax : matplotlib.axes.Axes, optional
            描画先 Axes。None の場合は新規作成。
        save_dir : Path, optional
            保存先ディレクトリ。None の場合は保存しない。
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ReconstructPipeline:
    """_proc.fits (×6) → 3D 再構成を一括実行するパイプライン.

    Attributes
    ----------
    slit_positions : list[float]
        各スリットの x 位置 [arcsec]。FITSヘッダーから取得せず外部設定する。
    velocity_field_model : str
        速度場モデル。"linear" または "power_law"。デフォルト: "linear"
    alpha : float
        `velocity_field_model="power_law"` のときのべき乗指数 α。デフォルト: 1.0
    recession_velocity : float
        銀河後退速度 [km/s]。デフォルト: 1148.0（NGC1068）
    rest_wavelength : float
        速度 v=0 の基準静止波長 [Å]。デフォルト: oiii5007_stp
    pixel_scale_arcsec : float
        DataCube.interpolate() の x グリッド間隔 [arcsec/pix]。デフォルト: 0.05
    suffix : str
        処理済みファイルの接尾辞。デフォルト: "_proc"
    depth : int
        InstrumentModel のサブディレクトリ探索深度。デフォルト: 0
    exclude_files : tuple[str, ...]
        除外するファイル名のパターン（部分一致）。デフォルト: ()
    """

    slit_positions: list[float]
    velocity_field_model: str = "linear"
    alpha: float = 1.0
    recession_velocity: float = 1148.0
    rest_wavelength: float = oiii5007_stp
    pixel_scale_arcsec: float = 0.05
    suffix: str = "_proc"
    depth: int = 0
    exclude_files: tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # パイプライン実行
    # ------------------------------------------------------------------

    def run(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        save_picture: bool = False,
    ) -> ReconstructResult:
        """パイプラインを実行する.

        Parameters
        ----------
        input_dir : Path | str
            `_proc.fits` が格納されたディレクトリ
        output_dir : Path | str
            確認プロット等の出力先ディレクトリ
        save_picture : bool, optional
            True の場合、output_dir に確認用画像を保存する。デフォルト: False

        Returns
        -------
        ReconstructResult

        Raises
        ------
        ValueError
            `slit_positions` の長さとファイル数が一致しない場合
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. ファイル探索
        instrument = InstrumentModel.load(
            file_directory=str(input_dir),
            suffix=self.suffix,
            extension=".fits",
            depth=self.depth,
            exclude_files=self.exclude_files,
        )
        proc_paths = instrument.path_list
        print(f"Step 1: Found {len(proc_paths)} files in {input_dir}")

        # 2. slit_positions との整合チェック
        if len(proc_paths) != len(self.slit_positions):
            raise ValueError(
                f"ファイル数 ({len(proc_paths)}) と slit_positions の長さ "
                f"({len(self.slit_positions)}) が一致しません。"
            )

        # 3. raw cube 構築
        raw_cube = DataCube.from_proc_files(
            paths=proc_paths,
            slit_positions=self.slit_positions,
            recession_velocity=self.recession_velocity,
            rest_wavelength=self.rest_wavelength,
        )
        print(f"Step 2: raw cube 構築完了  shape={raw_cube.data.shape}")

        # 4. x 方向補間
        interp_cube = raw_cube.interpolate(pixel_scale_arcsec=self.pixel_scale_arcsec)
        print(f"Step 3: interpolate 完了  shape={interp_cube.data.shape}")

        # 5. σ_v・σ_z を interpolated cube から計算
        _, sigma_v = interp_cube.sigma_v
        sigma_z = interp_cube.sigma_z
        print(f"Step 4: sigma_v={sigma_v:.2f} km/s  sigma_z={sigma_z:.4f} arcsec")

        # 6. VelocityField 構築（k をモデルから自動計算）
        velocity_field = self._build_velocity_field(sigma_v, sigma_z)
        print(f"Step 5: k={velocity_field.k:.4f} km/s/arcsec")

        # 7. 3D 再構成
        recon_cube = interp_cube.reconstruct(velocity_field)
        print(f"Step 6: reconstruct 完了  shape={recon_cube.data.shape}")

        # 8. 確認プロット（未実装のためスキップ）
        if save_picture:
            print("Step 7: 可視化メソッドは未実装のためスキップします。")

        return ReconstructResult(
            raw_cube=raw_cube,
            interpolated_cube=interp_cube,
            velocity_field=velocity_field,
            reconstructed_cube=recon_cube,
        )

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _build_velocity_field(self, sigma_v: float, sigma_z: float) -> VelocityField:
        """velocity_field_model に応じた VelocityField を構築し k を設定する."""
        if self.velocity_field_model == "linear":
            return LinearVelocityField().with_k_from_sigmas(sigma_v, sigma_z)
        elif self.velocity_field_model == "power_law":
            return PowerLawVelocityField(alpha=self.alpha).with_k_from_sigmas(
                sigma_v, sigma_z
            )
        else:
            raise ValueError(
                f"velocity_field_model には 'linear' か 'power_law' を指定してください: "
                f"{self.velocity_field_model!r}"
            )
