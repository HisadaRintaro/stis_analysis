"""処理パイプライン.

_lac.fits ファイルに対して以下を順番に実行するパイプライン:
  1. stistools.x2d による 2D 幾何補正（ファイルベース）
  2. 連続光差し引き
  3. OIII λ4959 除去
  4. velocity range clipping
  5. _proc.fits として書き出し
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from stis_analysis.core.fits_reader import ReaderCollection
from stis_analysis.core.instrument import InstrumentModel
from stis_analysis.processing.image import ProcessingImageCollection
from stis_analysis.processing.wave_constants import oiii5007_stp


@dataclass(frozen=True)
class ProcessingResult:
    """パイプライン実行結果.

    Attributes
    ----------
    before : ProcessingImageCollection
        処理前（x2d 適用後・連続光差し引き前）の画像コレクション
    after : ProcessingImageCollection
        全処理適用後の画像コレクション
    output_paths : list[Path]
        書き出された _proc.fits ファイルのパスリスト
    """

    before: ProcessingImageCollection
    after: ProcessingImageCollection
    output_paths: list[Path]


@dataclass(frozen=True)
class ProcessingPipeline:
    """STIS スペクトル処理パイプライン.

    Attributes
    ----------
    continuum_windows_kms : list[tuple[float, float]]
        連続光ウィンドウ [km/s] のリスト。
        例: [(-4000, -3200), (3000, 4000)]
        OIII λ5007 観測波長を v=0 とした相対速度で指定する。
    continuum_degree : int
        連続光フィットの多項式次数（デフォルト: 1 = 直線）
    recession_velocity : float
        銀河の後退速度 [km/s]（デフォルト: 1148.0 = NGC1068）
    rest_wavelength : float
        velocity clipping の基準静止波長 [Å]（デフォルト: oiii5007_stp）
    v_min : float
        velocity clipping の下限 [km/s]（デフォルト: -2500.0）
    v_max : float
        velocity clipping の上限 [km/s]（デフォルト: 2500.0）
    o3_scale : float | None
        OIII λ4959 除去スケール係数。None の場合は 1/2.98 を使用。
    o3_half_width_aa : float
        OIII λ4959 除去の処理対象半幅 [Å]（デフォルト: 30.0）
    suffix : str
        入力ファイルのサフィックス（デフォルト: "_lac"）
    extension : str
        入力ファイルの拡張子（デフォルト: ".fits"）
    depth : int
        入力ディレクトリの探索深度（デフォルト: 1）
    exclude_files : tuple[str, ...]
        除外するファイルの stem リスト
    dq_flags : int
        マスク対象の DQ ビットフラグ（デフォルト: 16 = hot pixel）
    """

    continuum_windows_kms: list[tuple[float, float]]
    continuum_degree: int = 1
    recession_velocity: float = 1148.0
    rest_wavelength: float = oiii5007_stp
    v_min: float = -2500.0
    v_max: float = 2500.0
    o3_scale: float | None = None
    o3_half_width_aa: float = 30.0
    suffix: str = "_lac"
    extension: str = ".fits"
    depth: int = 1
    exclude_files: tuple[str, ...] = ()
    dq_flags: int = 16

    def run(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        output_suffix: str = "_proc",
        overwrite: bool = False,
        run_x2d: bool = True,
    ) -> ProcessingResult:
        """パイプラインを実行する.

        Parameters
        ----------
        input_dir : str | Path
            _lac.fits が格納されたルートディレクトリ
        output_dir : str | Path
            出力先ディレクトリ（存在しない場合は自動生成）
        output_suffix : str, optional
            出力ファイルの接尾辞（デフォルト: "_proc"）
        overwrite : bool, optional
            既存ファイルの上書きを許可するか（デフォルト: False）
        run_x2d : bool, optional
            True の場合 stistools.x2d を実行する（デフォルト: True）。
            False にするとファイルをそのまま読み込む（テスト用途）。

        Returns
        -------
        ProcessingResult
            処理前後の ImageCollection と出力パスリスト
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. _lac.fits ファイルを探索
        instrument = InstrumentModel.load(
            file_directory=str(input_dir),
            suffix=self.suffix,
            extension=self.extension,
            depth=self.depth,
            exclude_files=self.exclude_files,
        )
        lac_paths = instrument.path_list

        # 2. x2d 幾何補正（ファイルベース）
        if run_x2d:
            x2d_paths = self._run_x2d_batch(lac_paths, output_dir)
        else:
            x2d_paths = lac_paths

        # 3. x2d 済みファイルを ProcessingImageCollection として読み込み
        readers = ReaderCollection.from_paths(x2d_paths)
        before = ProcessingImageCollection.from_readers(readers, dq_flags=self.dq_flags)

        # 4. インメモリ処理チェーン
        after = before
        after = after.subtract_continuum(
            continuum_windows_kms=self.continuum_windows_kms,
            recession_velocity=self.recession_velocity,
            rest_wavelength=self.rest_wavelength,
            degree=self.continuum_degree,
        )
        after = after.remove_o3_4959(
            recession_velocity=self.recession_velocity,
            scale=self.o3_scale,
            half_width_aa=self.o3_half_width_aa,
        )
        after = after.clip_velocity_range(
            v_min=self.v_min,
            v_max=self.v_max,
            recession_velocity=self.recession_velocity,
            rest_wavelength=self.rest_wavelength,
        )

        # 5. _proc.fits として書き出し
        output_paths = after.write_fits(
            output_suffix=output_suffix,
            output_dir=output_dir,
            overwrite=overwrite,
        )

        return ProcessingResult(
            before=before,
            after=after,
            output_paths=output_paths,
        )

    @staticmethod
    def _run_x2d_batch(
        lac_paths: list[Path],
        output_dir: Path,
    ) -> list[Path]:
        """_lac.fits のリストに stistools.x2d を一括実行する.

        Parameters
        ----------
        lac_paths : list[Path]
            _lac.fits ファイルパスのリスト
        output_dir : Path
            x2d 出力先ディレクトリ

        Returns
        -------
        list[Path]
            x2d 済みファイルパスのリスト（lac_paths と同順）
        """
        try:
            from stistools import x2d as x2d_module  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "stistools が見つかりません。"
                "pip install stistools または poetry add stistools でインストールしてください。"
            ) from exc

        x2d_paths: list[Path] = []
        for lac_path in lac_paths:
            out_path = output_dir / lac_path.name.replace("_lac", "_x2d")
            x2d_module.x2d(input=str(lac_path), output=str(out_path))
            x2d_paths.append(out_path)

        return x2d_paths
