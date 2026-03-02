"""LA-Cosmic パイプライン.

_crj FITS ファイルの読み込みから宇宙線除去、_lac ファイル出力までの
全ワークフローをひとつのメソッド呼び出しで実行する高レベル API。
"""

from dataclasses import dataclass
from pathlib import Path

from stis_analysis.core.instrument import InstrumentModel
from stis_analysis.core.fits_reader import ReaderCollection
from .image import ImageCollection


@dataclass(frozen=True)
class PipelineResult:
    """パイプライン実行結果.

    Attributes
    ----------
    before : ImageCollection
        宇宙線除去前の ImageCollection
    after : ImageCollection
        宇宙線除去後の ImageCollection
    output_paths : list[Path]
        書き出された _lac FITS ファイルのパスリスト
    """

    before: ImageCollection
    after: ImageCollection
    output_paths: list[Path]


@dataclass(frozen=True)
class LaCosmicPipeline:
    """LA-Cosmic 宇宙線除去パイプライン.

    パラメータを保持し、run() で
    「_crj 読み込み → 宇宙線除去 → _lac 書き出し」を一括実行する。

    Attributes
    ----------
    contrast : float
        ラプラシアン/ノイズ比のコントラスト閾値（デフォルト: 5.0）
    cr_threshold : float
        宇宙線検出のシグマクリッピング閾値（デフォルト: 5.0）
    neighbor_threshold : float
        近傍ピクセルの検出閾値（デフォルト: 5.0）
    error : float
        誤差配列のスケール係数（デフォルト: 5.0）
    dq_flags : int
        マスク対象の DQ ビットフラグ（デフォルト: 16 = hot pixel）
    suffix : str
        入力ファイルの接尾辞（デフォルト: "_crj"）
    extension : str
        入力ファイルの拡張子（デフォルト: ".fits"）
    depth : int
        ディレクトリ探索の深度（デフォルト: 1）
    exclude_files : tuple[str, ...]
        除外するファイル名のタプル（デフォルト: ()）
    """

    contrast: float = 5.0
    cr_threshold: float = 5.0
    neighbor_threshold: float = 5.0
    error: float = 5.0
    dq_flags: int = 16
    suffix: str = "_crj"
    extension: str = ".fits"
    depth: int = 1
    exclude_files: tuple[str, ...] = ()

    def run(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        output_suffix: str = "_lac",
        overwrite: bool = False,
    ) -> PipelineResult:
        """パイプラインを実行する.

        入力ディレクトリから _crj FITS ファイルを探索・読み込み、
        LA-Cosmic で宇宙線を除去し、結果を _lac FITS ファイルとして出力する。

        Parameters
        ----------
        input_dir : str | Path
            入力 _crj ファイルを含むディレクトリ
        output_dir : str | Path
            出力 _lac ファイルの書き出し先ディレクトリ
        output_suffix : str, optional
            出力ファイルの接尾辞（デフォルト: "_lac"）
        overwrite : bool, optional
            既存ファイルの上書きを許可するか（デフォルト: False）

        Returns
        -------
        PipelineResult
            before / after の ImageCollection と出力パスを含む結果オブジェクト
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. ファイル探索
        inst = InstrumentModel(
            file_directory=str(input_dir),
            suffix=self.suffix,
            extension=self.extension,
            depth=self.depth,
            exclude_files=self.exclude_files,
        )
        print(f"Found {len(inst.path_list)} files in {input_dir}")

        # 2. FITS 読み込み
        readers = ReaderCollection.from_paths(inst.path_list)
        before = ImageCollection.from_readers(
            readers,
            dq_flags=self.dq_flags,
            contrast=self.contrast,
            cr_threshold=self.cr_threshold,
            neighbor_threshold=self.neighbor_threshold,
            error=self.error,
        )

        # 3. 宇宙線除去
        print("Running LA-Cosmic...")
        after = before.remove_cosmic_ray()

        # 4. FITS 書き出し
        paths = after.write_fits(
            output_suffix=output_suffix,
            output_dir=output_path,
            overwrite=overwrite,
        )
        for p in paths:
            print(f"  wrote: {p}")

        print("Done.")
        return PipelineResult(before=before, after=after, output_paths=paths)
