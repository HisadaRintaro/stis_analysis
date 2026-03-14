"""LA-Cosmic パイプライン.

_crj FITS ファイルの読み込みから宇宙線除去、_lac ファイル出力までの
全ワークフローをひとつのメソッド呼び出しで実行する高レベル API。
"""

import warnings
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
    output_dir : Path
        実際に使用された出力ディレクトリ
    """

    before: ImageCollection
    after: ImageCollection
    output_paths: list[Path]
    output_dir: Path


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
    maxiter : int
        宇宙線除去の最大反復回数（デフォルト: 1）
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
    maxiter: int = 1
    dq_flags: int = 16
    suffix: str = "_crj"
    extension: str = ".fits"
    depth: int = 1
    exclude_files: tuple[str, ...] = ()

    @staticmethod
    def _resolve_output_dir(base: Path, output_suffix: str) -> Path:
        """output_dir に既存の出力ファイルがある場合、番号付きディレクトリを返す.

        ``base`` に ``*{output_suffix}.fits`` が 1 件以上存在する場合は
        ``{base}-2``, ``{base}-3``, ... と順に探し、
        該当ファイルが存在しない最初のパスを返す。
        存在しない or 空の場合はそのまま ``base`` を返す。

        Parameters
        ----------
        base : Path
            指定された出力先ディレクトリ
        output_suffix : str
            出力ファイルの接尾辞（例: "_lac"）

        Returns
        -------
        Path
            実際に使用するディレクトリパス
        """
        if not base.exists() or not any(base.glob(f"*{output_suffix}.fits")):
            return base
        n = 2
        while True:
            candidate = base.parent / f"{base.name}-{n}"
            if not candidate.exists() or not any(candidate.glob(f"*{output_suffix}.fits")):
                warnings.warn(
                    f"'{base}' には既存の '{output_suffix}.fits' ファイルがあります。"
                    f" '{candidate}' に保存します。",
                    UserWarning,
                    stacklevel=4,
                )
                return candidate
            n += 1

    def run(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        output_suffix: str = "_lac",
        save_picture: bool = False,
        slit_index: int | None = None,
        recession_velocity: float | None = None,
    ) -> PipelineResult:
        """パイプラインを実行する.

        入力ディレクトリから _crj FITS ファイルを探索・読み込み、
        LA-Cosmic で宇宙線を除去し、結果を _lac FITS ファイルとして出力する。

        output_dir に ``*{output_suffix}.fits`` が既に存在する場合は
        ``{output_dir}-2``, ``{output_dir}-3``, ... へ自動退避して保存する。

        Parameters
        ----------
        input_dir : str | Path
            入力 _crj ファイルを含むディレクトリ
        output_dir : str | Path
            出力 _lac ファイルの書き出し先ディレクトリ。
            既存ファイルがある場合は番号付きディレクトリに退避する。
        output_suffix : str, optional
            出力ファイルの接尾辞（デフォルト: "_lac"）
        save_picture : bool, optional
            True の場合、output_dir に確認用画像を保存する（デフォルト: False）。
            保存されるファイル:
            - imshow.png                      : 処理後画像一覧
            - imshow_mask_dq.png              : DQ マスク一覧
            - imshow_mask_cr.png              : LA-Cosmic 検出マスク一覧
            - spectrum_comparison_slit{N}.png : スペクトル比較（slit_index 指定時）
            - residual_figure(...).png        : 残差プロット（slit_index + recession_velocity 指定時）
        slit_index : int | None, optional
            スペクトル比較・残差プロットに使用するスリット行インデックス。
            save=True のときのみ有効。
        recession_velocity : float | None, optional
            残差プロット用の銀河後退速度 [km/s]。
            save=True かつ slit_index 指定時のみ有効。

        Returns
        -------
        PipelineResult
            before / after の ImageCollection・出力パス・実際の output_dir を含む結果オブジェクト
        """
        output_path = self._resolve_output_dir(Path(output_dir), output_suffix)
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
        )

        # 3. 宇宙線除去
        print("Running LA-Cosmic...")
        after = before.remove_cosmic_ray(maxiter=self.maxiter)

        # 4. FITS 書き出し
        paths = after.write_fits(
            output_suffix=output_suffix,
            output_dir=output_path,
            overwrite=False,
        )
        for p in paths:
            print(f"  wrote: {p}")

        # 5. 確認用画像の保存
        if save_picture:
            after.imshow(save_dir=output_path, title="after")
            before.imshow(save_dir=output_path, title="before")
            after.imshow_mask(mask_type="dq", save_dir=output_path)
            after.imshow_mask(mask_type="cr", save_dir=output_path)
            if slit_index is None:
                print("\nslit_index is None, skipping spectrum comparison and residual plot")
            else:
                before.plot_spectrum_comparison(after, slit_index, save_dir=output_path)
                if recession_velocity is None:
                    print("\nrecession_velocity is None, skipping residual plot")
                else:
                    before.plot_lacosmic_residual(
                        after, slit_index, recession_velocity, save_dir=output_path
                    )

        print("Done.")
        return PipelineResult(before=before, after=after, output_paths=paths, output_dir=output_path)
