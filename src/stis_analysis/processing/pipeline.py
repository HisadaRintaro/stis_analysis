"""処理パイプライン.

_lac.fits ファイルに対して以下を順番に実行するパイプライン:
  1. stistools.x2d による 2D 幾何補正（ファイルベース）
  2. 連続光差し引き
  3. OIII λ4959 除去
  4. velocity range clipping
  5. _proc.fits として書き出し
"""

from __future__ import annotations

import math
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from stis_analysis.core.fits_reader import ReaderCollection
from stis_analysis.core.instrument import InstrumentModel
from stis_analysis.processing.image import ProcessingImageCollection, ProcessingImageModel
from stis_analysis.processing.wave_constants import oiii5007_stp, oiii5007_oiii4959


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

    def __repr__(self) -> str:
        paths_str = "".join(f"    {p},\n" for p in self.output_paths)
        return (
            f"ProcessingResult(\n"
            f"  before={len(self.before.images)} images,\n"
            f"  after={len(self.after.images)} images,\n"
            f"  output_paths=[\n"
            f"{paths_str}"
            f"  ],\n"
            f")"
        )

    @staticmethod
    def save_fig(ax: np.ndarray, save_path: Path | str, title: str | None = None) -> None:
        """Axes 配列から Figure を取得し、タイトルを設定して画像を保存・クローズする."""
        fig = ax.flat[0].figure
        if title:
            fig.suptitle(title)
        fig.savefig(save_path)
        print(f"saved {save_path}")
        plt.close(fig)

    @staticmethod
    def _plot_continuum_fit(
        before: ProcessingImageCollection,
        after: ProcessingImageCollection,
        slit_index: int,
        save_dir: Path | str | None = None,
        title: str | None = None,
    ) -> np.ndarray:
        """before スペクトルに連続光フィットを重ねた 3 列タイルプロットを生成する.

        before 画像のスペクトルを波長軸で描画し、after 画像に格納された
        連続光フィット情報（フィット線・ウィンドウ・OIII 位置）をアノテーションとして重ねる。
        after.continuum が未設定（subtract_continuum() 未実行）の場合は ValueError。
        """
        n = len(before.images)
        if n == 0:
            raise ValueError("画像が存在しません。")

        ncols = min(n, 3)
        nrows = math.ceil(n / ncols)

        fig, axes_2d = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )
        axes_list = list(axes_2d.flat)

        for ax, before_img, after_img in zip(axes_list, before.images, after.images):
            before_img.sci.plot_spectrum(slit_index, ax=ax,
                                         color="steelblue", lw=0.8, label="spectrum (before)")
            cast(ProcessingImageModel, after_img).plot_continuum_fit(
                slit_index=slit_index, ax=ax
            )
            ax.set_title(f"{before_img.filename} (slit={slit_index})")

        for ax in axes_list[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        title = title or "Continuum_Fit"
        if save_dir is not None:
            ProcessingResult.save_fig(axes_2d, Path(save_dir) / f"{title}_slit{slit_index}.png", title=title)

        plt.show()
        return axes_2d

    def plot_continuum_fit(
        self,
        slit_index: int,
        save_dir: Path | str | None = None,
        title: str | None = None,
    ) -> np.ndarray:
        """before スペクトルに連続光フィットを重ねた 3 列タイルプロットを生成する.

        Parameters
        ----------
        slit_index : int
            確認するスリット行のインデックス（空間方向）
        save_dir : Path | str | None, optional
            保存先ディレクトリ。指定すると `{title}_slit{slit_index}.png` として保存する。
        title : str | None, optional
            プロットのタイトル。指定しない場合は `"Continuum_Fit"` を使用する。

        Returns
        -------
        np.ndarray
            Axes の 2D 配列（shape: (nrows, ncols)）
        """
        return ProcessingResult._plot_continuum_fit(
            self.before, self.after, slit_index, save_dir, title
        )

    @staticmethod
    def _plot_before_after(
        before: ProcessingImageCollection,
        after: ProcessingImageCollection,
        slit_index: int,
        labels: tuple[str, str] = ("before", "after"),
        save_dir: Path | str | None = None,
        title: str | None = None,
    ) -> np.ndarray:
        """処理前後のスペクトルを重ねて比較するプロットを生成する.

        before と after のスペクトルを同一 Axes に波長軸で重ねて描画し、
        3 列タイルで全画像を並べる。
        """
        n = len(before.images)
        if n == 0:
            raise ValueError("画像が存在しません。")

        ncols = min(n, 3)
        nrows = math.ceil(n / ncols)

        fig, axes_2d = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )
        axes_list = list(axes_2d.flat)

        for ax, img_before, img_after in zip(axes_list, before.images, after.images):
            img_before.sci.plot_spectrum(slit_index, ax=ax,
                                         color="steelblue", lw=0.8, alpha=0.8, label=labels[0])
            img_after.sci.plot_spectrum(slit_index, ax=ax,
                                        color="tomato", lw=0.8, alpha=0.8, label=labels[1])
            ax.set_title(f"{img_before.filename} (slit={slit_index})")
            ax.legend(fontsize="small")

        for ax in axes_list[n:]:
            ax.set_visible(False)

        fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.93))
        title = title or "Before_After"
        if save_dir is not None:
            ProcessingResult.save_fig(axes_2d, Path(save_dir) / f"{title}_slit{slit_index}.png", title=title)

        plt.show()
        return axes_2d

    def plot_before_after(
        self,
        slit_index: int,
        labels: tuple[str, str] = ("before", "after"),
        save_dir: Path | str | None = None,
        title: str | None = None,
    ) -> np.ndarray:
        """処理前後のスペクトルを重ねて比較するプロットを生成する.

        Parameters
        ----------
        slit_index : int
            確認するスリット行のインデックス（空間方向）
        labels : tuple[str, str], optional
            凡例ラベル（before, after の順）
        save_dir : Path | str | None, optional
            保存先ディレクトリ。指定すると `{title}_slit{slit_index}.png` として保存する。
        title : str | None, optional
            プロットのタイトル。指定しない場合は `"Before_After"` を使用する。

        Returns
        -------
        np.ndarray
            Axes の 2D 配列（shape: (nrows, ncols)）
        """
        return ProcessingResult._plot_before_after(
            self.before, self.after, slit_index, labels, save_dir, title
        )


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
    o3_scale : float
        OIII λ4959 除去スケール係数（デフォルト: 1 / oiii5007_oiii4959 ≈ 1/2.98）
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
    o3_scale: float = 1.0 / oiii5007_oiii4959
    o3_half_width_aa: float = 40.0
    suffix: str = "_lac"
    extension: str = ".fits"
    depth: int = 1
    exclude_files: tuple[str, ...] = ()
    dq_flags: int = 16

    def __repr__(self) -> str:
        windows = ", ".join(
            f"({lo}, {hi})" for lo, hi in self.continuum_windows_kms
        )
        return (
            f"ProcessingPipeline(\n"
            f"  continuum_windows_kms=[{windows}] km/s,\n"
            f"  continuum_degree={self.continuum_degree},\n"
            f"  recession_velocity={self.recession_velocity} km/s,\n"
            f"  rest_wavelength={self.rest_wavelength:.3f} Å,\n"
            f"  v_min={self.v_min}, v_max={self.v_max} km/s,\n"
            f"  o3_scale={self.o3_scale}, o3_half_width_aa={self.o3_half_width_aa} Å,\n"
            f"  suffix={self.suffix!r}, extension={self.extension!r}, depth={self.depth},\n"
            f"  exclude_files={self.exclude_files}"
            f"  dq_flags={self.dq_flags},\n"
            f")"
        )

    def run(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        output_suffix: str = "_proc",
        overwrite: bool = False,
        run_x2d: bool = True,
        save_picture: bool = False,
        slit_index: int | None = None,
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
        save_picture : bool, optional
            True の場合、各処理ステップの確認プロットを output_dir に保存する（デフォルト: False）。
            slit_index 指定時のみ有効。保存されるファイル:
            - After_continuum_subtraction_slit{N}.png
            - Continuum_Fit_slit{N}.png
            - After_OIII_removal_slit{N}.png
            - After_velocity_range_clipping_slit{N}.png
        slit_index : int | None, optional
            確認プロットに使用するスリット行インデックス。
            save_picture=True のときのみ有効。

        Returns
        -------
        ProcessingResult
            処理前後の ImageCollection と出力パスリスト
        """
        input_dir = Path(input_dir)
        output_dir = self._resolve_output_dir(Path(output_dir), output_suffix)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. ファイル探索
        instrument = InstrumentModel.load(
            file_directory=str(input_dir),
            suffix=self.suffix,
            extension=self.extension,
            depth=self.depth,
            exclude_files=self.exclude_files,
        )
        lac_paths = instrument.path_list
        print(f"Step 1: Found {len(lac_paths)} files in {input_dir}")

        # 2. stistools.x2d による 2D 幾何補正
        if run_x2d:
            self._check_reference_files(lac_paths)
            x2d_paths = self._run_x2d_batch(lac_paths, output_dir)
            print(f"Step 2: x2d 完了 ({len(x2d_paths)} files)")
        else:
            x2d_paths = lac_paths
            print("Step 2: x2d スキップ（run_x2d=False）")

        # 3. ProcessingImageCollection として読み込み
        readers = ReaderCollection.from_paths(x2d_paths)
        before = ProcessingImageCollection.setup(
            readers,
            recession_velocity=self.recession_velocity,
            continuum_windows_kms=self.continuum_windows_kms,
            rest_wavelength=self.rest_wavelength,
            o3_half_width_aa=self.o3_half_width_aa,
            dq_flags=self.dq_flags,
        )
        print(f"Step 3: 読み込み完了 ({len(before.images)} images)")

        # 4. 連続光差し引き
        cont_subtracted = before.subtract_continuum(degree=self.continuum_degree)
        print("Step 4: 連続光差し引き完了")

        # 5. OIII λ4959 除去
        o3_removed = cont_subtracted.remove_o3_4959(scale=self.o3_scale)
        print("Step 5: OIII λ4959 除去完了")

        # 6. velocity range clipping
        clipped = o3_removed.clip_velocity_range(v_min=self.v_min, v_max=self.v_max)
        print(f"Step 6: velocity clipping 完了 ({self.v_min} ~ {self.v_max} km/s)")

        # 7. _proc.fits として書き出し
        output_paths = clipped.write_fits(
            output_suffix=output_suffix,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        for p in output_paths:
            print(f"  wrote: {p}")
        print(f"Step 7: 書き出し完了 ({len(output_paths)} files)")

        # 8. 確認用画像の保存
        if save_picture:
            if slit_index is None:
                print("Step 8: slit_index が未指定のためプロット保存をスキップします。")
            else:
                ProcessingResult._plot_before_after(
                    before, cont_subtracted, slit_index,
                    save_dir=output_dir, title="After_continuum_subtraction",
                )
                ProcessingResult._plot_continuum_fit(
                    before, cont_subtracted, slit_index,
                    save_dir=output_dir, title="Continuum_Fit",
                )
                ProcessingResult._plot_before_after(
                    cont_subtracted, o3_removed, slit_index,
                    save_dir=output_dir, title="After_OIII_removal",
                )
                ProcessingResult._plot_before_after(
                    o3_removed, clipped, slit_index,
                    save_dir=output_dir, title="After_velocity_range_clipping",
                )
                print("Step 8: プロット保存完了")

        return ProcessingResult(
            before=before,
            after=clipped,
            output_paths=output_paths,
        )

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
            出力ファイルの接尾辞（例: "_proc"）

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

    @staticmethod
    def _check_reference_files(lac_paths: list[Path]) -> None:
        """x2d に必要な参照ファイルの存在を確認する.

        FITSヘッダーから参照ファイルパスを読み取り、oref 環境変数のディレクトリと
        照合して不足ファイルを表示する。不足がある場合は RuntimeError を送出する。

        Parameters
        ----------
        lac_paths : list[Path]
            _lac.fits ファイルパスのリスト

        Raises
        ------
        RuntimeError
            oref 環境変数が未設定、または参照ファイルが不足している場合
        """
        import os

        from astropy.io import fits

        _REF_KEYWORDS = (
            "SDCTAB", "APDESTAB", "DISPTAB", "INANGTAB",
            "SPTRCTAB", "PHOTTAB", "APERTAB", "PCTAB", "TDSTAB",
        )

        oref_dir = os.environ.get("oref", "")
        if not oref_dir:
            raise RuntimeError(
                "環境変数 oref が設定されていません。\n"
                "~/.zshrc に以下を追記して source してください:\n"
                '  export oref="$HOME/crds_cache/references/hst/oref/"'
            )
        oref_path = Path(oref_dir)

        missing: list[str] = []
        for lac_path in lac_paths:
            with fits.open(lac_path) as hdul:
                header = hdul[0].header #type: ignore
            for kw in _REF_KEYWORDS:
                val = header.get(kw, "")
                if not val or val == "N/A":
                    continue
                filename = val.split("$")[-1]
                if not (oref_path / filename).exists():
                    entry = f"{lac_path.name}  {kw}: {filename}"
                    if entry not in missing:
                        missing.append(entry)

        if missing:
            lines = "\n  ".join(missing)
            raise RuntimeError(
                f"参照ファイルが不足しています ({len(missing)} 件):\n  {lines}\n\n"
                "以下のコマンドで取得してください:\n"
                "  crds sync --contexts hst_latest.pmap --fetch-references \\\n"
                + "    --files "
                + " ".join(
                    m.split(": ")[-1] for m in missing
                )
            )

    @staticmethod
    def _x2d_path_for(lac_path: Path, output_dir: Path) -> Path:
        """_lac.fits パスから x2d 出力パスを計算する.

        Parameters
        ----------
        lac_path : Path
            入力 _lac.fits ファイルパス
        output_dir : Path
            x2d 出力先ディレクトリ

        Returns
        -------
        Path
            対応する _x2d.fits の出力パス
        """
        return output_dir / lac_path.name.replace("_lac", "_x2d")

    @staticmethod
    def _run_x2d_batch(
        lac_paths: list[Path],
        output_dir: Path,
    ) -> list[Path]:
        """_lac.fits のリストに stistools.x2d を一括実行する.

        出力先に既存の _x2d.fits がある場合は x2d を実行せず、
        既存ファイルのパスをそのまま返す。

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
            out_path = ProcessingPipeline._x2d_path_for(lac_path, output_dir)
            if out_path.exists():
                print(f"  skip x2d (already exists): {out_path.name}")
            else:
                x2d_module.x2d(input=str(lac_path), output=str(out_path))
            x2d_paths.append(out_path)

        return x2d_paths
