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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np

from stis_analysis.core.fits_reader import ReaderCollection
from stis_analysis.core.instrument import InstrumentModel
from stis_analysis.processing.image import ProcessingImageCollection, ProcessingImageModel
from stis_analysis.processing.wave_constants import c_kms, oiii5007_stp, oiii5007_oiii4959


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

    def plot_continuum_fit(
        self,
        slit_index: int,
        continuum_windows_kms: list[tuple[float, float]],
        recession_velocity: float,
        rest_wavelength: float = oiii5007_stp,
        degree: int = 1,
        which: Literal["before", "after"] = "before",
        o3_half_width_aa: float | None = None,
        save_dir: Path | str | None = None,
    ) -> np.ndarray:
        """連続光フィットを 3 列タイルで確認するプロットを生成する.

        Parameters
        ----------
        slit_index : int
            確認するスリット行のインデックス（空間方向）
        continuum_windows_kms : list[tuple[float, float]]
            連続光ウィンドウ [km/s]
        recession_velocity : float
            銀河の後退速度 [km/s]
        rest_wavelength : float, optional
            基準静止波長 [Å]。デフォルト: oiii5007_stp
        degree : int, optional
            フィット多項式次数（デフォルト: 1）
        which : {"before", "after"}, optional
            プロットする画像コレクション（デフォルト: "before"）。
            "before" は連続光差し引き前、"after" は全処理適用後。
        o3_half_width_aa : float | None, optional
            OIII λ4959 除去領域の半幅 [Å]。指定すると除去対象範囲を
            シェードで表示する。None の場合は非表示（デフォルト）。
        save_dir : Path | str | None, optional
            保存先ディレクトリ。指定すると `continuum_fit_slit{slit_index}_{which}.png` として保存する。

        Returns
        -------
        np.ndarray
            Axes の 2D 配列（shape: (nrows, ncols)）
        """
        collection = self.before if which == "before" else self.after
        n = len(collection.images)
        if n == 0:
            raise ValueError(f"'{which}' の画像が存在しません。")

        ncols = min(n, 3)
        nrows = math.ceil(n / ncols)

        fig, axes_2d = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )
        axes_list = list(axes_2d.flat)

        for ax, image in zip(axes_list, collection.images):
            cast(ProcessingImageModel, image).plot_continuum_fit(
                slit_index=slit_index,
                continuum_windows_kms=continuum_windows_kms,
                recession_velocity=recession_velocity,
                rest_wavelength=rest_wavelength,
                degree=degree,
                o3_half_width_aa=o3_half_width_aa,
                ax=ax,
            )

        for ax in axes_list[n:]:
            ax.set_visible(False)

        plt.tight_layout()

        if save_dir is not None:
            fig.savefig(Path(save_dir) / f"continuum_fit_slit{slit_index}_{which}.png")

        plt.show()
        return axes_2d

    def plot_before_after(
        self,
        slit_index: int,
        recession_velocity: float,
        rest_wavelength: float = oiii5007_stp,
        labels: tuple[str, str] = ("before", "after"),
        save_dir: Path | str | None = None,
    ) -> np.ndarray:
        """処理前後のスペクトルを重ねて比較するプロットを生成する.

        before と after のスペクトルを同一 Axes に重ねて描画し、
        3 列タイルで全画像を並べる。

        Parameters
        ----------
        slit_index : int
            確認するスリット行のインデックス（空間方向）
        recession_velocity : float
            銀河の後退速度 [km/s]
        rest_wavelength : float, optional
            速度 v=0 の基準静止波長 [Å]。デフォルト: oiii5007_stp
        labels : tuple[str, str], optional
            凡例ラベル（before, after の順）
        save_dir : Path | str | None, optional
            保存先ディレクトリ。指定すると `before_after_slit{slit_index}.png` として保存する。

        Returns
        -------
        np.ndarray
            Axes の 2D 配列（shape: (nrows, ncols)）
        """
        n = len(self.before.images)
        if n == 0:
            raise ValueError("画像が存在しません。")

        ncols = min(n, 3)
        nrows = math.ceil(n / ncols)

        z = recession_velocity / c_kms
        lambda_ref = rest_wavelength * (1.0 + z)

        fig, axes_2d = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
        )
        axes_list = list(axes_2d.flat)

        for ax, img_before, img_after in zip(
            axes_list, self.before.images, self.after.images
        ):
            wl_before = img_before.sci.wavelength
            if wl_before is not None:
                vel_before = c_kms * (wl_before / lambda_ref - 1.0)
                ax.plot(vel_before, img_before.sci.data[slit_index, :],
                        color="steelblue", lw=0.8, alpha=0.8, label=labels[0])

            wl_after = img_after.sci.wavelength
            if wl_after is not None:
                vel_after = c_kms * (wl_after / lambda_ref - 1.0)
                ax.plot(vel_after, img_after.sci.data[slit_index, :],
                        color="tomato", lw=0.8, alpha=0.8, label=labels[1])

            ax.set_xlabel("Velocity [km/s]")
            ax.set_ylabel("Counts")
            ax.set_title(
                f"{img_before.filename} (slit={slit_index}, "
                f"v_reces={recession_velocity} km/s)"
            )
            ax.legend(fontsize="small")

        for ax in axes_list[n:]:
            ax.set_visible(False)

        plt.tight_layout()

        if save_dir is not None:
            fig.savefig(Path(save_dir) / f"before_after_slit{slit_index}.png")

        plt.show()
        return axes_2d


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
    o3_half_width_aa: float = 30.0
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
        before = ProcessingImageCollection.from_readers(readers, dq_flags=self.dq_flags)
        print(f"Step 3: 読み込み完了 ({len(before.images)} images)")

        # 4. 連続光差し引き
        after = before.subtract_continuum(
            continuum_windows_kms=self.continuum_windows_kms,
            recession_velocity=self.recession_velocity,
            rest_wavelength=self.rest_wavelength,
            degree=self.continuum_degree,
        )
        print("Step 4: 連続光差し引き完了")

        # 5. OIII λ4959 除去
        after = after.remove_o3_4959(
            recession_velocity=self.recession_velocity,
            scale=self.o3_scale,
            half_width_aa=self.o3_half_width_aa,
        )
        print("Step 5: OIII λ4959 除去完了")

        # 6. velocity range clipping
        after = after.clip_velocity_range(
            v_min=self.v_min,
            v_max=self.v_max,
            recession_velocity=self.recession_velocity,
            rest_wavelength=self.rest_wavelength,
        )
        print(f"Step 6: velocity clipping 完了 ({self.v_min} ~ {self.v_max} km/s)")

        # 7. _proc.fits として書き出し
        output_paths = after.write_fits(
            output_suffix=output_suffix,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        for p in output_paths:
            print(f"  wrote: {p}")
        print(f"Step 7: 書き出し完了 ({len(output_paths)} files)")

        return ProcessingResult(
            before=before,
            after=after,
            output_paths=output_paths,
        )

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
