"""InstrumentModel のテスト."""

from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits

from stis_analysis.core.instrument import InstrumentModel


def _write_dummy_fits(path: Path) -> None:
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=np.ones((10, 20), dtype=np.float32)),
        fits.ImageHDU(data=np.ones((10, 20), dtype=np.float32)),
        fits.ImageHDU(data=np.zeros((10, 20), dtype=np.int16)),
    ])
    hdul.writeto(path)


@pytest.fixture
def crj_dir(tmp_path: Path) -> Path:
    """depth=1 で _crj.fits ファイルが 3 つある構造を作成する."""
    subdir = tmp_path / "obs"
    subdir.mkdir()
    for name in ("a_crj.fits", "b_crj.fits", "c_crj.fits"):
        _write_dummy_fits(subdir / name)
    return tmp_path


class TestInstrumentModelLoad:
    def test_load_returns_instance(self, crj_dir: Path):
        inst = InstrumentModel.load(
            str(crj_dir), suffix="_crj", extension=".fits"
        )
        assert isinstance(inst, InstrumentModel)

    def test_load_sets_defaults(self, crj_dir: Path):
        inst = InstrumentModel.load(str(crj_dir))
        assert inst.depth == 1
        assert inst.exclude_files == ()

    def test_direct_construction(self, crj_dir: Path):
        inst = InstrumentModel(
            file_directory=str(crj_dir),
            suffix="_crj",
            extension=".fits",
        )
        assert inst.suffix == "_crj"


class TestInstrumentModelPathList:
    def test_path_list_finds_files(self, crj_dir: Path):
        inst = InstrumentModel.load(
            str(crj_dir), suffix="_crj", extension=".fits"
        )
        assert len(inst.path_list) == 3

    def test_path_list_is_sorted(self, crj_dir: Path):
        inst = InstrumentModel.load(
            str(crj_dir), suffix="_crj", extension=".fits"
        )
        paths = inst.path_list
        assert paths == sorted(paths)

    def test_path_list_returns_path_objects(self, crj_dir: Path):
        inst = InstrumentModel.load(
            str(crj_dir), suffix="_crj", extension=".fits"
        )
        for p in inst.path_list:
            assert isinstance(p, Path)

    def test_path_list_empty_when_no_match(self, crj_dir: Path):
        inst = InstrumentModel.load(
            str(crj_dir), suffix="_lac", extension=".fits"
        )
        assert inst.path_list == []

    def test_exclude_files_by_stem(self, crj_dir: Path):
        inst = InstrumentModel.load(
            str(crj_dir),
            suffix="_crj",
            extension=".fits",
            exclude_files=("a_crj",),
        )
        names = [p.name for p in inst.path_list]
        assert "a_crj.fits" not in names
        assert len(names) == 2

    def test_exclude_files_by_fullname(self, crj_dir: Path):
        inst = InstrumentModel.load(
            str(crj_dir),
            suffix="_crj",
            extension=".fits",
            exclude_files=("b_crj.fits",),
        )
        names = [p.name for p in inst.path_list]
        assert "b_crj.fits" not in names
        assert len(names) == 2

    def test_depth_controls_search_level(self, tmp_path: Path):
        """depth=2 で2階層下のファイルを検索できること."""
        deep = tmp_path / "level1" / "level2"
        deep.mkdir(parents=True)
        _write_dummy_fits(deep / "x_crj.fits")

        inst_depth1 = InstrumentModel.load(
            str(tmp_path), suffix="_crj", extension=".fits", depth=1
        )
        inst_depth2 = InstrumentModel.load(
            str(tmp_path), suffix="_crj", extension=".fits", depth=2
        )
        assert len(inst_depth1.path_list) == 0
        assert len(inst_depth2.path_list) == 1
