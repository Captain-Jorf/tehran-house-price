"""Archive integrity checks; these do not verify publisher claims."""

import shutil

import pytest
from tehran_house_price.data.listing_archive import DATA, load_archive


def test_snapshot_counts_and_separate_measure():
    frame = load_archive()
    assert frame.groupby("source_id").size().to_dict() == {"104477": 39, "111526": 38, "147516": 54}
    assert frame.measure.unique().tolist() == ["asking_price"]
    assert frame.publication_date_jalali.nunique() == 3


def test_new_build_is_not_zero_age():
    frame = load_archive()
    new = frame[frame.age_raw == "نوساز"]
    assert len(new) == 19
    assert new.age_years.isna().all()


def test_current_snapshot_scope_and_source_table_conflict():
    frame = load_archive()
    current = frame[frame.source_id == "147516"]
    assert set(current.district_reported) == {8, 13, 14}
    assert current.age_years.between(10, 25).all()
    assert current[current.district_reported == 13].price_per_m2_toman.min() == 140_000_000
    assert current.publication_date_jalali.eq("1405-06-17").all()


def test_corruption_is_rejected(tmp_path):
    shutil.copytree(DATA, tmp_path / "archive")
    path = tmp_path / "archive/listings.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        load_archive(tmp_path / "archive")
