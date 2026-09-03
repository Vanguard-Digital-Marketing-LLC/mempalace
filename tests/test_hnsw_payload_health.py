import os
from pathlib import Path

from mempalace.backends.chroma import (
    _HNSW_LINK_TO_DATA_MAX_RATIO,
    _hnsw_link_to_data_ratio,
    _segment_appears_healthy,
    quarantine_stale_hnsw,
)


def _write_segment(
    seg_dir: Path,
    *,
    data_size: int = 100,
    link_size: int = 100,
    write_metadata: bool = True,
) -> None:
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "data_level0.bin").write_bytes(b"\0" * data_size)
    (seg_dir / "link_lists.bin").write_bytes(b"\0" * link_size)

    if write_metadata:
        # Enough bytes to pass the existing pickle envelope sniff-test:
        # starts with pickle protocol marker 0x80 and ends with STOP 0x2e.
        (seg_dir / "index_metadata.pickle").write_bytes(b"\x80" + b"x" * 16 + b"\x2e")


def test_hnsw_link_to_data_ratio_reports_payload_size_ratio(tmp_path):
    seg_dir = tmp_path / "11111111-2222-3333-4444-555555555555"
    _write_segment(seg_dir, data_size=100, link_size=250)

    assert _hnsw_link_to_data_ratio(str(seg_dir)) == 2.5


def test_segment_health_rejects_exploded_link_lists_even_with_valid_pickle(tmp_path):
    seg_dir = tmp_path / "11111111-2222-3333-4444-555555555555"
    _write_segment(
        seg_dir,
        data_size=100,
        link_size=int(100 * (_HNSW_LINK_TO_DATA_MAX_RATIO + 1)),
        write_metadata=True,
    )

    assert not _segment_appears_healthy(str(seg_dir))


def test_segment_health_keeps_reasonable_payload_with_valid_pickle(tmp_path):
    seg_dir = tmp_path / "11111111-2222-3333-4444-555555555555"
    _write_segment(
        seg_dir,
        data_size=100,
        link_size=int(100 * _HNSW_LINK_TO_DATA_MAX_RATIO),
        write_metadata=True,
    )

    assert _segment_appears_healthy(str(seg_dir))


def test_quarantine_catches_link_bloat_without_mtime_drift(tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()

    db_path = palace / "chroma.sqlite3"
    db_path.write_text("sqlite placeholder")

    seg_dir = palace / "11111111-2222-3333-4444-555555555555"
    _write_segment(
        seg_dir,
        data_size=100,
        link_size=int(100 * (_HNSW_LINK_TO_DATA_MAX_RATIO + 1)),
        write_metadata=True,
    )

    # Make sqlite and HNSW mtimes identical. The old mtime-only gate would
    # skip this segment even though the payload is structurally corrupt.
    same_time = 1_700_000_000
    os.utime(db_path, (same_time, same_time))
    os.utime(seg_dir / "data_level0.bin", (same_time, same_time))

    moved = quarantine_stale_hnsw(str(palace), stale_seconds=999_999)

    assert len(moved) == 1
    assert not seg_dir.exists()

    moved_path = Path(moved[0])
    assert moved_path.exists()
    assert moved_path.name.startswith("11111111-2222-3333-4444-555555555555.drift-")


def test_quarantine_leaves_reasonable_payload_in_place(tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()

    db_path = palace / "chroma.sqlite3"
    db_path.write_text("sqlite placeholder")

    seg_dir = palace / "11111111-2222-3333-4444-555555555555"
    _write_segment(
        seg_dir,
        data_size=100,
        link_size=100,
        write_metadata=True,
    )

    same_time = 1_700_000_000
    os.utime(db_path, (same_time, same_time))
    os.utime(seg_dir / "data_level0.bin", (same_time, same_time))

    moved = quarantine_stale_hnsw(str(palace), stale_seconds=999_999)

    assert moved == []
    assert seg_dir.exists()


def test_segment_health_rejects_zero_byte_link_lists_with_payload(tmp_path):
    """Regression #1457: real HNSW payload with empty link_lists.bin is corrupt."""
    seg_dir = tmp_path / "11111111-2222-3333-4444-555555555555"

    _write_segment(
        seg_dir,
        data_size=2_000,
        link_size=0,
        write_metadata=True,
    )

    assert not _segment_appears_healthy(str(seg_dir))


def test_quarantine_catches_zero_byte_link_lists_when_stale(tmp_path):
    """Regression #1457: stale segments with empty link_lists.bin are quarantined."""
    palace = tmp_path / "palace"
    palace.mkdir()

    db_path = palace / "chroma.sqlite3"
    db_path.write_text("sqlite placeholder")

    seg_dir = palace / "11111111-2222-3333-4444-555555555555"
    _write_segment(
        seg_dir,
        data_size=2_000,
        link_size=0,
        write_metadata=True,
    )

    hnsw_time = 1_700_000_000
    sqlite_time = hnsw_time + 1_000
    os.utime(seg_dir / "data_level0.bin", (hnsw_time, hnsw_time))
    os.utime(db_path, (sqlite_time, sqlite_time))

    moved = quarantine_stale_hnsw(str(palace), stale_seconds=300)

    assert len(moved) == 1
    assert not seg_dir.exists()

    moved_path = Path(moved[0])
    assert moved_path.exists()
    assert moved_path.name.startswith("11111111-2222-3333-4444-555555555555.drift-")


# ── total_elements_added is cumulative, not a live count ──────────────
#
# Regression tests for the false-quarantine that cost a 183,009-vector
# index. hnswlib's ``total_elements_added`` counts every add the segment
# has ever made, including ones later deleted or replaced, so it is
# legitimately GREATER than ``len(id_to_label)`` on any palace that has
# ever removed a drawer. Demanding equality condemned a healthy index.


def _write_pickled_segment(seg_dir: Path, state: dict, *, payload: int = 4096) -> None:
    """Write a segment whose metadata pickle carries ``state``."""
    import pickle

    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "data_level0.bin").write_bytes(b"\0" * payload)
    (seg_dir / "link_lists.bin").write_bytes(b"\0" * (payload // 8))
    with open(seg_dir / "index_metadata.pickle", "wb") as f:
        pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)


def _state(*, labels: int, total: int, dimensionality=None) -> dict:
    return {
        "dimensionality": dimensionality,
        "total_elements_added": total,
        "max_seq_id": None,
        "id_to_label": {f"d-{i}": i for i in range(labels)},
        "label_to_id": {i: f"d-{i}" for i in range(labels)},
        "id_to_seq_id": {},
    }


def test_missing_dimensionality_recoverable_when_total_exceeds_labels(tmp_path):
    """Deletions make total_elements_added > label count. Still recoverable."""
    from mempalace.backends.chroma import _missing_dimensionality_appears_recoverable

    seg_dir = tmp_path / "11111111-2222-3333-4444-555555555555"
    state = _state(labels=183_009, total=188_220)
    _write_pickled_segment(seg_dir, state)

    assert _missing_dimensionality_appears_recoverable(state, state["id_to_label"], str(seg_dir))


def test_missing_dimensionality_not_recoverable_when_total_below_labels(tmp_path):
    """total < labels is genuinely impossible - stay unrecoverable."""
    from mempalace.backends.chroma import _missing_dimensionality_appears_recoverable

    seg_dir = tmp_path / "11111111-2222-3333-4444-555555555555"
    state = _state(labels=100, total=40)
    _write_pickled_segment(seg_dir, state)

    assert not _missing_dimensionality_appears_recoverable(
        state, state["id_to_label"], str(seg_dir)
    )


def test_quarantine_spares_healthy_index_with_cumulative_total(tmp_path):
    """End-to-end: the 2026-08-22 shape must NOT be quarantined."""
    from mempalace.backends.chroma import quarantine_invalid_hnsw_metadata

    seg = "11111111-2222-3333-4444-555555555555"
    seg_dir = tmp_path / seg
    _write_pickled_segment(seg_dir, _state(labels=183_009, total=188_220))

    assert quarantine_invalid_hnsw_metadata(str(tmp_path)) == []
    assert seg_dir.is_dir(), "healthy index was quarantined"


def test_quarantine_still_catches_inconsistent_label_maps(tmp_path):
    """Relaxing the counter check must not spare a truly broken index."""
    from mempalace.backends.chroma import quarantine_invalid_hnsw_metadata

    seg = "11111111-2222-3333-4444-555555555555"
    seg_dir = tmp_path / seg
    state = _state(labels=100, total=120)
    state["label_to_id"] = {i: f"WRONG-{i}" for i in range(100)}
    _write_pickled_segment(seg_dir, state)

    moved = quarantine_invalid_hnsw_metadata(str(tmp_path))
    assert len(moved) == 1
    assert not seg_dir.is_dir()
