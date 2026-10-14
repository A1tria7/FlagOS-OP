from __future__ import annotations

from pathlib import Path

import yaml

from scripts.build_ops_index import (
    OUTPUT_PATH,
    load_catalog,
    render_index,
    validate_catalog,
)


ROOT = Path(__file__).resolve().parents[2]


def test_catalog_models_six_incremental_platform_families():
    catalog = load_catalog()
    assert catalog["target_scope"]["vendor_family_count"] == 6
    assert set(catalog["platform_families"]) == {
        "ascend", "cambricon", "kunlunxin", "thead", "enflame", "biren"
    }

    planned = {
        name: meta
        for name, meta in catalog["platform_families"].items()
        if meta["status"] == "planned"
    }
    assert set(planned) == {"thead", "enflame", "biren"}
    assert all(not meta["current_profiles"] for meta in planned.values())


def test_catalog_is_valid_and_covers_every_operator_directory():
    catalog = load_catalog()
    validate_catalog(catalog)

    actual = {
        path.name for path in (ROOT / "ops").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    cataloged = {op["directory"] for op in catalog["operators"]}
    assert cataloged == actual


def test_generated_ops_index_is_current():
    catalog = load_catalog()
    validate_catalog(catalog)
    assert OUTPUT_PATH.read_text() == render_index(catalog)


def test_platform_claims_are_incremental_and_backed_by_profiles():
    catalog = load_catalog()
    operators_by_family = {
        family: [
            op for op in catalog["operators"] if family in op["platforms"]
        ]
        for family in catalog["platform_families"]
    }

    for family, meta in catalog["platform_families"].items():
        if meta["status"] == "planned":
            assert operators_by_family[family] == []
        else:
            assert operators_by_family[family]
            assert meta["current_profiles"]


def test_catalog_yaml_is_parseable_and_versioned():
    data = yaml.safe_load((ROOT / "ops/catalog.yaml").read_text())
    assert data["schema"] == 1
