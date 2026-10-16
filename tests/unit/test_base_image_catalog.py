from __future__ import annotations

from copy import deepcopy

import pytest


def test_base_image_catalog_is_valid_and_generated_index_is_current():
    from scripts.build_base_image_index import (
        OUTPUT_PATH,
        load_catalog,
        render_index,
        validate_catalog,
    )

    catalog = load_catalog()
    validate_catalog(catalog)
    assert OUTPUT_PATH.read_text(encoding="utf-8") == render_index(catalog)


def test_base_image_source_is_pinned_and_index_covers_all_families():
    from scripts.build_base_image_index import (
        load_catalog,
        render_index,
        validate_catalog,
    )
    from scripts.build_ops_index import load_catalog as load_operator_catalog

    catalog = load_catalog()
    validate_catalog(catalog)

    assert catalog["source"]["revision"].startswith("74197bf7d4b5dc8cdd3b2b89")
    assert catalog["source"]["revision"] in catalog["source"]["url"]
    assert {
        image["family"] for image in catalog["images"]
    } <= set(load_operator_catalog()["platform_families"])

    rendered = render_index(catalog)
    for family in load_operator_catalog()["platform_families"].values():
        assert family["display_name"] in rendered


def test_base_image_catalog_distinguishes_version_alignment_from_validation():
    from scripts.build_base_image_index import load_catalog, validate_catalog

    catalog = load_catalog()
    validate_catalog(catalog)
    p800 = next(
        image for image in catalog["images"]
        if image["id"] == "kunlunxin-p800-operator-runtime-v2"
    )

    assert p800["flagos_op_relation"] == "version-aligned"
    assert p800["related_profiles"] == ["p800-kunlunxin"]
    assert any("同镜像实例" in note for note in p800["notes"])


def test_historical_base_image_requires_successor_or_reason():
    from scripts.build_base_image_index import load_catalog, validate_catalog

    catalog = deepcopy(load_catalog())
    catalog["images"][0]["current"] = False
    with pytest.raises(ValueError, match="historical entries need"):
        validate_catalog(catalog)
