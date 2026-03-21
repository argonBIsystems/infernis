"""Tests for BEC_ZONE_NAMES mapping in enums.py."""

from infernis.models.enums import BECZone, BEC_ZONE_NAMES


def test_all_bec_zone_values_have_names():
    """All 14 BECZone enum values must have entries in BEC_ZONE_NAMES."""
    for zone in BECZone:
        assert zone.value in BEC_ZONE_NAMES, (
            f"Missing name for BECZone.{zone.name} ({zone.value!r})"
        )


def test_spot_check_idf():
    assert BEC_ZONE_NAMES["IDF"] == "Interior Douglas-fir"


def test_spot_check_cwh():
    assert BEC_ZONE_NAMES["CWH"] == "Coastal Western Hemlock"


def test_spot_check_essf():
    assert BEC_ZONE_NAMES["ESSF"] == "Engelmann Spruce-Subalpine Fir"


def test_spot_check_sbs():
    assert BEC_ZONE_NAMES["SBS"] == "Sub-Boreal Spruce"


def test_exactly_14_entries():
    assert len(BEC_ZONE_NAMES) == 14
