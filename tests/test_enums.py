"""Tests for domain enums."""

from infernis.models.enums import BECZone, DangerLevel, FuelType


class TestDangerLevel:
    def test_from_score_boundaries(self):
        assert DangerLevel.from_score(0.0) == DangerLevel.VERY_LOW
        assert DangerLevel.from_score(0.04) == DangerLevel.VERY_LOW
        assert DangerLevel.from_score(0.05) == DangerLevel.LOW
        assert DangerLevel.from_score(0.14) == DangerLevel.LOW
        assert DangerLevel.from_score(0.15) == DangerLevel.MODERATE
        assert DangerLevel.from_score(0.34) == DangerLevel.MODERATE
        assert DangerLevel.from_score(0.35) == DangerLevel.HIGH
        assert DangerLevel.from_score(0.59) == DangerLevel.HIGH
        assert DangerLevel.from_score(0.60) == DangerLevel.VERY_HIGH
        assert DangerLevel.from_score(0.79) == DangerLevel.VERY_HIGH
        assert DangerLevel.from_score(0.80) == DangerLevel.EXTREME
        assert DangerLevel.from_score(1.0) == DangerLevel.EXTREME

    def test_color_property(self):
        assert DangerLevel.VERY_LOW.color == "#22C55E"
        assert DangerLevel.EXTREME.color == "#1A0000"

    def test_all_levels_have_colors(self):
        for level in DangerLevel:
            assert level.color.startswith("#")


class TestFuelType:
    def test_conifer_types(self):
        assert FuelType.C1.value == "C1"
        assert FuelType.C7.value == "C7"

    def test_special_types(self):
        assert FuelType.NON_FUEL.value == "NF"
        assert FuelType.WATER.value == "WA"

    def test_count(self):
        assert len(FuelType) == 20


class TestBECZone:
    def test_all_14_zones(self):
        assert len(BECZone) == 14

    def test_interior_zones(self):
        assert BECZone.IDF.value == "IDF"
        assert BECZone.SBPS.value == "SBPS"
