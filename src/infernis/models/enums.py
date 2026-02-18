from enum import Enum


class DangerLevel(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"
    EXTREME = "EXTREME"

    @property
    def color(self) -> str:
        return {
            DangerLevel.VERY_LOW: "#22C55E",
            DangerLevel.LOW: "#3B82F6",
            DangerLevel.MODERATE: "#EAB308",
            DangerLevel.HIGH: "#F97316",
            DangerLevel.VERY_HIGH: "#EF4444",
            DangerLevel.EXTREME: "#1A0000",
        }[self]

    @classmethod
    def from_score(cls, score: float) -> "DangerLevel":
        if score < 0.05:
            return cls.VERY_LOW
        elif score < 0.15:
            return cls.LOW
        elif score < 0.35:
            return cls.MODERATE
        elif score < 0.60:
            return cls.HIGH
        elif score < 0.80:
            return cls.VERY_HIGH
        else:
            return cls.EXTREME


class FuelType(str, Enum):
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"
    C6 = "C6"
    C7 = "C7"
    D1 = "D1"
    D2 = "D2"
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M4 = "M4"
    O1A = "O1A"
    O1B = "O1B"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    NON_FUEL = "NF"
    WATER = "WA"


class BECZone(str, Enum):
    AT = "AT"
    BG = "BG"
    BWBS = "BWBS"
    CDF = "CDF"
    CWH = "CWH"
    ESSF = "ESSF"
    ICH = "ICH"
    IDF = "IDF"
    MH = "MH"
    MS = "MS"
    PP = "PP"
    SBPS = "SBPS"
    SBS = "SBS"
    SWB = "SWB"
