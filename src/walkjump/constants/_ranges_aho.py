from typing import Literal

LENGTH_FV_LIGHT_AHO = 148
LENGTH_FV_HEAVY_AHO = 149

LOOP_HEAVY = Literal["H1", "H2", "H3", "H4"]
FR_HEAVY = Literal["HFR1", "HFR2", "HFR3a", "HFR3b", "HFR4"]
LOOP_LIGHT = Literal["L1", "L2", "L3", "L4"]
FR_LIGHT = Literal["LFR1", "LFR2", "LFR3a", "LFR3b", "LFR4"]

REGION_HEAVY = LOOP_HEAVY | FR_HEAVY
REGION_LIGHT = LOOP_LIGHT | FR_LIGHT

REGION_AHO = REGION_HEAVY | REGION_LIGHT


CDR_RANGES_AHO = {
    "L1": (24, 42),
    "L2": (58, 72),
    "L3": (107, 138),
    "L4": (82, 90),
    "H1": (24, 42),
    "H2": (57, 76),
    "H3": (107, 138),
    "H4": (82, 90),
}

FR_RANGES_AHO = {
    "LFR1": (0, CDR_RANGES_AHO["L1"][0]),
    "LFR2": (CDR_RANGES_AHO["L1"][1], CDR_RANGES_AHO["L2"][0]),
    "LFR3a": (CDR_RANGES_AHO["L2"][1], CDR_RANGES_AHO["L4"][0]),
    "LFR3b": (CDR_RANGES_AHO["L4"][1], CDR_RANGES_AHO["L3"][0]),
    "LFR4": (CDR_RANGES_AHO["L3"][1], LENGTH_FV_LIGHT_AHO + 1),
    "HFR1": (0, CDR_RANGES_AHO["H1"][0]),
    "HFR2": (CDR_RANGES_AHO["H1"][1], CDR_RANGES_AHO["H2"][0]),
    "HFR3a": (CDR_RANGES_AHO["H2"][1], CDR_RANGES_AHO["H4"][0]),
    "HFR3b": (CDR_RANGES_AHO["H4"][1], CDR_RANGES_AHO["H3"][0]),
    "HFR4": (CDR_RANGES_AHO["H3"][1], LENGTH_FV_HEAVY_AHO + 1),
}

RANGES_AHO = CDR_RANGES_AHO | FR_RANGES_AHO
