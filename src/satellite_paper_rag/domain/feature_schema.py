from __future__ import annotations

from typing import Mapping


SLSTR_SENSOR_NAME = "Sentinel-3 SLSTR"


def rag_band_schema(band_features: Mapping[str, object]) -> dict[str, dict[str, str]]:
    schema: dict[str, dict[str, str]] = {}
    for band_name in band_features:
        normalized = band_name.upper()
        if normalized in {f"S{index}" for index in range(1, 7)}:
            schema[normalized] = {
                "sensor": SLSTR_SENSOR_NAME,
                "source_file": f"{normalized}_radiance_in.nc",
                "physical_quantity": "radiance",
                "unit": "unknown",
                "processing": "xarray-decoded NetCDF value; no manual conversion in CSV builder",
                "constraint": "do not interpret as reflectance without a documented conversion",
            }
        elif normalized in {"S7", "S8", "S9"}:
            schema[normalized] = {
                "sensor": SLSTR_SENSOR_NAME,
                "source_file": f"{normalized}_BT_in.nc",
                "physical_quantity": "brightness_temperature",
                "unit": "unknown",
                "processing": "xarray-decoded NetCDF value; no manual conversion in CSV builder",
                "constraint": "do not apply a numeric threshold until the NetCDF unit is confirmed",
            }
    return schema
