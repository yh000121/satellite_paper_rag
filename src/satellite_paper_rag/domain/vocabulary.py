from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainVocabulary:
    aliases: dict[str, str]
    sensor_band_map: dict[str, str]

    @classmethod
    def default(cls) -> "DomainVocabulary":
        aliases = {
            "bt": "thermal_brightness_temperature",
            "brightness temperature": "thermal_brightness_temperature",
            "thermal band": "thermal_brightness_temperature",
            "thermal channel": "thermal_brightness_temperature",
            "clear sky probability": "clear_sky_probability",
            "clear-sky probability": "clear_sky_probability",
            "clear_sky_probability": "clear_sky_probability",
            "lsd": "local_standard_deviation",
            "local standard deviation": "local_standard_deviation",
            "local standard deviation of brightness temperature": "local_standard_deviation",
            "rho": "rho",
            "rho value": "rho",
            "rho parameter": "rho",
            "ρ": "rho",
            "visible": "visible_reflectance",
            "visible reflectance": "visible_reflectance",
            "reflectance": "reflectance",
            "ndvi": "NDVI",
            "ndwi": "NDWI",
            "ndsi": "NDSI",
            "cloud": "cloud",
            "cloud-contaminated": "cloud",
            "sea ice": "sea_ice",
            "ice": "sea_ice",
            "open water": "open_water",
            "water": "open_water",
            "sea": "open_water",
            "land": "land",
            "vegetation": "vegetation",
            "snow": "snow",
            "sentinel-3": "Sentinel-3",
            "sentinel-2": "Sentinel-2",
            "landsat-8": "Landsat-8",
            "modis": "MODIS",
            "slstr": "SLSTR",
            "msi": "MSI",
            "oli": "OLI",
            "tirs": "TIRS",
        }
        sensor_band_map = {
            "sentinel-3/slstr/s1": "visible_reflectance",
            "sentinel-3/slstr/s2": "visible_reflectance",
            "sentinel-3/slstr/s3": "visible_reflectance",
            "sentinel-3/slstr/s4": "visible_reflectance",
            "sentinel-3/slstr/s5": "visible_reflectance",
            "sentinel-3/slstr/s6": "visible_reflectance",
            "sentinel-3/slstr/s7": "thermal_brightness_temperature",
            "sentinel-3/slstr/s8": "thermal_brightness_temperature",
            "sentinel-3/slstr/s9": "thermal_brightness_temperature",
            "landsat-8/tirs/b10": "thermal_brightness_temperature",
            "landsat-8/tirs/b11": "thermal_brightness_temperature",
        }
        return cls(aliases=aliases, sensor_band_map=sensor_band_map)

    def normalize_alias(self, value: str) -> str | None:
        return self.aliases.get(value.strip().lower())

    def normalize_band(self, satellite: str, sensor: str, band: str) -> str | None:
        key = f"{satellite}/{sensor}/{band}".strip().lower()
        return self.sensor_band_map.get(key)
