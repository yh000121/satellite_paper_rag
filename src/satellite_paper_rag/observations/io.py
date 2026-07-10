from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from satellite_paper_rag.observations.schema import ObservationSample


ID_FIELDS = ("point_id", "sample_id", "id")
METADATA_FIELDS = {
    "label",
    "image_id",
    "row",
    "col",
    "predicted_label",
    "predicted_class",
    "confidence",
    "top_features",
}
CONTROL_FIELDS = {"point_id", "sample_id", "id", "source_type", "satellite", "sensor", "metadata", *METADATA_FIELDS}
LABEL_ID_TO_NAME = {"0": "sea", "1": "ice", "2": "cloud"}
KNOWN_LABEL_NAMES = {"sea", "ice", "cloud"}


def load_observations(path: Path) -> list[ObservationSample]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json(path)
    if suffix == ".csv":
        return _load_csv(path)
    raise ValueError(f"Unsupported observation file type: {path.suffix}")


def _load_json(path: Path) -> list[ObservationSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("observations")
        if rows is None:
            rows = [payload]
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError("JSON observation input must be a list or an object with an observations list.")
    return [_row_to_sample(row, index, "json") for index, row in enumerate(rows, start=1)]


def _load_csv(path: Path) -> list[ObservationSample]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [_row_to_sample(row, index, "csv") for index, row in enumerate(rows, start=1)]


def _row_to_sample(row: Any, index: int, source_type: str) -> ObservationSample:
    if not isinstance(row, dict):
        raise ValueError("Each observation row must be an object.")
    sample_id = _sample_id(row, index)
    features = {
        str(key): _parse_scalar(value)
        for key, value in row.items()
        if key not in CONTROL_FIELDS and value not in (None, "")
    }
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    row_metadata = {
        str(key): str(value)
        for key, value in row.items()
        if key in METADATA_FIELDS and value not in (None, "")
    }
    _add_label_metadata(row_metadata)
    _add_prediction_metadata(row_metadata)
    row_metadata.update({str(key): str(value) for key, value in metadata.items()})
    return ObservationSample(
        sample_id=sample_id,
        source_type=str(row.get("source_type") or source_type),
        satellite=_optional_string(row.get("satellite")),
        sensor=_optional_string(row.get("sensor")),
        features=features,
        metadata=row_metadata,
    )


def _sample_id(row: dict[str, Any], index: int) -> str:
    for field in ID_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    image_id = row.get("image_id")
    row_index = row.get("row")
    col_index = row.get("col")
    if image_id not in (None, "") and row_index not in (None, "") and col_index not in (None, ""):
        return f"{image_id}:{row_index}:{col_index}"
    return f"row_{index:04d}"


def _optional_string(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _parse_scalar(value: object) -> float | str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return str(value)


def _add_label_metadata(metadata: dict[str, str]) -> None:
    raw_label = metadata.get("label")
    if raw_label in (None, ""):
        return
    metadata["label_id"] = raw_label
    normalized = raw_label.strip().lower()
    if normalized in LABEL_ID_TO_NAME:
        metadata["label_name"] = LABEL_ID_TO_NAME[normalized]
    elif normalized in KNOWN_LABEL_NAMES:
        metadata["label_name"] = normalized


def _add_prediction_metadata(metadata: dict[str, str]) -> None:
    raw_prediction = metadata.get("predicted_label") or metadata.get("predicted_class")
    if raw_prediction in (None, ""):
        return
    normalized = raw_prediction.strip().lower()
    if normalized in LABEL_ID_TO_NAME:
        metadata["predicted_label_id"] = normalized
        metadata["predicted_class"] = LABEL_ID_TO_NAME[normalized]
    elif normalized in KNOWN_LABEL_NAMES:
        metadata["predicted_class"] = normalized
