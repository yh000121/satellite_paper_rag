from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservationSample:
    sample_id: str
    source_type: str
    satellite: str | None
    sensor: str | None
    features: dict[str, float | str]
    normalized_features: dict[str, float | str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
