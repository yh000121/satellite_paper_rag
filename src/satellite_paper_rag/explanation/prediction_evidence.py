from __future__ import annotations

from satellite_paper_rag.domain.feature_schema import rag_band_schema
from satellite_paper_rag.observations.schema import ObservationSample


CLASS_BASE_QUERIES = {
    "cloud": [
        "cloud detection SLSTR S1 S6 radiance S7 S9 brightness temperature",
        "cloud mask S7 S8 brightness temperature threshold",
        "cloud ice ocean solar radiance thermal channels",
    ],
    "ice": [
        "sea ice SLSTR S1 S6 radiance brighter ocean",
        "ice cloud ocean radiance brightness temperature",
        "sea ice boundary cloud detection solar radiance",
    ],
    "sea": [
        "open ocean clear-sky SLSTR brightness temperature",
        "darker ocean S1 S6 radiance cloud ice",
        "clear-sky pixels S7 S8 brightness temperature 1:1 line",
    ],
}
RAG_BAND_NAMES = tuple(f"S{index}" for index in range(1, 10))


def build_prediction_evidence_queries(observation: ObservationSample) -> list[dict[str, str]]:
    predicted_class = _evidence_class(observation)
    if not predicted_class:
        raise ValueError("Prediction evidence retrieval requires label or predicted_class metadata.")
    queries = [
        {
            "kind": "class_evidence",
            "prediction_class": predicted_class,
            "query": query,
        }
        for query in CLASS_BASE_QUERIES.get(predicted_class, [f"{predicted_class} satellite classification evidence"])
    ]
    queries.extend(_feature_queries(predicted_class, observation.metadata.get("top_features", "")))
    return _dedupe_queries(queries)


def prediction_summary(observation: ObservationSample) -> dict[str, object]:
    top_features = _rag_top_features(observation.metadata.get("top_features", ""))
    band_features = _rag_band_features(observation)
    metadata = dict(observation.metadata)
    if "top_features" in metadata:
        metadata["top_features"] = ",".join(top_features)
    return {
        "sample_id": observation.sample_id,
        "predicted_label_id": observation.metadata.get("predicted_label_id") or observation.metadata.get("label_id"),
        "predicted_class": _evidence_class(observation),
        "confidence": observation.metadata.get("confidence"),
        "top_features": top_features,
        "band_features": band_features,
        "band_schema": rag_band_schema(band_features),
        "metadata": metadata,
    }


def _feature_queries(predicted_class: str, top_features: str) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    features = _rag_top_features(top_features)
    thermal_features = [feature for feature in features if feature.upper() in {"S7", "S8", "S9"}]
    visible_features = [feature for feature in features if feature.upper() in {"S1", "S2", "S3", "S4", "S5", "S6"}]
    if len(thermal_features) >= 2:
        joined = " ".join(feature.upper() for feature in thermal_features[:3])
        queries.append(
            {
                "kind": "feature_evidence",
                "prediction_class": predicted_class,
                "query": f"{joined} brightness temperature {predicted_class} discrimination",
            }
        )
    for feature in visible_features[:2]:
        queries.append(
            {
                "kind": "feature_evidence",
                "prediction_class": predicted_class,
                "query": f"{feature.upper()} radiance cloud ice ocean discrimination",
            }
        )
    return queries


def _evidence_class(observation: ObservationSample) -> str | None:
    return observation.metadata.get("predicted_class") or observation.metadata.get("label_name")


def _rag_band_features(observation: ObservationSample) -> dict[str, float | str]:
    feature_lookup = {name.upper(): value for name, value in observation.features.items()}
    return {name: feature_lookup[name] for name in RAG_BAND_NAMES if name in feature_lookup}


def _rag_top_features(raw_value: str) -> list[str]:
    return [feature for feature in _parse_top_features(raw_value) if not feature.lower().startswith("pc")]


def _parse_top_features(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    normalized = raw_value.replace(";", ",").replace("|", ",")
    return [feature.strip() for feature in normalized.split(",") if feature.strip()]


def _dedupe_queries(queries: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for query in queries:
        query_text = query["query"]
        if query_text in seen:
            continue
        seen.add(query_text)
        deduped.append(query)
    return deduped
