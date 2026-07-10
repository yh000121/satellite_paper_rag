from __future__ import annotations

from satellite_paper_rag.observations.schema import ObservationSample


CLASS_BASE_QUERIES = {
    "cloud": [
        "cloud detection SLSTR brightness temperature reflectance",
        "cloud mask S7 S8 threshold",
        "cloud ice ocean visible reflectance thermal channels",
    ],
    "ice": [
        "sea ice SLSTR reflectance brighter ocean",
        "ice cloud ocean reflectance wavelengths",
        "sea ice boundary cloud detection reflectance",
    ],
    "sea": [
        "open ocean clear-sky SLSTR brightness temperature",
        "darker ocean reflectance wavelengths ice cloud",
        "clear-sky pixels S7 S8 brightness temperature 1:1 line",
    ],
}


def build_prediction_evidence_queries(observation: ObservationSample) -> list[dict[str, str]]:
    predicted_class = observation.metadata.get("predicted_class")
    if not predicted_class:
        raise ValueError("Prediction evidence retrieval requires predicted_class metadata.")
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
    return {
        "sample_id": observation.sample_id,
        "predicted_label_id": observation.metadata.get("predicted_label_id"),
        "predicted_class": observation.metadata.get("predicted_class"),
        "confidence": observation.metadata.get("confidence"),
        "top_features": _parse_top_features(observation.metadata.get("top_features", "")),
        "metadata": observation.metadata,
    }


def _feature_queries(predicted_class: str, top_features: str) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    features = _parse_top_features(top_features)
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
                "query": f"{feature.upper()} visible reflectance cloud ice ocean",
            }
        )
    for feature in features:
        if feature.lower().startswith("pc"):
            queries.append(
                {
                    "kind": "feature_evidence",
                    "prediction_class": predicted_class,
                    "query": f"spectral component feature contribution {predicted_class} classification",
                }
            )
            break
    return queries


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
