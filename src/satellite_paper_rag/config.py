from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChunkingConfig:
    sentence_window_radius: int = 1
    min_rule_signal_count: int = 2
    max_child_chars: int = 1200
    recursive_chunk_size: int = 1000
    recursive_chunk_overlap: int = 120


@dataclass(frozen=True)
class RetrievalConfig:
    chunk_type_boosts: dict[str, float] = field(
        default_factory=lambda: {
            "rule_candidate": 3.0,
            "sentence_window_child": 2.0,
            "figure_table": 1.5,
            "paragraph_child": 1.0,
            "section_parent": 0.5,
        }
    )
    threshold_boost: float = 5.0
    band_or_feature_boost: float = 1.0
