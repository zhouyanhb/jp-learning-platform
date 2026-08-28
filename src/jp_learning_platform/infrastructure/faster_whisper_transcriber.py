"""faster-whisper transcription adapter."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from math import inf, isfinite
from types import SimpleNamespace
from typing import Any
import unicodedata

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG,
)
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    JapaneseMorpheme,
    JapaneseMorphologicalAnalyzer,
    SudachiMorphologicalAnalyzer,
    morphological_particle_chain_penalty,
)
from jp_learning_platform.workflow.whisper_stage import (
    ShortAnomalyRetryAudit,
    ShortUtteranceAnalysisAudit,
    WhisperRetryDecision,
    WhisperTranscript,
    WhisperTranscriptionRequest,
)
from jp_learning_platform.workflow.transcript_omission_shadow_stage import (
    TranscriptOmissionCandidateDisagreement,
    TranscriptOmissionForegroundProbeAudit,
    TranscriptOmissionShadowAudit,
    TranscriptOmissionShadowRequest,
)

_FOREGROUND_PROBE_MINIMUM_GAP_SECONDS = 14.0
_FOREGROUND_PROBE_WINDOW_SECONDS = 4.0
_FOREGROUND_PROBE_STRIDE_SECONDS = 3.0
_FOREGROUND_PROBE_MAX_WINDOWS = 6

DEFAULT_WHISPER_MODEL_SIZE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.model_size
DEFAULT_WHISPER_LANGUAGE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.language
DEFAULT_WHISPER_INITIAL_PROMPT = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.initial_prompt
DEFAULT_WHISPER_DEVICE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.device
DEFAULT_WHISPER_COMPUTE_TYPE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.compute_type
DEFAULT_WHISPER_BEAM_SIZE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.beam_size
DEFAULT_WHISPER_BEST_OF = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.best_of
DEFAULT_WHISPER_TEMPERATURE = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.temperature
DEFAULT_WHISPER_WORD_TIMESTAMPS = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.word_timestamps
)
DEFAULT_WHISPER_VAD_FILTER = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.vad_filter
DEFAULT_VAD_MIN_SILENCE_MS = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.vad_min_silence_ms
)
DEFAULT_WHISPER_CONDITION_ON_PREVIOUS_TEXT = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.condition_on_previous_text
)
DEFAULT_WHISPER_HALLUCINATION_SILENCE_THRESHOLD_SECONDS = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.hallucination_silence_threshold_seconds
)
DEFAULT_RETRY_CONFIDENCE_THRESHOLD = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_confidence_threshold
)
DEFAULT_RETRY_CONTEXT_CONFIDENCE_THRESHOLD = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_context_confidence_threshold
)
DEFAULT_RETRY_MIN_CONFIDENCE_IMPROVEMENT = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_min_confidence_improvement
)
DEFAULT_RETRY_MAX_SEGMENTS = DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_max_segments
DEFAULT_RETRY_LOCAL_CANDIDATE_COUNT = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_local_candidate_count
)
DEFAULT_RETRY_MORPHOLOGICAL_WORD_CONFIDENCE_THRESHOLD = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_morphological_word_confidence_threshold
)
DEFAULT_RETRY_INTERNAL_WORD_GAP_SECONDS = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_internal_word_gap_seconds
)
DEFAULT_RETRY_INTERNAL_GAP_EDGE_CONFIDENCE = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_internal_gap_edge_confidence
)
DEFAULT_RETRY_MINIMUM_GAP_REDUCTION_RATIO = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_minimum_gap_reduction_ratio
)
DEFAULT_RETRY_MINIMUM_TEXT_SIMILARITY = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_minimum_text_similarity
)
DEFAULT_RETRY_MINIMUM_ORIGINAL_CHARACTER_COVERAGE = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_minimum_original_character_coverage
)
DEFAULT_RETRY_MAX_LANGUAGE_MODEL_REGRESSION = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_max_language_model_regression
)
DEFAULT_RETRY_MORPHOLOGICAL_REPAIR_MAX_LANGUAGE_MODEL_REGRESSION = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_morphological_repair_max_language_model_regression
)
DEFAULT_RETRY_MAX_CONFIDENCE_REGRESSION = (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG.retry_max_confidence_regression
)


class FasterWhisperDependencyError(RuntimeError):
    """Raised when faster-whisper is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "faster-whisper is required for transcription. "
            "Install it with: python -m pip install -e '.[asr]'"
        )


@dataclass(frozen=True, slots=True)
class _InternalWordGap:
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True, slots=True)
class _ShortRetryAttempt:
    short_window_text: str = ""
    extracted_text: str = ""
    left_anchor_matched: bool = False
    right_anchor_matched: bool = False
    anchor_order_valid: bool = False
    failure_reason: str = ""


@dataclass(slots=True)
class FasterWhisperTranscriber:
    """Transcribe Japanese audio into word-aware domain segments."""

    model_size: str = DEFAULT_WHISPER_MODEL_SIZE
    language: str = DEFAULT_WHISPER_LANGUAGE
    initial_prompt: str = DEFAULT_WHISPER_INITIAL_PROMPT
    device: str = DEFAULT_WHISPER_DEVICE
    compute_type: str = DEFAULT_WHISPER_COMPUTE_TYPE
    beam_size: int = DEFAULT_WHISPER_BEAM_SIZE
    best_of: int = DEFAULT_WHISPER_BEST_OF
    temperature: float = DEFAULT_WHISPER_TEMPERATURE
    word_timestamps: bool = DEFAULT_WHISPER_WORD_TIMESTAMPS
    vad_filter: bool = DEFAULT_WHISPER_VAD_FILTER
    vad_min_silence_ms: int = DEFAULT_VAD_MIN_SILENCE_MS
    condition_on_previous_text: bool = DEFAULT_WHISPER_CONDITION_ON_PREVIOUS_TEXT
    hallucination_silence_threshold_seconds: float = (
        DEFAULT_WHISPER_HALLUCINATION_SILENCE_THRESHOLD_SECONDS
    )
    retry_confidence_threshold: float = DEFAULT_RETRY_CONFIDENCE_THRESHOLD
    retry_context_confidence_threshold: float = (
        DEFAULT_RETRY_CONTEXT_CONFIDENCE_THRESHOLD
    )
    retry_min_confidence_improvement: float = (
        DEFAULT_RETRY_MIN_CONFIDENCE_IMPROVEMENT
    )
    retry_max_segments: int = DEFAULT_RETRY_MAX_SEGMENTS
    retry_local_candidate_count: int = DEFAULT_RETRY_LOCAL_CANDIDATE_COUNT
    retry_morphological_word_confidence_threshold: float = (
        DEFAULT_RETRY_MORPHOLOGICAL_WORD_CONFIDENCE_THRESHOLD
    )
    retry_internal_word_gap_seconds: float = DEFAULT_RETRY_INTERNAL_WORD_GAP_SECONDS
    retry_internal_gap_edge_confidence: float = (
        DEFAULT_RETRY_INTERNAL_GAP_EDGE_CONFIDENCE
    )
    retry_minimum_gap_reduction_ratio: float = (
        DEFAULT_RETRY_MINIMUM_GAP_REDUCTION_RATIO
    )
    retry_minimum_text_similarity: float = DEFAULT_RETRY_MINIMUM_TEXT_SIMILARITY
    retry_minimum_original_character_coverage: float = (
        DEFAULT_RETRY_MINIMUM_ORIGINAL_CHARACTER_COVERAGE
    )
    retry_max_language_model_regression: float = (
        DEFAULT_RETRY_MAX_LANGUAGE_MODEL_REGRESSION
    )
    retry_morphological_repair_max_language_model_regression: float = (
        DEFAULT_RETRY_MORPHOLOGICAL_REPAIR_MAX_LANGUAGE_MODEL_REGRESSION
    )
    retry_max_confidence_regression: float = DEFAULT_RETRY_MAX_CONFIDENCE_REGRESSION
    _model: Any | None = field(default=None, init=False, repr=False)
    _morphological_analyzer: JapaneseMorphologicalAnalyzer | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def transcribe(self, request: WhisperTranscriptionRequest) -> WhisperTranscript:
        if not isinstance(request, WhisperTranscriptionRequest):
            raise TypeError("request must be a WhisperTranscriptionRequest.")

        model = self._load_model()
        source_path = str(request.source_path)
        external_segments, _info = model.transcribe(
            source_path,
            **self._transcription_options(),
        )
        (
            selected_segments,
            retry_decisions,
            short_anomaly_retry_audits,
            short_utterance_analysis_audits,
        ) = self._retry_low_confidence_segments(model, source_path, tuple(external_segments))

        segments: list[Segment] = []
        for external_segment in selected_segments:
            text = str(getattr(external_segment, "text", "")).strip()
            if text:
                segments.append(self._convert_segment(len(segments), external_segment))

        return WhisperTranscript(
            source_path=request.source_path,
            segments=tuple(segments),
            retry_decisions=retry_decisions,
            short_anomaly_retry_audits=short_anomaly_retry_audits,
            short_utterance_analysis_audits=short_utterance_analysis_audits,
        )

    def recognize_omission_candidates(
        self,
        request: TranscriptOmissionShadowRequest,
    ) -> tuple[TranscriptOmissionShadowAudit, ...]:
        model = self._load_model()
        by_position = {segment.position: segment for segment in request.segments}
        audits: list[TranscriptOmissionShadowAudit] = []
        for candidate in request.candidates:
            left = by_position.get(candidate.segment_positions[0])
            right = by_position.get(candidate.segment_positions[-1])
            prompt = left.text[-160:] if left is not None else self.initial_prompt
            raw_texts: list[str] = []
            extracted_texts: list[str] = []
            coverages: list[float] = []
            decoded_candidates: list[tuple[Any, ...]] = []
            for candidate_index in range(self.retry_local_candidate_count):
                options = self._transcription_options()
                options.update(
                    {
                        "clip_timestamps": [
                            max(0.0, candidate.time_range.start_seconds - 1.5),
                            candidate.time_range.end_seconds + 1.5,
                        ],
                        "initial_prompt": prompt,
                        "condition_on_previous_text": False,
                        "vad_filter": False,
                    }
                )
                options.update(_local_decode_profile(candidate_index))
                external_segments, _info = model.transcribe(
                    str(request.source_path),
                    **options,
                )
                decoded = tuple(external_segments)
                raw_texts.append(_external_text(decoded).strip())
                extracted = _segments_inside_time_range(
                    decoded,
                    candidate.time_range,
                )
                decoded_candidates.append(extracted)
                extracted_texts.append(_external_text(extracted).strip())
                coverages.append(
                    _external_time_coverage(extracted, candidate.time_range)
                )
            consensus_text, consensus_count = _candidate_text_consensus(
                tuple(extracted_texts)
            )
            consensus_reached = bool(
                consensus_text
                and consensus_count == self.retry_local_candidate_count
            )
            analyzer = self._get_morphological_analyzer()
            assessment = _assess_omission_shadow_candidates(
                tuple(extracted_texts),
                tuple(decoded_candidates),
                tuple(coverages),
                left.text if left is not None else "",
                right.text if right is not None else "",
                analyzer,
            )
            foreground_probes = self._recognize_foreground_probe_candidates(
                model,
                str(request.source_path),
                candidate.time_range,
                left.text if left is not None else "",
                right.text if right is not None else "",
                analyzer,
            )
            foreground_probes = _annotate_foreground_probe_hallucinations(
                foreground_probes
            )
            foreground_assessment = _assess_foreground_probe_candidates(
                tuple(extracted_texts),
                foreground_probes,
                left.text if left is not None else "",
                right.text if right is not None else "",
                analyzer,
            )
            reasons: list[str] = []
            if not any(extracted_texts):
                reasons.append("no_candidate_text")
            if not consensus_reached:
                reasons.append("candidate_consensus_missing")
            if max(coverages, default=0.0) < 0.5:
                reasons.append("insufficient_time_coverage")
            if consensus_reached:
                reasons.append("candidate_consensus_reached")
            reasons.extend(assessment["reasons"])
            audits.append(
                TranscriptOmissionShadowAudit(
                    time_range=candidate.time_range,
                    segment_positions=candidate.segment_positions,
                    retry_attempted=True,
                    foreground_probe_attempted=bool(foreground_probes),
                    foreground_probe_audits=foreground_probes,
                    foreground_filtered_candidate_texts=foreground_assessment[
                        "filtered_texts"
                    ],
                    foreground_full_gap_candidate_rejection_reasons=(
                        foreground_assessment["full_gap_rejection_reasons"]
                    ),
                    foreground_stable_character_consensus=foreground_assessment[
                        "stable_characters"
                    ],
                    foreground_stable_morpheme_consensus=foreground_assessment[
                        "stable_morphemes"
                    ],
                    foreground_candidate_disagreements=foreground_assessment[
                        "disagreements"
                    ],
                    foreground_lexical_uncertainty_detected=foreground_assessment[
                        "lexical_uncertainty_detected"
                    ],
                    foreground_lexical_uncertainty_reasons=foreground_assessment[
                        "lexical_uncertainty_reasons"
                    ],
                    foreground_alignment_reasons=foreground_assessment[
                        "alignment_reasons"
                    ],
                    raw_candidate_texts=tuple(raw_texts),
                    extracted_candidate_texts=tuple(extracted_texts),
                    recovered_time_coverage=tuple(coverages),
                    candidate_consensus_text=consensus_text,
                    candidate_consensus_count=consensus_count,
                    candidate_count=self.retry_local_candidate_count,
                    consensus_reached=consensus_reached,
                    normalized_candidate_texts=assessment["normalized_texts"],
                    core_character_consensus=assessment["core_characters"],
                    core_character_coverage=assessment["core_coverage"],
                    core_morpheme_consensus=assessment["core_morphemes"],
                    candidate_disagreements=assessment["disagreements"],
                    candidate_confidences=assessment["confidences"],
                    candidate_language_model_scores=assessment[
                        "language_model_scores"
                    ],
                    candidate_morphology_penalties=assessment[
                        "morphology_penalties"
                    ],
                    context_validation_passed=assessment["context_passed"],
                    confidence_validation_passed=assessment[
                        "confidence_passed"
                    ],
                    language_model_validation_passed=assessment[
                        "language_model_passed"
                    ],
                    morphology_validation_passed=assessment[
                        "morphology_passed"
                    ],
                    lexical_uncertainty_detected=assessment[
                        "lexical_uncertainty_detected"
                    ],
                    lexical_uncertainty_reasons=assessment[
                        "lexical_uncertainty_reasons"
                    ],
                    uncertain_noun_sequences=assessment[
                        "uncertain_noun_sequences"
                    ],
                    validation_passed=assessment["validation_passed"],
                    automatic_replacement_allowed=False,
                    review_reasons=tuple(reasons),
                )
            )
        return tuple(audits)

    def _recognize_foreground_probe_candidates(
        self,
        model: Any,
        source_path: str,
        time_range: TimeRange,
        left_context: str,
        right_context: str,
        analyzer: JapaneseMorphologicalAnalyzer,
    ) -> tuple[TranscriptOmissionForegroundProbeAudit, ...]:
        audits: list[TranscriptOmissionForegroundProbeAudit] = []
        for window in _foreground_probe_windows(time_range):
            options = self._transcription_options()
            options.update(
                {
                    "clip_timestamps": [
                        window.start_seconds,
                        window.end_seconds,
                    ],
                    "initial_prompt": self.initial_prompt,
                    "condition_on_previous_text": False,
                    "vad_filter": False,
                    "hallucination_silence_threshold": None,
                    "temperature": 0.0,
                }
            )
            external_segments, _info = model.transcribe(source_path, **options)
            decoded = tuple(external_segments)
            extracted = _segments_inside_time_range(decoded, window)
            raw_text = _external_text(decoded).strip()
            extracted_text = _external_text(extracted).strip()
            audits.append(
                TranscriptOmissionForegroundProbeAudit(
                    time_range=window,
                    raw_text=raw_text,
                    extracted_text=extracted_text,
                    confidence=_finite_or_none(
                        _external_segment_confidence(extracted)
                    ),
                    language_model_score=_mean_finite_attribute(
                        extracted,
                        "avg_logprob",
                    ),
                    morphology_penalty=(
                        _morphological_structure_penalty(extracted_text, analyzer)
                        + _grammatical_structure_penalty(extracted_text, analyzer)
                    ),
                    context_anchor_detected=_contains_context_anchor(
                        extracted_text,
                        left_context,
                        right_context,
                    ),
                )
            )
        return tuple(audits)

    def _transcription_options(self) -> dict[str, object]:
        return {
            "language": self.language,
            "initial_prompt": self.initial_prompt,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "word_timestamps": self.word_timestamps,
            "vad_filter": self.vad_filter,
            "vad_parameters": {
                "min_silence_duration_ms": self.vad_min_silence_ms,
            },
            # The first pass is intentionally neutral. Context is introduced only
            # for bounded retries whose preceding text is already high confidence.
            "condition_on_previous_text": self.condition_on_previous_text,
            "hallucination_silence_threshold": (
                self.hallucination_silence_threshold_seconds
            ),
        }

    def _retry_low_confidence_segments(
        self,
        model: Any,
        source_path: str,
        segments: tuple[Any, ...],
    ) -> tuple[
        tuple[Any, ...],
        tuple[WhisperRetryDecision, ...],
        tuple[ShortAnomalyRetryAudit, ...],
        tuple[ShortUtteranceAnalysisAudit, ...],
    ]:
        selected: list[Any] = []
        decisions: list[WhisperRetryDecision] = []
        short_audits: list[ShortAnomalyRetryAudit] = []
        short_analysis_audits: list[ShortUtteranceAnalysisAudit] = []
        retries = 0
        reliable_context = ""
        for segment_index, segment in enumerate(segments):
            left_anchor = segments[segment_index - 1] if segment_index else None
            right_anchor = (
                segments[segment_index + 1]
                if segment_index + 1 < len(segments)
                else None
            )
            confidence = _external_segment_confidence((segment,))
            internal_gap = _suspicious_internal_word_gap(
                segment,
                self.retry_internal_word_gap_seconds,
                self.retry_internal_gap_edge_confidence,
            )
            morphological_anomaly = self._has_low_confidence_morphological_anomaly(
                segment
            )
            short_utterance_anomaly = self._has_short_utterance_anomaly(segment)
            short_analysis = _short_utterance_analysis_audit(
                segment_index,
                segment,
                self._get_morphological_analyzer(),
                short_utterance_anomaly,
            )
            if short_analysis is not None:
                short_analysis_audits.append(short_analysis)
            requires_morphological_repair = bool(
                morphological_anomaly or short_utterance_anomaly
            )
            replacement = (segment,)
            if (
                (
                    confidence < self.retry_confidence_threshold
                    or internal_gap is not None
                    or requires_morphological_repair
                )
                and (retries < self.retry_max_segments or requires_morphological_repair)
            ):
                retries += 1
                clip_start = float(getattr(segment, "start"))
                clip_end = float(getattr(segment, "end"))
                if internal_gap is not None:
                    clip_start = max(
                        clip_start,
                        internal_gap.start_seconds - 5.0,
                    )
                    clip_end = min(
                        clip_end,
                        internal_gap.end_seconds + 2.0,
                    )
                elif short_utterance_anomaly:
                    clip_start = max(
                        0.0,
                        clip_start - 3.0,
                        float(getattr(left_anchor, "start"))
                        if left_anchor is not None
                        else 0.0,
                    )
                    clip_end = min(
                        clip_end + 6.0,
                        float(getattr(right_anchor, "end"))
                        if right_anchor is not None
                        else clip_end + 6.0,
                    )
                raw_candidates, short_attempts = self._local_retry_candidates(
                    model,
                    source_path,
                    segment,
                    reliable_context,
                    internal_gap,
                    requires_morphological_repair,
                    short_utterance_anomaly,
                    left_anchor,
                    right_anchor,
                    clip_start,
                    clip_end,
                )
                candidate = raw_candidates[0] if raw_candidates else ()
                candidate_confidence = _external_segment_confidence(candidate)
                confidence_improved = bool(
                    candidate
                    and isfinite(candidate_confidence)
                    and (
                        not isfinite(confidence)
                        or candidate_confidence
                        >= confidence + self.retry_min_confidence_improvement
                    )
                )
                gap_repaired = False
                ordinary_retry_accepted = False
                if (internal_gap is not None or requires_morphological_repair) and raw_candidates:
                    candidate, gap_repaired, candidate_decisions = (
                        self._select_local_retry_candidate(
                            original=(segment,),
                            candidates=raw_candidates,
                            internal_gap=internal_gap,
                            requires_morphological_repair=requires_morphological_repair,
                            original_confidence=confidence,
                        )
                    )
                    decisions.extend(candidate_decisions)
                    candidate_confidence = _external_segment_confidence(candidate)
                elif candidate:
                    analyzer = self._get_morphological_analyzer()
                    morphology_preserved = bool(
                        _morphological_structure_penalty(
                            _external_text(candidate),
                            analyzer,
                        )
                        <= _morphological_structure_penalty(
                            _external_text((segment,)),
                            analyzer,
                        )
                    )
                    ordinary_retry_accepted = bool(
                        confidence_improved and morphology_preserved
                    )
                    decisions.append(
                        self._retry_decision(
                            original=(segment,),
                            candidate=candidate,
                            internal_gap=None,
                            original_confidence=confidence,
                            candidate_confidence=candidate_confidence,
                            candidate_internal_gap=None,
                            accepted=ordinary_retry_accepted,
                            passed_validation=ordinary_retry_accepted,
                        )
                    )
                if gap_repaired or (
                    internal_gap is None
                    and not requires_morphological_repair
                    and ordinary_retry_accepted
                ):
                    replacement = candidate
                    confidence = candidate_confidence
                if short_utterance_anomaly:
                    failure_reasons = {
                        attempt.failure_reason
                        for attempt in short_attempts
                        if attempt.failure_reason
                    }
                    if raw_candidates and not gap_repaired:
                        maximum_support = max(
                            (
                                decision.candidate_support_count
                                for decision in candidate_decisions
                            ),
                            default=0,
                        )
                        if maximum_support < self.retry_local_candidate_count:
                            failure_reasons.add("candidate_consensus_missing")
                        else:
                            failure_reasons.add("candidate_validation_rejected")
                    short_audits.append(
                        ShortAnomalyRetryAudit(
                            segment_position=segment_index,
                            time_range=TimeRange(
                                float(getattr(segment, "start")),
                                float(getattr(segment, "end")),
                            ),
                            original_text=str(getattr(segment, "text", "")).strip(),
                            short_anomaly_detected=True,
                            retry_attempted=True,
                            short_window_candidate_texts=tuple(
                                attempt.short_window_text for attempt in short_attempts
                            ),
                            extracted_candidate_texts=tuple(
                                attempt.extracted_text for attempt in short_attempts
                            ),
                            left_anchor_matches=tuple(
                                attempt.left_anchor_matched for attempt in short_attempts
                            ),
                            right_anchor_matches=tuple(
                                attempt.right_anchor_matched for attempt in short_attempts
                            ),
                            anchor_orders_valid=tuple(
                                attempt.anchor_order_valid for attempt in short_attempts
                            ),
                            accepted=gap_repaired,
                            failure_reasons=tuple(sorted(failure_reasons)),
                        )
                    )

            selected.extend(replacement)
            if confidence >= self.retry_context_confidence_threshold:
                reliable_context = "".join(
                    str(getattr(item, "text", "")).strip()
                    for item in replacement
                )

        return (
            tuple(selected),
            tuple(decisions),
            tuple(short_audits),
            tuple(short_analysis_audits),
        )

    def _local_retry_candidates(
        self,
        model: Any,
        source_path: str,
        segment: Any,
        reliable_context: str,
        internal_gap: _InternalWordGap | None,
        morphological_anomaly: bool,
        short_utterance_anomaly: bool,
        left_anchor: Any | None,
        right_anchor: Any | None,
        clip_start: float,
        clip_end: float,
    ) -> tuple[tuple[tuple[Any, ...], ...], tuple[_ShortRetryAttempt, ...]]:
        candidate_count = (
            self.retry_local_candidate_count
            if internal_gap is not None or morphological_anomaly
            else 1
        )
        generated: list[tuple[Any, ...]] = []
        short_attempts: list[_ShortRetryAttempt] = []
        for candidate_index in range(max(1, candidate_count)):
            retry_clip = (clip_start, clip_end)
            if short_utterance_anomaly:
                retry_clip = (
                    max(0.0, float(getattr(segment, "start")) - 1.5),
                    float(getattr(segment, "end")) + 1.5,
                )
            options = self._retry_options(
                segment,
                reliable_context,
                disable_vad=internal_gap is not None,
                clip_timestamps=retry_clip,
            )
            if internal_gap is not None or morphological_anomaly:
                options.update(_local_decode_profile(candidate_index))
            retry_segments, _info = model.transcribe(source_path, **options)
            candidate = tuple(retry_segments)
            short_window_text = _external_text(candidate).strip()
            extracted_text = ""
            left_matched = False
            right_matched = False
            order_valid = False
            failure_reason = ""
            if internal_gap is not None and candidate:
                candidate = (
                    _splice_external_retry(
                        segment,
                        candidate,
                        clip_start,
                        clip_end,
                        internal_gap,
                    ),
                )
            elif short_utterance_anomaly and candidate:
                candidate = _extract_external_retry_window(segment, candidate)
                extracted_text = _external_text(candidate).strip()
                if not candidate:
                    failure_reason = "target_window_empty"
                if candidate and left_anchor is not None and right_anchor is not None:
                    anchor_options = self._retry_options(
                        segment,
                        reliable_context,
                        clip_timestamps=(clip_start, clip_end),
                    )
                    anchor_options.update(_local_decode_profile(candidate_index))
                    anchor_segments, _anchor_info = model.transcribe(
                        source_path,
                        **anchor_options,
                    )
                    left_matched, right_matched, order_valid = (
                        _text_anchor_match_status(
                        tuple(anchor_segments),
                        left_anchor,
                        right_anchor,
                        )
                    )
                    if not order_valid:
                        if not left_matched:
                            failure_reason = "left_anchor_missing"
                        elif not right_matched:
                            failure_reason = "right_anchor_missing"
                        else:
                            failure_reason = "anchor_order_invalid"
                        candidate = ()
            elif short_utterance_anomaly:
                failure_reason = "short_retry_returned_empty"
            if short_utterance_anomaly:
                short_attempts.append(
                    _ShortRetryAttempt(
                        short_window_text=short_window_text,
                        extracted_text=extracted_text,
                        left_anchor_matched=left_matched,
                        right_anchor_matched=right_matched,
                        anchor_order_valid=order_valid,
                        failure_reason=failure_reason,
                    )
                )
            if candidate:
                generated.append(
                    _preserve_retry_envelope((segment,), candidate)
                )
        return tuple(generated), tuple(short_attempts)

    def _select_local_retry_candidate(
        self,
        *,
        original: tuple[Any, ...],
        candidates: tuple[tuple[Any, ...], ...],
        internal_gap: _InternalWordGap | None,
        requires_morphological_repair: bool,
        original_confidence: float,
    ) -> tuple[tuple[Any, ...], bool, tuple[WhisperRetryDecision, ...]]:
        candidates = tuple(
            _preserve_retry_envelope(original, candidate)
            for candidate in candidates
        )
        grouped: dict[str, list[tuple[Any, ...]]] = {}
        for candidate in candidates:
            grouped.setdefault(_normalized_external_text(candidate), []).append(candidate)

        assessed: list[tuple[tuple[Any, ...], bool, int, float]] = []
        for variants in grouped.values():
            candidate = max(variants, key=_external_segment_confidence)
            passed = self._local_candidate_passes_validation(
                original,
                candidate,
                internal_gap,
                requires_morphological_repair,
                original_confidence,
                support_count=len(variants),
                generated_count=len(candidates),
            )
            score = self._local_candidate_selection_score(
                original,
                candidate,
                support_count=len(variants),
                generated_count=len(candidates),
            )
            assessed.append((candidate, passed, len(variants), score))

        valid = [item for item in assessed if item[1]]
        selected = max(valid, key=lambda item: item[3]) if valid else None
        decisions = tuple(
            self._retry_decision(
                original=original,
                candidate=candidate,
                internal_gap=internal_gap,
                original_confidence=original_confidence,
                candidate_confidence=_external_segment_confidence(candidate),
                candidate_internal_gap=_suspicious_internal_word_gap(
                    candidate[0],
                    self.retry_internal_word_gap_seconds,
                    self.retry_internal_gap_edge_confidence,
                ),
                accepted=bool(selected and candidate is selected[0]),
                passed_validation=passed,
                candidate_support_count=support_count,
                selection_score=score,
            )
            for candidate, passed, support_count, score in assessed
        )
        if selected is None:
            return original, False, decisions
        return selected[0], True, decisions

    def _local_candidate_passes_validation(
        self,
        original: tuple[Any, ...],
        candidate: tuple[Any, ...],
        internal_gap: _InternalWordGap | None,
        requires_morphological_repair: bool,
        original_confidence: float,
        *,
        support_count: int,
        generated_count: int,
    ) -> bool:
        candidate_confidence = _external_segment_confidence(candidate)
        candidate_gap = _suspicious_internal_word_gap(
            candidate[0],
            self.retry_internal_word_gap_seconds,
            self.retry_internal_gap_edge_confidence,
        )
        resolves_morphology = self._resolves_morphological_anomaly(
            original,
            candidate,
        )
        trusted_short_repair = bool(
            resolves_morphology
            and _short_utterance_structure_penalty(
                _external_text(original),
                self._get_morphological_analyzer(),
            )
            and support_count == generated_count
            and generated_count >= 3
            and candidate_confidence
            >= original_confidence + self.retry_min_confidence_improvement
        )
        language_model_acceptable = _language_model_score_is_acceptable(
            original,
            candidate,
            (
                self.retry_morphological_repair_max_language_model_regression
                if resolves_morphology
                else self.retry_max_language_model_regression
            ),
        )
        return bool(
            isfinite(candidate_confidence)
            and candidate_confidence
            >= original_confidence - self.retry_max_confidence_regression
            and (
                internal_gap is None
                or candidate_gap is None
                or candidate_gap.duration_seconds
                <= internal_gap.duration_seconds
                * self.retry_minimum_gap_reduction_ratio
            )
            and (
                trusted_short_repair
                or _external_text_similarity(original, candidate)
                >= self.retry_minimum_text_similarity
            )
            and (
                trusted_short_repair
                or _original_character_coverage(original, candidate)
                >= self.retry_minimum_original_character_coverage
            )
            and language_model_acceptable
            and (not requires_morphological_repair or resolves_morphology)
            and (
                trusted_short_repair
                or not self._introduces_grammatical_degradation(original, candidate)
            )
            and (
                resolves_morphology
                or self._content_deletion_has_sufficient_confidence(
                    original,
                    candidate,
                    original_confidence,
                    candidate_confidence,
                )
            )
        )

    def _has_low_confidence_morphological_anomaly(self, segment: Any) -> bool:
        probabilities = tuple(
            float(value)
            for word in (getattr(segment, "words", None) or ())
            if (value := getattr(word, "probability", None)) is not None
        )
        return bool(
            probabilities
            and min(probabilities)
            <= self.retry_morphological_word_confidence_threshold
            and morphological_particle_chain_penalty(
                str(getattr(segment, "text", "")),
                self._get_morphological_analyzer(),
            )
        )

    def _has_short_utterance_anomaly(self, segment: Any) -> bool:
        return bool(
            _short_utterance_structure_penalty(
                str(getattr(segment, "text", "")),
                self._get_morphological_analyzer(),
            )
        )

    def _local_candidate_selection_score(
        self,
        original: tuple[Any, ...],
        candidate: tuple[Any, ...],
        *,
        support_count: int,
        generated_count: int,
    ) -> float:
        analyzer = self._get_morphological_analyzer()
        original_grammar = _grammatical_structure_penalty(
            _external_text(original), analyzer
        )
        candidate_grammar = _grammatical_structure_penalty(
            _external_text(candidate), analyzer
        )
        confidence = _external_segment_confidence(candidate)
        return round(
            0.40 * (support_count / generated_count)
            + 0.25 * _external_text_similarity(original, candidate)
            + 0.20 * _original_character_coverage(original, candidate)
            + 0.10 * max(0.0, min(1.0, confidence))
            + 0.05 * float(candidate_grammar < original_grammar),
            6,
        )

    def _retry_decision(
        self,
        *,
        original: tuple[Any, ...],
        candidate: tuple[Any, ...],
        internal_gap: _InternalWordGap | None,
        original_confidence: float,
        candidate_confidence: float,
        candidate_internal_gap: _InternalWordGap | None,
        accepted: bool,
        passed_validation: bool | None = None,
        candidate_support_count: int = 1,
        selection_score: float | None = None,
    ) -> WhisperRetryDecision:
        analyzer = self._get_morphological_analyzer()
        original_text = _external_text(original)
        candidate_text = _external_text(candidate)
        text_similarity = _external_text_similarity(original, candidate)
        character_coverage = _original_character_coverage(original, candidate)
        original_lm = _mean_finite_attribute(original, "avg_logprob")
        candidate_lm = _mean_finite_attribute(candidate, "avg_logprob")
        original_grammar = _grammatical_structure_penalty(original_text, analyzer)
        candidate_grammar = _grammatical_structure_penalty(candidate_text, analyzer)
        deletes_content = _deletes_content_or_particle(
            original_text,
            candidate_text,
            analyzer,
        )
        resolves_morphological_anomaly = _resolves_morphological_anomaly_text(
            original_text,
            candidate_text,
            analyzer,
        )
        reasons: list[str] = []
        if candidate_confidence < (
            original_confidence - self.retry_max_confidence_regression
        ):
            reasons.append("confidence_regression")
        if (
            internal_gap is not None
            and
            candidate_internal_gap is not None
            and candidate_internal_gap.duration_seconds
            > internal_gap.duration_seconds * self.retry_minimum_gap_reduction_ratio
        ):
            reasons.append("insufficient_gap_reduction")
        if text_similarity < self.retry_minimum_text_similarity:
            reasons.append("low_text_similarity")
        if character_coverage < self.retry_minimum_original_character_coverage:
            reasons.append("low_original_character_coverage")
        if not _language_model_score_is_acceptable(
            original,
            candidate,
            self.retry_max_language_model_regression,
        ):
            reasons.append("language_model_regression")
        if candidate_grammar > original_grammar:
            reasons.append("grammatical_structure_degradation")
        if _morphological_structure_penalty(
            candidate_text,
            analyzer,
        ) > _morphological_structure_penalty(original_text, analyzer):
            reasons.append("morphological_structure_degradation")
        if deletes_content and candidate_confidence < (
            original_confidence + self.retry_min_confidence_improvement
        ):
            reasons.append("content_deletion_without_confidence_gain")
        candidate_passed = accepted if passed_validation is None else passed_validation
        if candidate_passed and not accepted:
            reasons = ["valid_candidate_not_selected"]
        elif accepted and resolves_morphological_anomaly:
            reasons = ["accepted_morphological_repair"]
        return WhisperRetryDecision(
            time_range=(
                TimeRange(internal_gap.start_seconds, internal_gap.end_seconds)
                if internal_gap is not None
                else TimeRange(
                    float(getattr(original[0], "start")),
                    float(getattr(original[-1], "end")),
                )
            ),
            accepted=accepted,
            reasons=tuple(reasons) if reasons else ("accepted_all_checks",),
            original_text=original_text,
            candidate_text=candidate_text,
            original_confidence=(
                original_confidence if isfinite(original_confidence) else None
            ),
            candidate_confidence=(
                candidate_confidence if isfinite(candidate_confidence) else None
            ),
            text_similarity=text_similarity,
            original_character_coverage=character_coverage,
            original_language_model_score=original_lm,
            candidate_language_model_score=candidate_lm,
            original_grammar_penalty=original_grammar,
            candidate_grammar_penalty=candidate_grammar,
            deletes_content_or_particle=deletes_content,
            passed_validation=candidate_passed,
            selected=accepted,
            candidate_support_count=candidate_support_count,
            selection_score=selection_score,
        )

    def _introduces_grammatical_degradation(
        self,
        original: tuple[Any, ...],
        candidate: tuple[Any, ...],
    ) -> bool:
        analyzer = self._get_morphological_analyzer()
        return _grammatical_structure_penalty(
            _external_text(original),
            analyzer,
        ) < _grammatical_structure_penalty(
            _external_text(candidate),
            analyzer,
        )

    def _resolves_morphological_anomaly(
        self,
        original: tuple[Any, ...],
        candidate: tuple[Any, ...],
    ) -> bool:
        return _resolves_morphological_anomaly_text(
            _external_text(original),
            _external_text(candidate),
            self._get_morphological_analyzer(),
        )

    def _content_deletion_has_sufficient_confidence(
        self,
        original: tuple[Any, ...],
        candidate: tuple[Any, ...],
        original_confidence: float,
        candidate_confidence: float,
    ) -> bool:
        analyzer = self._get_morphological_analyzer()
        if not _deletes_content_or_particle(
            _external_text(original),
            _external_text(candidate),
            analyzer,
        ):
            return True
        return candidate_confidence >= (
            original_confidence + self.retry_min_confidence_improvement
        )

    def _get_morphological_analyzer(self) -> JapaneseMorphologicalAnalyzer:
        if self._morphological_analyzer is None:
            self._morphological_analyzer = SudachiMorphologicalAnalyzer()
        return self._morphological_analyzer

    def _retry_options(
        self,
        segment: Any,
        reliable_context: str,
        *,
        disable_vad: bool = False,
        clip_timestamps: tuple[float, float] | None = None,
    ) -> dict[str, object]:
        options = self._transcription_options()
        options.update(
            {
                "clip_timestamps": [
                    *(clip_timestamps or (
                        float(getattr(segment, "start")),
                        float(getattr(segment, "end")),
                    )),
                ],
                "initial_prompt": reliable_context or self.initial_prompt,
                "condition_on_previous_text": False,
                "vad_filter": False if disable_vad else self.vad_filter,
            }
        )
        return options

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise FasterWhisperDependencyError() from error

            self._model = _instantiate_whisper_model(
                WhisperModel,
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

        return self._model

    def _convert_segment(self, position: int, external_segment: Any) -> Segment:
        text = str(getattr(external_segment, "text", "")).strip()
        start_seconds = float(getattr(external_segment, "start"))
        end_seconds = float(getattr(external_segment, "end"))
        words = tuple(
            self._convert_word(external_word)
            for external_word in (getattr(external_segment, "words", None) or ())
            if str(getattr(external_word, "word", "")).strip()
        )

        if words:
            start_seconds = min(start_seconds, words[0].time_range.start_seconds)
            end_seconds = max(end_seconds, words[-1].time_range.end_seconds)

        time_range = TimeRange(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        sentence = Sentence(
            text=text,
            time_range=time_range,
            words=words,
            asr_boundary_word_indexes=_text_boundary_word_indexes(text, words),
        )
        return Segment(
            position=position,
            text=text,
            time_range=time_range,
            sentences=(sentence,),
        )

    def _convert_word(self, external_word: Any) -> Word:
        probability = getattr(external_word, "probability", None)
        return Word(
            text=str(getattr(external_word, "word", "")).strip(),
            time_range=TimeRange(
                start_seconds=float(getattr(external_word, "start")),
                end_seconds=float(getattr(external_word, "end")),
            ),
            confidence=float(probability) if probability is not None else None,
        )


def _instantiate_whisper_model(
    model_class: Any,
    model_size_or_path: str,
    **options: Any,
) -> Any:
    """Prefer a complete local snapshot and download only when none exists."""
    try:
        return model_class(
            model_size_or_path,
            local_files_only=True,
            **options,
        )
    except Exception as error:
        try:
            from huggingface_hub.errors import LocalEntryNotFoundError
        except ImportError:
            raise error
        if not isinstance(error, LocalEntryNotFoundError):
            raise
    return model_class(
        model_size_or_path,
        local_files_only=False,
        **options,
    )


def _external_segment_confidence(segments: tuple[Any, ...]) -> float:
    probabilities = tuple(
        float(probability)
        for segment in segments
        for word in (getattr(segment, "words", None) or ())
        if (probability := getattr(word, "probability", None)) is not None
    )
    if not probabilities:
        return -inf

    return sum(probabilities) / len(probabilities)


def _normalized_external_text(segments: tuple[Any, ...]) -> str:
    return "".join(_external_text(segments).split())


def _segments_inside_time_range(
    segments: tuple[Any, ...],
    time_range: TimeRange,
) -> tuple[Any, ...]:
    return tuple(
        segment
        for segment in segments
        if (
            float(getattr(segment, "start"))
            + float(getattr(segment, "end"))
        )
        / 2
        >= time_range.start_seconds
        and (
            float(getattr(segment, "start"))
            + float(getattr(segment, "end"))
        )
        / 2
        <= time_range.end_seconds
    )


def _external_time_coverage(
    segments: tuple[Any, ...],
    time_range: TimeRange,
) -> float:
    intervals = sorted(
        (
            max(time_range.start_seconds, float(getattr(segment, "start"))),
            min(time_range.end_seconds, float(getattr(segment, "end"))),
        )
        for segment in segments
    )
    covered = 0.0
    current_start = 0.0
    current_end = 0.0
    for start, end in intervals:
        if end <= start:
            continue
        if covered == 0.0 and current_end == 0.0:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    if current_end > current_start:
        covered += current_end - current_start
    return round(
        min(1.0, covered / time_range.duration_seconds),
        3,
    )


def _foreground_probe_windows(time_range: TimeRange) -> tuple[TimeRange, ...]:
    if time_range.duration_seconds < _FOREGROUND_PROBE_MINIMUM_GAP_SECONDS:
        return ()
    latest_start = time_range.end_seconds - _FOREGROUND_PROBE_WINDOW_SECONDS
    starts: list[float] = []
    current = time_range.start_seconds
    while (
        current <= latest_start
        and len(starts) < _FOREGROUND_PROBE_MAX_WINDOWS - 1
    ):
        starts.append(current)
        current += _FOREGROUND_PROBE_STRIDE_SECONDS
    if not starts or starts[-1] < latest_start:
        starts.append(latest_start)
    return tuple(
        TimeRange(
            round(start, 3),
            round(start + _FOREGROUND_PROBE_WINDOW_SECONDS, 3),
        )
        for start in starts
    )


def _annotate_foreground_probe_hallucinations(
    audits: tuple[TranscriptOmissionForegroundProbeAudit, ...],
) -> tuple[TranscriptOmissionForegroundProbeAudit, ...]:
    normalized = tuple(
        _normalize_consensus_text(audit.extracted_text) for audit in audits
    )
    counts = Counter(text for text in normalized if text)
    annotated: list[TranscriptOmissionForegroundProbeAudit] = []
    for audit, text in zip(audits, normalized, strict=True):
        reasons: list[str] = []
        if text and _has_repeated_text_cycle(text):
            reasons.append("repeated_text_cycle")
        if text and counts[text] >= 3:
            reasons.append("repeated_across_probe_windows")
        annotated.append(replace(audit, hallucination_reasons=tuple(reasons)))
    return tuple(annotated)


def _has_repeated_text_cycle(text: str) -> bool:
    minimum_repetitions = 4
    minimum_repeated_characters = 12
    for start in range(len(text)):
        remaining = len(text) - start
        for unit_length in range(1, min(30, remaining // minimum_repetitions) + 1):
            unit = text[start : start + unit_length]
            repetitions = 1
            cursor = start + unit_length
            while text[cursor : cursor + unit_length] == unit:
                repetitions += 1
                cursor += unit_length
            repeated_length = repetitions * unit_length
            if (
                repetitions >= minimum_repetitions
                and repeated_length >= minimum_repeated_characters
                and repeated_length >= len(text) * 0.5
            ):
                return True
    return False


def _assess_foreground_probe_candidates(
    full_gap_texts: tuple[str, ...],
    probes: tuple[TranscriptOmissionForegroundProbeAudit, ...],
    left_context: str,
    right_context: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> dict[str, object]:
    full_gap_rejection_reasons = tuple(
        _foreground_full_gap_rejection_reasons(
            text,
            left_context,
            right_context,
        )
        for text in full_gap_texts
    )
    filtered_texts = tuple(
        dict.fromkeys(
            text.strip()
            for text in (
                *(
                    text
                    for text, rejection_reasons in zip(
                        full_gap_texts,
                        full_gap_rejection_reasons,
                        strict=True,
                    )
                    if not rejection_reasons
                ),
                *(
                    probe.extracted_text
                    for probe in probes
                    if not probe.hallucination_reasons
                    and not _is_foreground_context_anchor(
                        probe.extracted_text,
                        left_context,
                        right_context,
                    )
                ),
            )
            if text.strip()
        )
    )
    normalized = tuple(_normalize_consensus_text(text) for text in filtered_texts)
    morphemes = tuple(
        tuple(
            morpheme.surface
            for morpheme in analyzer.analyze(text)
            if morpheme.part_of_speech[0] != "補助記号"
        )
        for text in filtered_texts
    )
    has_independent_support = len(filtered_texts) >= 2
    stable_morphemes = (
        _common_ordered_items(morphemes) if has_independent_support else ()
    )
    stable_characters = (
        _common_ordered_items(normalized)
        if has_independent_support and stable_morphemes
        else ()
    )
    disagreements = (
        _candidate_morpheme_disagreements(morphemes)
        if has_independent_support
        else ()
    )
    lexical_uncertainty = bool(has_independent_support and disagreements)
    return {
        "filtered_texts": filtered_texts,
        "full_gap_rejection_reasons": full_gap_rejection_reasons,
        "stable_characters": "".join(stable_characters),
        "stable_morphemes": stable_morphemes,
        "disagreements": disagreements,
        "lexical_uncertainty_detected": lexical_uncertainty,
        "lexical_uncertainty_reasons": (
            ("conflicting_lexical_fragments_across_candidates",)
            if lexical_uncertainty
            else ()
        ),
        "alignment_reasons": (
            ()
            if has_independent_support
            else ("insufficient_independent_support",)
        ),
    }


def _foreground_full_gap_rejection_reasons(
    text: str,
    left_context: str,
    right_context: str,
) -> tuple[str, ...]:
    normalized = _normalize_consensus_text(text)
    reasons: list[str] = []
    if not normalized:
        reasons.append("empty_candidate")
    if normalized and _has_repeated_text_cycle(normalized):
        reasons.append("repeated_text_cycle")
    if normalized and _is_foreground_context_anchor(
        text,
        left_context,
        right_context,
    ):
        reasons.append("context_anchor")
    return tuple(reasons)


def _is_foreground_context_anchor(
    candidate: str,
    left_context: str,
    right_context: str,
) -> bool:
    if _contains_context_anchor(candidate, left_context, right_context):
        return True
    normalized = _normalize_consensus_text(candidate)
    left = _normalize_consensus_text(left_context)
    right = _normalize_consensus_text(right_context)
    return bool(
        len(normalized) >= 5
        and (left.endswith(normalized) or right.startswith(normalized))
    )


def _candidate_text_consensus(texts: tuple[str, ...]) -> tuple[str, int]:
    grouped: dict[str, list[str]] = {}
    for text in texts:
        normalized = _normalize_consensus_text(text)
        if normalized:
            grouped.setdefault(normalized, []).append(text)
    if not grouped:
        return "", 0
    _normalized, variants = max(
        grouped.items(),
        key=lambda item: (len(item[1]), len(item[0])),
    )
    return variants[0], len(variants)


def _assess_omission_shadow_candidates(
    texts: tuple[str, ...],
    candidates: tuple[tuple[Any, ...], ...],
    coverages: tuple[float, ...],
    left_context: str,
    right_context: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> dict[str, object]:
    normalized = tuple(_normalize_consensus_text(text) for text in texts)
    core_characters = _common_ordered_items(normalized)
    core_coverage = tuple(
        round(len(core_characters) / len(text), 3) if text else 0.0
        for text in normalized
    )
    morpheme_sequences = tuple(
        tuple(
            morpheme.surface
            for morpheme in analyzer.analyze(text)
            if morpheme.part_of_speech[0] != "補助記号"
        )
        for text in texts
    )
    core_morphemes = _common_ordered_items(morpheme_sequences)
    disagreements = _candidate_morpheme_disagreements(morpheme_sequences)
    confidences = tuple(
        _finite_or_none(_external_segment_confidence(candidate))
        for candidate in candidates
    )
    language_model_scores = tuple(
        _mean_finite_attribute(candidate, "avg_logprob")
        for candidate in candidates
    )
    morphology_penalties = tuple(
        _morphological_structure_penalty(text, analyzer)
        + _grammatical_structure_penalty(text, analyzer)
        for text in texts
    )
    context_passed = bool(
        texts
        and all(coverage >= 0.5 for coverage in coverages)
        and not any(
            _contains_context_anchor(text, left_context, right_context)
            for text in texts
        )
    )
    confidence_passed = bool(
        confidences
        and all(value is not None and value >= 0.65 for value in confidences)
    )
    language_model_passed = bool(
        language_model_scores
        and all(
            value is not None and value >= -1.0
            for value in language_model_scores
        )
    )
    morphology_passed = bool(
        morphology_penalties and not any(morphology_penalties)
    )
    uncertain_noun_sequences = tuple(
        dict.fromkeys(
            sequence
            for text in texts
            if (sequence := _sentence_initial_uncertain_noun_sequence(text, analyzer))
        )
    )
    lexical_uncertainty_detected = bool(uncertain_noun_sequences)
    lexical_uncertainty_reasons = (
        ("unlexicalized_sentence_initial_noun_sequence",)
        if lexical_uncertainty_detected
        else ()
    )
    core_passed = bool(
        core_characters
        and core_morphemes
        and all(value >= 0.8 for value in core_coverage)
    )
    no_lexical_disagreement = not disagreements
    validation_passed = bool(
        context_passed
        and confidence_passed
        and language_model_passed
        and morphology_passed
        and core_passed
        and no_lexical_disagreement
        and not lexical_uncertainty_detected
    )
    reasons: list[str] = []
    if not core_passed:
        reasons.append("insufficient_core_consensus")
    if disagreements:
        reasons.append("lexical_candidate_disagreement")
    if not context_passed:
        reasons.append("context_coverage_validation_failed")
    if not confidence_passed:
        reasons.append("candidate_confidence_validation_failed")
    if not language_model_passed:
        reasons.append("language_model_validation_failed")
    if not morphology_passed:
        reasons.append("morphology_validation_failed")
    if lexical_uncertainty_detected:
        reasons.append("lexical_uncertainty_requires_review")
    if validation_passed:
        reasons.append("multi_evidence_validation_passed")
    return {
        "normalized_texts": normalized,
        "core_characters": "".join(core_characters),
        "core_coverage": core_coverage,
        "core_morphemes": core_morphemes,
        "disagreements": disagreements,
        "confidences": confidences,
        "language_model_scores": language_model_scores,
        "morphology_penalties": morphology_penalties,
        "context_passed": context_passed,
        "confidence_passed": confidence_passed,
        "language_model_passed": language_model_passed,
        "morphology_passed": morphology_passed,
        "lexical_uncertainty_detected": lexical_uncertainty_detected,
        "lexical_uncertainty_reasons": lexical_uncertainty_reasons,
        "uncertain_noun_sequences": uncertain_noun_sequences,
        "validation_passed": validation_passed,
        "reasons": tuple(reasons),
    }


def _normalize_consensus_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )


def _common_ordered_items(sequences):
    if not sequences:
        return ()
    common = tuple(sequences[0])
    for sequence in sequences[1:]:
        matcher = SequenceMatcher(None, common, tuple(sequence), autojunk=False)
        common = tuple(
            common[index]
            for block in matcher.get_matching_blocks()
            for index in range(block.a, block.a + block.size)
        )
    return common


def _candidate_morpheme_disagreements(
    sequences: tuple[tuple[str, ...], ...],
) -> tuple[TranscriptOmissionCandidateDisagreement, ...]:
    disagreements: list[TranscriptOmissionCandidateDisagreement] = []
    seen: set[tuple[str, str, str]] = set()
    for left_index, left in enumerate(sequences):
        for right_index in range(left_index + 1, len(sequences)):
            right = sequences[right_index]
            matcher = SequenceMatcher(None, left, right, autojunk=False)
            for operation, left_start, left_end, right_start, right_end in (
                matcher.get_opcodes()
            ):
                if operation == "equal":
                    continue
                left_fragment = "".join(left[left_start:left_end])
                right_fragment = "".join(right[right_start:right_end])
                if _normalize_consensus_text(
                    left_fragment
                ) == _normalize_consensus_text(right_fragment):
                    continue
                key = (operation, left_fragment, right_fragment)
                if key in seen:
                    continue
                seen.add(key)
                disagreements.append(
                    TranscriptOmissionCandidateDisagreement(
                        candidate_indexes=(left_index, right_index),
                        left_fragment=left_fragment,
                        right_fragment=right_fragment,
                        operation=operation,
                    )
                )
    return tuple(disagreements)


def _contains_context_anchor(
    candidate: str,
    left_context: str,
    right_context: str,
) -> bool:
    normalized = _normalize_consensus_text(candidate)
    left = _normalize_consensus_text(left_context)
    right = _normalize_consensus_text(right_context)
    return bool(
        (len(left) >= 8 and left[-8:] in normalized)
        or (len(right) >= 8 and right[:8] in normalized)
    )


def _sentence_initial_uncertain_noun_sequence(
    text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> str:
    morphemes = tuple(
        morpheme
        for morpheme in analyzer.analyze(text)
        if morpheme.part_of_speech[0] != "補助記号"
    )
    topic_index = next(
        (
            index
            for index, morpheme in enumerate(morphemes)
            if morpheme.surface == "は"
            and morpheme.part_of_speech[:2] == ("助詞", "係助詞")
        ),
        None,
    )
    if topic_index is None or topic_index < 2:
        return ""
    topic = morphemes[:topic_index]
    if not all(
        morpheme.part_of_speech[:2] == ("名詞", "普通名詞")
        and len(morpheme.surface) == 1
        and not morpheme.surface.isdigit()
        for morpheme in topic
    ):
        return ""
    return "".join(morpheme.surface for morpheme in topic)


def _finite_or_none(value: float) -> float | None:
    return value if isfinite(value) else None


def _local_decode_profile(candidate_index: int) -> dict[str, object]:
    profiles = (
        {"temperature": 0.0},
        {"temperature": 0.2, "beam_size": 1, "best_of": 5},
        {"temperature": 0.4, "beam_size": 1, "best_of": 5},
    )
    return dict(profiles[candidate_index % len(profiles)])


def _suspicious_internal_word_gap(
    segment: Any,
    minimum_gap_seconds: float,
    maximum_edge_confidence: float,
) -> _InternalWordGap | None:
    words = tuple(getattr(segment, "words", None) or ())
    suspicious: list[_InternalWordGap] = []
    for left, right in zip(words, words[1:]):
        gap = float(getattr(right, "start")) - float(getattr(left, "end"))
        probabilities = tuple(
            float(value)
            for value in (
                getattr(left, "probability", None),
                getattr(right, "probability", None),
            )
            if value is not None
        )
        if (
            gap >= minimum_gap_seconds
            and probabilities
            and min(probabilities) <= maximum_edge_confidence
        ):
            suspicious.append(
                _InternalWordGap(
                    start_seconds=float(getattr(left, "end")),
                    end_seconds=float(getattr(right, "start")),
                )
            )
    return (
        max(suspicious, key=lambda item: item.duration_seconds)
        if suspicious
        else None
    )


def _splice_external_retry(
    original: Any,
    retry_segments: tuple[Any, ...],
    clip_start: float,
    clip_end: float,
    internal_gap: _InternalWordGap,
) -> Any:
    original_words = tuple(getattr(original, "words", None) or ())
    retry_words = tuple(
        word
        for segment in retry_segments
        for word in (getattr(segment, "words", None) or ())
    )
    replacement_start = max(clip_start, internal_gap.start_seconds - 1.0)
    replacement_end = min(clip_end, internal_gap.end_seconds + 1.0)
    before = tuple(
        word
        for word in original_words
        if float(getattr(word, "end")) <= replacement_start
    )
    after = tuple(
        word
        for word in original_words
        if float(getattr(word, "start")) >= replacement_end
    )
    local_retry_words = tuple(
        word
        for word in retry_words
        if float(getattr(word, "end")) > replacement_start
        and float(getattr(word, "start")) < replacement_end
    )
    words = (*before, *local_retry_words, *after)
    retry_logprob = _mean_finite_attribute(retry_segments, "avg_logprob")
    return SimpleNamespace(
        text="".join(str(getattr(word, "word", "")) for word in words).strip(),
        start=float(getattr(original, "start")),
        end=float(getattr(original, "end")),
        words=words,
        avg_logprob=retry_logprob,
    )


def _extract_external_retry_window(
    original: Any,
    retry_segments: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Keep the retry segment that best overlaps one short original utterance."""
    start = float(getattr(original, "start"))
    end = float(getattr(original, "end"))
    overlapping = tuple(
        segment
        for segment in retry_segments
        if min(end, float(getattr(segment, "end")))
        > max(start, float(getattr(segment, "start")))
    )
    if not overlapping:
        return ()
    selected = max(
        overlapping,
        key=lambda segment: (
            min(end, float(getattr(segment, "end")))
            - max(start, float(getattr(segment, "start"))),
            -abs(
                (float(getattr(segment, "start")) + float(getattr(segment, "end")))
                / 2
                - (start + end) / 2
            ),
        ),
    )
    words = tuple(
        _copy_external_word(
            word,
            start=max(start, float(getattr(word, "start"))),
            end=min(end, float(getattr(word, "end"))),
        )
        for word in (getattr(selected, "words", None) or ())
        if min(end, float(getattr(word, "end")))
        > max(start, float(getattr(word, "start")))
    )
    if not words:
        return ()
    return (
        SimpleNamespace(
            text="".join(str(getattr(word, "word", "")) for word in words).strip(),
            start=start,
            end=end,
            words=words,
            avg_logprob=_mean_finite_attribute(retry_segments, "avg_logprob"),
        ),
    )


def _extract_external_retry_between_anchors(
    original: Any,
    retry_segments: tuple[Any, ...],
    left_anchor: Any | None,
    right_anchor: Any | None,
) -> tuple[Any, ...]:
    """Extract a short retry only when stable surrounding segments locate it."""
    if left_anchor is None or right_anchor is None or len(retry_segments) < 3:
        return _extract_external_retry_window(original, retry_segments)

    left_index, left_score = _best_anchor_match(
        left_anchor,
        retry_segments[:-1],
    )
    if left_index is None or left_score < 0.6:
        return ()
    right_offset, right_score = _best_anchor_match(
        right_anchor,
        retry_segments[left_index + 1 :],
    )
    if right_offset is None or right_score < 0.6:
        return ()
    right_index = left_index + 1 + right_offset
    between = retry_segments[left_index + 1 : right_index]
    if not between:
        return ()

    start = float(getattr(original, "start"))
    end = float(getattr(original, "end"))
    words = tuple(
        _copy_external_word(
            word,
            start=max(start, float(getattr(word, "start"))),
            end=min(end, float(getattr(word, "end"))),
        )
        for segment in between
        for word in (getattr(segment, "words", None) or ())
        if min(end, float(getattr(word, "end")))
        > max(start, float(getattr(word, "start")))
    )
    if not words:
        return ()
    return (
        SimpleNamespace(
            text="".join(str(getattr(word, "word", "")) for word in words).strip(),
            start=start,
            end=end,
            words=words,
            avg_logprob=_mean_finite_attribute(between, "avg_logprob"),
        ),
    )


def _best_anchor_match(
    anchor: Any,
    candidates: tuple[Any, ...],
) -> tuple[int | None, float]:
    anchor_text = "".join(_significant_characters(str(getattr(anchor, "text", ""))))
    if not anchor_text:
        return None, 0.0
    scored = tuple(
        (
            index,
            SequenceMatcher(
                None,
                anchor_text,
                "".join(
                    _significant_characters(str(getattr(candidate, "text", "")))
                ),
                autojunk=False,
            ).ratio(),
        )
        for index, candidate in enumerate(candidates)
    )
    return max(scored, key=lambda item: item[1], default=(None, 0.0))


def _has_ordered_text_anchors(
    retry_segments: tuple[Any, ...],
    left_anchor: Any,
    right_anchor: Any,
) -> bool:
    return _text_anchor_match_status(
        retry_segments,
        left_anchor,
        right_anchor,
    )[2]


def _text_anchor_match_status(
    retry_segments: tuple[Any, ...],
    left_anchor: Any,
    right_anchor: Any,
) -> tuple[bool, bool, bool]:
    candidate_text = "".join(
        _significant_characters(_external_text(retry_segments))
    )
    left_text = "".join(
        _significant_characters(str(getattr(left_anchor, "text", "")))
    )
    right_text = "".join(
        _significant_characters(str(getattr(right_anchor, "text", "")))
    )
    left_span, left_score = _best_approximate_text_span(candidate_text, left_text)
    right_span, right_score = _best_approximate_text_span(candidate_text, right_text)
    left_matched = bool(left_span is not None and left_score >= 0.7)
    right_matched = bool(right_span is not None and right_score >= 0.7)
    ordered = bool(
        left_matched
        and right_matched
        and left_span is not None
        and right_span is not None
        and left_span[1] <= right_span[0]
    )
    return left_matched, right_matched, ordered


def _best_approximate_text_span(
    text: str,
    anchor: str,
) -> tuple[tuple[int, int] | None, float]:
    if not text or not anchor:
        return None, 0.0
    minimum = max(1, round(len(anchor) * 0.8))
    maximum = min(len(text), max(minimum, round(len(anchor) * 1.2)))
    best_span: tuple[int, int] | None = None
    best_score = 0.0
    for length in range(minimum, maximum + 1):
        for start in range(0, len(text) - length + 1):
            score = SequenceMatcher(
                None,
                anchor,
                text[start : start + length],
                autojunk=False,
            ).ratio()
            if score > best_score:
                best_span = (start, start + length)
                best_score = score
    return best_span, best_score


def _preserve_retry_envelope(
    original: tuple[Any, ...],
    candidate: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Keep accepted local retries inside the original temporal/text envelope."""
    if not original or not candidate:
        return candidate

    original_start = float(getattr(original[0], "start"))
    original_end = float(getattr(original[-1], "end"))
    original_text = _external_text(original).rstrip()
    candidate_text = _external_text(candidate).rstrip()
    terminal_suffix = _terminal_punctuation_suffix(original_text)
    restore_terminal = bool(
        terminal_suffix
        and not _terminal_punctuation_suffix(candidate_text)
    )

    adjusted = list(candidate)
    first_words = tuple(getattr(adjusted[0], "words", None) or ())
    if first_words:
        first_word = first_words[0]
        first_words = (
            _copy_external_word(
                first_word,
                start=min(original_start, float(getattr(first_word, "start"))),
            ),
            *first_words[1:],
        )
    adjusted[0] = _copy_external_segment(
        adjusted[0],
        start=min(original_start, float(getattr(adjusted[0], "start"))),
        words=first_words,
    )
    last = adjusted[-1]
    words = tuple(getattr(last, "words", None) or ())
    if words and restore_terminal:
        final_word = words[-1]
        words = (
            *words[:-1],
            _copy_external_word(
                final_word,
                word=_restore_terminal_punctuation(
                    str(getattr(final_word, "word", "")),
                    terminal_suffix,
                ),
                end=max(original_end, float(getattr(final_word, "end"))),
            ),
        )
    elif words:
        final_word = words[-1]
        words = (
            *words[:-1],
            _copy_external_word(
                final_word,
                end=max(original_end, float(getattr(final_word, "end"))),
            ),
        )

    adjusted[-1] = _copy_external_segment(
        last,
        text=(
            _restore_terminal_punctuation(
                str(getattr(last, "text", "")),
                terminal_suffix,
            )
            if restore_terminal
            else str(getattr(last, "text", ""))
        ),
        end=max(original_end, float(getattr(last, "end"))),
        words=words,
    )
    return tuple(adjusted)


def _terminal_punctuation_suffix(text: str) -> str:
    closing = "\u300d\u300f\uff09)\u3011\uff3d]\u3009\u300b"
    index = len(text)
    while index and text[index - 1] in closing:
        index -= 1
    if index and text[index - 1] in "\u3002\uff1f\uff01?!":
        return text[index - 1 :]
    return ""


def _restore_terminal_punctuation(text: str, suffix: str) -> str:
    closing = "\u300d\u300f\uff09)\u3011\uff3d]\u3009\u300b"
    return f"{text.rstrip().rstrip(closing)}{suffix}"


def _copy_external_segment(segment: Any, **changes: Any) -> SimpleNamespace:
    values = {
        "text": str(getattr(segment, "text", "")),
        "start": float(getattr(segment, "start")),
        "end": float(getattr(segment, "end")),
        "words": tuple(getattr(segment, "words", None) or ()),
        "avg_logprob": getattr(segment, "avg_logprob", None),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _copy_external_word(source_word: Any, **changes: Any) -> SimpleNamespace:
    values = {
        "word": str(getattr(source_word, "word", "")),
        "start": float(getattr(source_word, "start")),
        "end": float(getattr(source_word, "end")),
        "probability": getattr(source_word, "probability", None),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _external_text_similarity(
    original: tuple[Any, ...],
    candidate: tuple[Any, ...],
) -> float:
    original_text = "".join(
        str(getattr(segment, "text", "")).replace(" ", "")
        for segment in original
    )
    candidate_text = "".join(
        str(getattr(segment, "text", "")).replace(" ", "")
        for segment in candidate
    )
    return SequenceMatcher(None, original_text, candidate_text, autojunk=False).ratio()


def _original_character_coverage(
    original: tuple[Any, ...],
    candidate: tuple[Any, ...],
) -> float:
    original_text = "".join(_significant_characters(_external_text(original)))
    candidate_text = "".join(_significant_characters(_external_text(candidate)))
    if not original_text:
        return 1.0
    matching = sum(
        block.size
        for block in SequenceMatcher(
            None,
            original_text,
            candidate_text,
            autojunk=False,
        ).get_matching_blocks()
    )
    return matching / len(original_text)


def _significant_characters(text: str) -> tuple[str, ...]:
    return tuple(
        character
        for character in text
        if not character.isspace() and character not in "、。！？!?"
    )


def _language_model_score_is_acceptable(
    original: tuple[Any, ...],
    candidate: tuple[Any, ...],
    maximum_regression: float,
) -> bool:
    original_score = _mean_finite_attribute(original, "avg_logprob")
    candidate_score = _mean_finite_attribute(candidate, "avg_logprob")
    if original_score is None or candidate_score is None:
        return True
    return candidate_score >= original_score - maximum_regression


def _mean_finite_attribute(
    items: tuple[Any, ...],
    attribute: str,
) -> float | None:
    values = tuple(
        float(value)
        for item in items
        if (value := getattr(item, attribute, None)) is not None
        and isfinite(float(value))
    )
    return sum(values) / len(values) if values else None


def _external_text(segments: tuple[Any, ...]) -> str:
    return "".join(str(getattr(segment, "text", "")) for segment in segments)


def _grammatical_structure_penalty(
    text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> int:
    """Count category-level connection errors without matching specific phrases."""
    morphemes = tuple(
        morpheme
        for morpheme in analyzer.analyze(text)
        if morpheme.part_of_speech[0] != "補助記号"
    )
    penalty = 0
    for previous, current in zip(morphemes, morphemes[1:]):
        previous_pos = previous.part_of_speech
        current_pos = current.part_of_speech

        # A case particle needs a nominal or predicate host. Another particle
        # immediately before it indicates that the host was probably dropped.
        if current_pos[:2] == ("助詞", "格助詞") and previous_pos[0] == "助詞":
            penalty += 1

        # The polite auxiliary attaches to an inflecting predicate. Sudachi's
        # conjugation type lets us validate that connection independently of words.
        if (
            current.conjugation_type == "助動詞-マス"
            and previous_pos[0] not in {"動詞", "助動詞"}
        ):
            penalty += 1

    if morphemes and _is_unfinished_morpheme(morphemes[-1]):
        penalty += 1
    return penalty


def _resolves_morphological_anomaly_text(
    original_text: str,
    candidate_text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    original_penalty = _morphological_structure_penalty(original_text, analyzer)
    if original_penalty == 0:
        return False
    candidate_grammar_is_acceptable = bool(
        _grammatical_structure_penalty(candidate_text, analyzer)
        <= _grammatical_structure_penalty(original_text, analyzer)
        or (
            _short_utterance_structure_penalty(original_text, analyzer) > 0
            and _is_short_elliptical_response(candidate_text, analyzer)
        )
    )
    return bool(
        _morphological_structure_penalty(candidate_text, analyzer)
        < original_penalty
        and candidate_grammar_is_acceptable
    )


def _morphological_structure_penalty(
    text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> int:
    return (
        _repeated_conjunctive_predicate_penalty(text, analyzer)
        + morphological_particle_chain_penalty(text, analyzer)
        + _short_utterance_structure_penalty(text, analyzer)
    )


def _short_utterance_structure_penalty(
    text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> int:
    normalized = "".join(str(text).split())
    if not normalized or len(normalized) > 8:
        return 0
    morphemes = tuple(
        item
        for item in analyzer.analyze(normalized)
        if item.part_of_speech and item.part_of_speech[0] != "補助記号"
    )
    if len(morphemes) < 2:
        return 0
    return int(
        morphemes[0].part_of_speech[0] == "助動詞"
        and not morphemes[0].conjugation_form.startswith("終止形")
        and morphemes[-1].part_of_speech[:2] == ("助詞", "終助詞")
    )


def _short_utterance_analysis_audit(
    segment_position: int,
    segment: Any,
    analyzer: JapaneseMorphologicalAnalyzer,
    detected: bool,
) -> ShortUtteranceAnalysisAudit | None:
    original_text = str(getattr(segment, "text", ""))
    normalized_text = "".join(original_text.split())
    if not normalized_text or len(normalized_text) > 8:
        return None
    morphemes = tuple(analyzer.analyze(normalized_text))
    return ShortUtteranceAnalysisAudit(
        segment_position=segment_position,
        time_range=TimeRange(
            float(getattr(segment, "start")),
            float(getattr(segment, "end")),
        ),
        original_text=original_text.strip(),
        normalized_text=normalized_text,
        morpheme_surfaces=tuple(item.surface for item in morphemes),
        morpheme_part_of_speech=tuple(
            tuple(item.part_of_speech) for item in morphemes
        ),
        morpheme_conjugation_types=tuple(
            item.conjugation_type for item in morphemes
        ),
        structure_penalty=_short_utterance_structure_penalty(
            original_text,
            analyzer,
        ),
        short_anomaly_detected=detected,
    )


def _is_short_elliptical_response(
    text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    """Recognize a compact nominal response whose predicate is omitted naturally."""
    normalized = "".join(str(text).split())
    if not normalized or len(normalized) > 8:
        return False
    morphemes = tuple(
        item
        for item in analyzer.analyze(normalized)
        if item.part_of_speech and item.part_of_speech[0] != "補助記号"
    )
    return bool(
        2 <= len(morphemes) <= 3
        and morphemes[0].part_of_speech[0] in {"名詞", "代名詞"}
        and morphemes[-1].part_of_speech[:2] == ("助詞", "格助詞")
    )


def _repeated_conjunctive_predicate_penalty(
    text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> int:
    morphemes = tuple(
        morpheme
        for morpheme in analyzer.analyze(text)
        if morpheme.part_of_speech[0] != "補助記号"
    )
    penalty = 0
    for previous, connective, following in zip(
        morphemes,
        morphemes[1:],
        morphemes[2:],
    ):
        if not (
            connective.part_of_speech[:2] == ("助詞", "接続助詞")
            and previous.part_of_speech[0] in {"動詞", "形容詞"}
            and following.part_of_speech[0] == previous.part_of_speech[0]
        ):
            continue
        previous_lemma = previous.dictionary_form or previous.normalized_form
        following_lemma = following.dictionary_form or following.normalized_form
        if (
            previous_lemma
            and previous_lemma == following_lemma
            and previous.conjugation_type == following.conjugation_type
        ):
            penalty += 1
    return penalty


def _is_unfinished_morpheme(morpheme: JapaneseMorpheme) -> bool:
    pos = morpheme.part_of_speech
    if pos[:2] in {
        ("助詞", "格助詞"),
        ("助詞", "係助詞"),
        ("助詞", "接続助詞"),
    }:
        return True
    return (
        pos[0] in {"動詞", "形容詞", "助動詞"}
        and any(
            form in morpheme.conjugation_form
            for form in ("未然形", "連用形")
        )
    )


def _deletes_content_or_particle(
    original_text: str,
    candidate_text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    original = _meaningful_morpheme_keys(analyzer.analyze(original_text))
    candidate = _meaningful_morpheme_keys(analyzer.analyze(candidate_text))
    return any(
        tag in {"delete", "replace"} and original_start < original_end
        for tag, original_start, original_end, _, _ in SequenceMatcher(
            None,
            original,
            candidate,
            autojunk=False,
        ).get_opcodes()
    )


def _meaningful_morpheme_keys(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> tuple[tuple[str, str], ...]:
    meaningful_parts = {
        "名詞",
        "代名詞",
        "動詞",
        "形容詞",
        "副詞",
        "連体詞",
        "助詞",
    }
    return tuple(
        (
            morpheme.dictionary_form or morpheme.normalized_form or morpheme.surface,
            morpheme.part_of_speech[0],
        )
        for morpheme in morphemes
        if morpheme.part_of_speech[0] in meaningful_parts
    )


def _text_boundary_word_indexes(text: str, words: tuple[Word, ...]) -> tuple[int, ...]:
    boundary_offsets: set[int] = set()
    offset = 0
    for character in text:
        if character.isspace():
            if offset:
                boundary_offsets.add(offset)
        else:
            offset += 1
    indexes: list[int] = []
    word_offset = 0
    for index, word in enumerate(words[:-1], start=1):
        word_offset += len("".join(word.text.split()))
        if word_offset in boundary_offsets:
            indexes.append(index)
    return tuple(indexes)


__all__ = [
    "DEFAULT_VAD_MIN_SILENCE_MS",
    "DEFAULT_WHISPER_BEAM_SIZE",
    "DEFAULT_WHISPER_BEST_OF",
    "DEFAULT_WHISPER_COMPUTE_TYPE",
    "DEFAULT_WHISPER_CONDITION_ON_PREVIOUS_TEXT",
    "DEFAULT_WHISPER_DEVICE",
    "DEFAULT_WHISPER_HALLUCINATION_SILENCE_THRESHOLD_SECONDS",
    "DEFAULT_WHISPER_LANGUAGE",
    "DEFAULT_WHISPER_MODEL_SIZE",
    "DEFAULT_WHISPER_TEMPERATURE",
    "DEFAULT_WHISPER_VAD_FILTER",
    "DEFAULT_WHISPER_WORD_TIMESTAMPS",
    "DEFAULT_RETRY_CONFIDENCE_THRESHOLD",
    "DEFAULT_RETRY_CONTEXT_CONFIDENCE_THRESHOLD",
    "DEFAULT_RETRY_MAX_SEGMENTS",
    "DEFAULT_RETRY_MIN_CONFIDENCE_IMPROVEMENT",
    "FasterWhisperDependencyError",
    "FasterWhisperTranscriber",
]
