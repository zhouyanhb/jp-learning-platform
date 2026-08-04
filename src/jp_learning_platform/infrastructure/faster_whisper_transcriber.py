"""faster-whisper transcription adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from math import inf, isfinite
from types import SimpleNamespace
from typing import Any

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_WHISPER_TRANSCRIPTION_CONFIG,
)
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    JapaneseMorpheme,
    JapaneseMorphologicalAnalyzer,
    SudachiMorphologicalAnalyzer,
)
from jp_learning_platform.workflow.whisper_stage import (
    WhisperRetryDecision,
    WhisperTranscript,
    WhisperTranscriptionRequest,
)

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
        selected_segments, retry_decisions = self._retry_low_confidence_segments(
            model,
            source_path,
            tuple(external_segments),
        )

        segments: list[Segment] = []
        for external_segment in selected_segments:
            text = str(getattr(external_segment, "text", "")).strip()
            if text:
                segments.append(self._convert_segment(len(segments), external_segment))

        return WhisperTranscript(
            source_path=request.source_path,
            segments=tuple(segments),
            retry_decisions=retry_decisions,
        )

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
    ) -> tuple[tuple[Any, ...], tuple[WhisperRetryDecision, ...]]:
        if self.retry_max_segments <= 0:
            return segments, ()

        selected: list[Any] = []
        decisions: list[WhisperRetryDecision] = []
        retries = 0
        reliable_context = ""
        for segment in segments:
            confidence = _external_segment_confidence((segment,))
            internal_gap = _suspicious_internal_word_gap(
                segment,
                self.retry_internal_word_gap_seconds,
                self.retry_internal_gap_edge_confidence,
            )
            replacement = (segment,)
            if (
                (confidence < self.retry_confidence_threshold or internal_gap is not None)
                and retries < self.retry_max_segments
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
                retry_segments, _info = model.transcribe(
                    source_path,
                    **self._retry_options(
                        segment,
                        reliable_context,
                        disable_vad=internal_gap is not None,
                        clip_timestamps=(clip_start, clip_end),
                    ),
                )
                candidate = tuple(retry_segments)
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
                candidate_confidence = _external_segment_confidence(candidate)
                candidate_internal_gap = _suspicious_internal_word_gap(
                    candidate[0],
                    self.retry_internal_word_gap_seconds,
                    self.retry_internal_gap_edge_confidence,
                ) if candidate else None
                confidence_improved = bool(
                    candidate
                    and isfinite(candidate_confidence)
                    and (
                        not isfinite(confidence)
                        or candidate_confidence
                        >= confidence + self.retry_min_confidence_improvement
                    )
                )
                gap_repaired = bool(
                    internal_gap is not None
                    and candidate
                    and isfinite(candidate_confidence)
                    and candidate_confidence
                    >= confidence - self.retry_max_confidence_regression
                    and (
                        candidate_internal_gap is None
                        or candidate_internal_gap.duration_seconds
                        <= internal_gap.duration_seconds
                        * self.retry_minimum_gap_reduction_ratio
                    )
                    and _external_text_similarity((segment,), candidate)
                    >= self.retry_minimum_text_similarity
                    and _original_character_coverage((segment,), candidate)
                    >= self.retry_minimum_original_character_coverage
                    and _language_model_score_is_acceptable(
                        (segment,),
                        candidate,
                        self.retry_max_language_model_regression,
                    )
                    and not self._introduces_grammatical_degradation(
                        (segment,),
                        candidate,
                    )
                    and self._content_deletion_has_sufficient_confidence(
                        (segment,),
                        candidate,
                        confidence,
                        candidate_confidence,
                    )
                )
                if internal_gap is not None and candidate:
                    decisions.append(
                        self._retry_decision(
                            original=(segment,),
                            candidate=candidate,
                            internal_gap=internal_gap,
                            original_confidence=confidence,
                            candidate_confidence=candidate_confidence,
                            candidate_internal_gap=candidate_internal_gap,
                            accepted=gap_repaired,
                        )
                    )
                if gap_repaired or (
                    internal_gap is None and confidence_improved
                ):
                    replacement = candidate
                    confidence = candidate_confidence

            selected.extend(replacement)
            if confidence >= self.retry_context_confidence_threshold:
                reliable_context = "".join(
                    str(getattr(item, "text", "")).strip()
                    for item in replacement
                )

        return tuple(selected), tuple(decisions)

    def _retry_decision(
        self,
        *,
        original: tuple[Any, ...],
        candidate: tuple[Any, ...],
        internal_gap: _InternalWordGap,
        original_confidence: float,
        candidate_confidence: float,
        candidate_internal_gap: _InternalWordGap | None,
        accepted: bool,
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
        reasons: list[str] = []
        if candidate_confidence < (
            original_confidence - self.retry_max_confidence_regression
        ):
            reasons.append("confidence_regression")
        if (
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
        if deletes_content and candidate_confidence < (
            original_confidence + self.retry_min_confidence_improvement
        ):
            reasons.append("content_deletion_without_confidence_gain")
        return WhisperRetryDecision(
            time_range=TimeRange(
                internal_gap.start_seconds,
                internal_gap.end_seconds,
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

            self._model = WhisperModel(
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
