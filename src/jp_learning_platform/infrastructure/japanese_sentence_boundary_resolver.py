"""Pause-aware Japanese sentence boundary resolver."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_SENTENCE_BOUNDARY_CONFIG,
)
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    JapaneseMorpheme,
    JapaneseMorphologicalAnalyzer,
)
from jp_learning_platform.workflow.sentence_boundary_stage import (
    CrossSegmentMergeDecision,
    CrossSegmentMergeEvidence,
    SentenceBoundaryDecision,
    SentenceBoundaryResolution,
    SentenceBoundaryResolutionRequest,
    SpeakerTurnCandidate,
)

DEFAULT_SENTENCE_BOUNDARY_MIN_PAUSE_SECONDS = (
    DEFAULT_SENTENCE_BOUNDARY_CONFIG.min_pause_seconds
)
DEFAULT_SENTENCE_BOUNDARY_TERMINAL_MARKS = (
    DEFAULT_SENTENCE_BOUNDARY_CONFIG.terminal_marks
)
DEFAULT_SENTENCE_BOUNDARY_FINAL_SUFFIXES = (
    DEFAULT_SENTENCE_BOUNDARY_CONFIG.sentence_final_suffixes
)

_CLOSING_QUOTES = frozenset(("」", "』", "）", ")", "】", "］", "]", "〉", "》"))
_STRONG_PAUSE_SECONDS = 1.5
_SENTENCE_FINAL_PARTICLES = frozenset(("か", "ね", "よ", "な"))
_INDEPENDENT_DISCOURSE_STARTS = (
    "でも",
    "しかし",
    "けれども",
    "ところで",
)
_RESPONSE_STARTS = (
    "はい",
    "うん",
    "そうですか",
    "そうですね",
    "なるほど",
    "ごめんなさい",
    "すみません",
)
_SHORT_RESPONSE_UTTERANCES = frozenset(_RESPONSE_STARTS)
_ELLIPTICAL_TURN_ENDS = ("て", "で", "し", "から", "けど", "が", "ので")


@dataclass(frozen=True, slots=True)
class JapaneseSentenceBoundaryResolver:
    """Split sentence-sized ASR segments using punctuation and word-level pauses."""

    min_pause_seconds: float = DEFAULT_SENTENCE_BOUNDARY_MIN_PAUSE_SECONDS
    terminal_marks: tuple[str, ...] = DEFAULT_SENTENCE_BOUNDARY_TERMINAL_MARKS
    sentence_final_suffixes: tuple[str, ...] = DEFAULT_SENTENCE_BOUNDARY_FINAL_SUFFIXES
    morphological_analyzer: JapaneseMorphologicalAnalyzer | None = None

    def resolve(
        self,
        request: SentenceBoundaryResolutionRequest,
    ) -> SentenceBoundaryResolution:
        if not isinstance(request, SentenceBoundaryResolutionRequest):
            raise TypeError("request must be a SentenceBoundaryResolutionRequest.")

        segments = _reattach_syntactic_prefixes(
            request.segments,
            self.morphological_analyzer,
        )
        segments, early_cross_segment_merges = (
            _merge_high_confidence_cross_segment_continuations(
                segments,
                self.morphological_analyzer,
                max_fragment_gap_seconds=(
                    DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_cross_segment_grammar_gap_seconds
                ),
            )
        )
        segments = _merge_adjacent_dependent_continuations(
            segments,
            self.morphological_analyzer,
        )
        segments = _merge_adjacent_connective_continuations(
            segments,
            self.morphological_analyzer,
        )
        resolved_segments: list[Segment] = []
        decisions: list[SentenceBoundaryDecision] = []
        speaker_turn_candidates: list[SpeakerTurnCandidate] = []
        internal_cross_segment_merges: list[CrossSegmentMergeDecision] = []
        for segment in segments:
            (
                resolved_segment,
                segment_decisions,
                segment_turn_candidates,
                segment_cross_merges,
            ) = self._resolve_segment(segment)
            resolved_segments.append(resolved_segment)
            decisions.extend(segment_decisions)
            speaker_turn_candidates.extend(segment_turn_candidates)
            internal_cross_segment_merges.extend(segment_cross_merges)

        resolved_segments, late_cross_segment_merges = (
            _merge_high_confidence_cross_segment_continuations(
                tuple(resolved_segments),
                self.morphological_analyzer,
            )
        )
        resolved_segments, adjacent_sentence_merges = (
            _merge_high_confidence_adjacent_sentences(
                resolved_segments,
                self.morphological_analyzer,
            )
        )
        cross_segment_merges = _deduplicate_cross_segment_merges(
            (
                *early_cross_segment_merges,
                *internal_cross_segment_merges,
                *late_cross_segment_merges,
                *adjacent_sentence_merges,
            )
        )

        resolved_segments, cross_segment_decisions = (
            _resolve_cross_segment_numbering_bodies(
                tuple(resolved_segments),
                self.morphological_analyzer,
            )
        )
        decisions.extend(cross_segment_decisions)

        source_text = _compact_text("".join(segment.text for segment in request.segments))
        resolved_text = _compact_text("".join(segment.text for segment in resolved_segments))
        if not _is_ordered_subsequence(source_text, resolved_text):
            raise ValueError(
                "Sentence boundary resolution must not delete or reorder source text."
            )

        return SentenceBoundaryResolution(
            source_path=request.source_path,
            segments=tuple(resolved_segments),
            decisions=tuple(decisions),
            speaker_turn_candidates=tuple(speaker_turn_candidates),
            cross_segment_merges=cross_segment_merges,
        )

    def _resolve_segment(
        self,
        segment: Segment,
    ) -> tuple[
        Segment,
        tuple[SentenceBoundaryDecision, ...],
        tuple[SpeakerTurnCandidate, ...],
        tuple[CrossSegmentMergeDecision, ...],
    ]:
        sentences = segment.sentences or (
            Sentence(
                text=segment.text,
                time_range=segment.time_range,
                words=(),
            ),
        )

        resolved_sentences: list[Sentence] = []
        decisions: list[SentenceBoundaryDecision] = []
        speaker_turn_candidates: list[SpeakerTurnCandidate] = []
        cross_segment_merges: list[CrossSegmentMergeDecision] = []
        for sentence_index, sentence in enumerate(sentences):
            (
                sentence_parts,
                sentence_decisions,
                sentence_turn_candidates,
                sentence_cross_merges,
            ) = self._split_sentence(segment.position, sentence_index, sentence)
            resolved_sentences.extend(sentence_parts)
            decisions.extend(sentence_decisions)
            speaker_turn_candidates.extend(sentence_turn_candidates)
            cross_segment_merges.extend(sentence_cross_merges)

        if tuple(resolved_sentences) == sentences:
            return (
                segment,
                tuple(decisions),
                tuple(speaker_turn_candidates),
                tuple(cross_segment_merges),
            )

        return (
            Segment(
                position=segment.position,
                text="".join(sentence.text for sentence in resolved_sentences),
                time_range=segment.time_range,
                sentences=tuple(resolved_sentences),
            ),
            tuple(decisions),
            tuple(speaker_turn_candidates),
            tuple(cross_segment_merges),
        )

    def _split_sentence(
        self,
        segment_position: int,
        sentence_index: int,
        sentence: Sentence,
    ) -> tuple[
        tuple[Sentence, ...],
        tuple[SentenceBoundaryDecision, ...],
        tuple[SpeakerTurnCandidate, ...],
        tuple[CrossSegmentMergeDecision, ...],
    ]:
        if len(sentence.words) < 2:
            return (sentence,), (), (), ()

        boundaries: list[int] = []
        asr_boundaries = _asr_boundary_word_indexes(sentence)
        structured_numbering_starts = _structured_numbering_start_indexes(
            sentence.words,
            self.morphological_analyzer,
            asr_boundaries,
        )
        decisions: list[SentenceBoundaryDecision] = []
        speaker_turn_candidates: list[SpeakerTurnCandidate] = []
        cross_segment_merges: list[CrossSegmentMergeDecision] = []
        chunk_start = 0
        for word_index in range(len(sentence.words) - 1):
            turn_reason = _speaker_turn_candidate_reason(
                sentence.words,
                chunk_start,
                word_index,
                word_index + 1 in asr_boundaries,
                self.morphological_analyzer,
            )
            cross_merge = self._internal_asr_merge_decision(
                segment_position,
                sentence,
                chunk_start,
                word_index,
                word_index + 1 in asr_boundaries,
            )
            if cross_merge is not None:
                reason = None
                cross_segment_merges.append(cross_merge)
            elif word_index + 1 in structured_numbering_starts:
                reason = "structured_numbering_sequence"
            else:
                reason = self._boundary_reason(
                    sentence.words,
                    chunk_start,
                    word_index,
                    word_index + 1 in asr_boundaries,
                    turn_reason,
                )
            left_text = _words_text(sentence.words[chunk_start : word_index + 1])
            right_text = _words_text(sentence.words[word_index + 1 :])
            gap_seconds = _effective_word_gap_seconds(
                sentence.words[word_index],
                sentence.words[word_index + 1],
            )
            if turn_reason is not None and left_text and right_text:
                speaker_turn_candidates.append(
                    SpeakerTurnCandidate(
                        segment_position=segment_position,
                        sentence_index=sentence_index,
                        word_index=word_index,
                        gap_seconds=max(gap_seconds, 0.0),
                        reason=turn_reason,
                        left_text=left_text,
                        right_text=right_text,
                        boundary_accepted=reason is not None,
                    )
                )
            if reason is None:
                continue

            if not left_text or not right_text:
                continue

            boundaries.append(word_index)
            decisions.append(
                SentenceBoundaryDecision(
                    segment_position=segment_position,
                    sentence_index=sentence_index,
                    word_index=word_index,
                    gap_seconds=max(gap_seconds, 0.0),
                    reason=reason,
                    left_text=left_text,
                    right_text=right_text,
                )
            )
            chunk_start = word_index + 1

        if not boundaries:
            return (
                (sentence,),
                (),
                tuple(speaker_turn_candidates),
                tuple(cross_segment_merges),
            )

        parts: list[Sentence] = []
        start_index = 0
        for boundary, decision in zip(boundaries, decisions, strict=True):
            words = sentence.words[start_index : boundary + 1]
            if _is_extended_alignment_word(words[-1]):
                words = _trim_extended_boundary_word(words)
            parts.append(
                _sentence_from_words(
                    words,
                    is_question=(
                        decision.reason in {
                            "sentence_final_question_particle",
                            "question_answer_transition",
                            "asr_question_answer_transition",
                        }
                        or _ends_with_question_mark(decision.left_text)
                    ),
                    asr_boundary_word_indexes=_slice_asr_boundaries(
                        sentence.asr_boundary_word_indexes,
                        start_index,
                        boundary + 1,
                    ),
                )
            )
            start_index = boundary + 1

        parts.append(
            _sentence_from_words(
                sentence.words[start_index:],
                asr_boundary_word_indexes=_slice_asr_boundaries(
                    sentence.asr_boundary_word_indexes,
                    start_index,
                    len(sentence.words),
                ),
            )
        )
        return (
            tuple(parts),
            tuple(decisions),
            tuple(speaker_turn_candidates),
            tuple(cross_segment_merges),
        )

    def _internal_asr_merge_decision(
        self,
        segment_position: int,
        sentence: Sentence,
        chunk_start: int,
        word_index: int,
        is_asr_boundary: bool,
    ) -> CrossSegmentMergeDecision | None:
        if not is_asr_boundary or self.morphological_analyzer is None:
            return None
        left = _sentence_from_words(sentence.words[chunk_start : word_index + 1])
        right = _sentence_from_words(sentence.words[word_index + 1 :])
        gap_seconds = _effective_word_gap_seconds(
            sentence.words[word_index],
            sentence.words[word_index + 1],
        )
        score, evidence, has_fragment = _cross_segment_merge_score(
            left,
            right,
            gap_seconds,
            self.morphological_analyzer,
        )
        config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
        allowed_gap = (
            config.max_cross_segment_fragment_gap_seconds
            if has_fragment
            else config.max_cross_segment_grammar_gap_seconds
        )
        if (
            gap_seconds > allowed_gap
            or score < config.cross_segment_merge_score_threshold
            or _starts_independent_discourse(right.text)
            or _starts_topic_shift_expression(
                self.morphological_analyzer.analyze(_compact_text(right.text))
            )
            or (
                _starts_cross_segment_response(
                    right.text,
                    self.morphological_analyzer.analyze(_compact_text(right.text)),
                )
                and not has_fragment
            )
        ):
            return None
        return CrossSegmentMergeDecision(
            left_segment_position=segment_position,
            right_segment_position=segment_position,
            word_index=word_index,
            left_end_seconds=left.time_range.end_seconds,
            right_start_seconds=right.time_range.start_seconds,
            gap_seconds=max(gap_seconds, 0.0),
            score=score,
            reason=(
                "cross_asr_word_fragment_reconstruction"
                if has_fragment
                else "cross_asr_syntactic_continuation"
            ),
            left_text=left.text,
            right_text=right.text,
            evidence=evidence,
        )

    def _boundary_reason(
        self,
        words: tuple[Word, ...],
        chunk_start: int,
        word_index: int,
        is_asr_boundary: bool = False,
        speaker_turn_reason: str | None = None,
    ) -> str | None:
        current_text = _word_text(words[word_index])
        left_text = _words_text(words[chunk_start : word_index + 1])
        if _ends_with_terminal_mark(current_text, self.terminal_marks):
            return "terminal_mark"

        if _ends_with_terminal_mark(left_text, self.terminal_marks):
            return "terminal_mark"

        gap_seconds = _effective_word_gap_seconds(
            words[word_index], words[word_index + 1]
        )
        right_text = _words_text(words[word_index + 1 :])
        has_speaker_turn_boundary = bool(
            speaker_turn_reason is not None
            and self.morphological_analyzer is not None
            and _speaker_turn_supports_sentence_boundary(
                left_text,
                right_text,
                speaker_turn_reason,
                self.morphological_analyzer,
            )
        )
        if (
            _starts_with_sentence_final_particle(right_text)
            and not has_speaker_turn_boundary
        ):
            return None
        if _starts_with_dependent_continuation(right_text):
            return None

        if current_text == "か" and _has_question_boundary_evidence(
            words,
            chunk_start,
            word_index,
            self.sentence_final_suffixes,
        ):
            return "sentence_final_question_particle"

        if right_text[:1].isdigit() and not _looks_sentence_final(
            left_text,
            self.sentence_final_suffixes,
        ):
            return None

        has_trusted_asr_restart = bool(
            self.morphological_analyzer is not None
            and is_asr_boundary
            and _word_gap_seconds(words[word_index], words[word_index + 1])
            < self.min_pause_seconds
            and _has_high_confidence_clause_restart(
                left_text,
                right_text,
                self.morphological_analyzer,
                self.sentence_final_suffixes,
            )
        )
        has_extended_conjunctive_continuation = bool(
            self.morphological_analyzer is not None
            and _is_extended_alignment_word(words[word_index])
            and _word_gap_seconds(words[word_index], words[word_index + 1])
            <= DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_dependent_continuation_gap_seconds
            and not has_trusted_asr_restart
            and not has_speaker_turn_boundary
            and _has_conjunctive_predicate_continuation(
                left_text,
                right_text,
                self.morphological_analyzer,
            )
        )
        if has_extended_conjunctive_continuation:
            return None
        if (
            self.morphological_analyzer is not None
            and gap_seconds
            <= DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_syntactic_dependency_gap_seconds
            and not has_trusted_asr_restart
            and not has_speaker_turn_boundary
            and _has_cross_boundary_morphological_dependency(
                left_text,
                right_text,
                self.morphological_analyzer,
            )
        ):
            return None

        if gap_seconds >= _STRONG_PAUSE_SECONDS:
            return "strong_pause"

        if self.morphological_analyzer is not None:
            left = self.morphological_analyzer.analyze(_compact_text(left_text))
            right = self.morphological_analyzer.analyze(_compact_text(right_text))
            relative_pause = gap_seconds / max(
                _reasonable_word_duration_seconds(words[word_index]),
                _reasonable_word_duration_seconds(words[word_index + 1]),
                0.1,
            )
            if (
                left
                and right
                and _is_question_clause(left)
                and _starts_independent_clause(right)
                and gap_seconds
                >= self.min_pause_seconds
                * DEFAULT_SENTENCE_BOUNDARY_CONFIG.question_answer_min_pause_ratio
                and relative_pause
                >= DEFAULT_SENTENCE_BOUNDARY_CONFIG.question_answer_min_relative_pause
            ):
                return "question_answer_transition"
            if (
                left
                and right
                and _ends_in_connective_form(left)
                and _starts_independent_response(right)
                and gap_seconds
                >= self.min_pause_seconds
                * DEFAULT_SENTENCE_BOUNDARY_CONFIG.connective_response_min_pause_ratio
                and relative_pause
                >= DEFAULT_SENTENCE_BOUNDARY_CONFIG.connective_response_min_relative_pause
            ):
                return "connective_response_transition"
            if (
                left
                and right
                and _is_complete_clause(left)
                and _starts_structural_restart(right)
                and gap_seconds >= self.min_pause_seconds
            ):
                return "structural_restart"

        if _looks_sentence_final(left_text, self.sentence_final_suffixes):
            if gap_seconds >= self.min_pause_seconds:
                return "pause_after_sentence_final"

        if is_asr_boundary and self.morphological_analyzer is not None:
            left = self.morphological_analyzer.analyze(_compact_text(left_text))
            right = self.morphological_analyzer.analyze(_compact_text(right_text))
            if left and right and _starts_structural_restart(right):
                return "asr_structural_restart"
            if left and right and _is_complete_clause(left) and _starts_independent_clause(right):
                if _is_question_clause(left) and _is_strong_independent_start(right[0]):
                    return "asr_question_answer_transition"
                return "asr_complete_clause_independent_start"

        if has_speaker_turn_boundary:
            return "speaker_turn_supported_boundary"

        return None


def _sentence_from_words(
    words: tuple[Word, ...],
    *,
    is_question: bool = False,
    asr_boundary_word_indexes: tuple[int, ...] = (),
) -> Sentence:
    if not words:
        raise ValueError("words must not be empty.")

    return Sentence(
        text=_words_text(words),
        time_range=TimeRange(
            words[0].time_range.start_seconds,
            words[-1].time_range.end_seconds,
        ),
        words=words,
        is_question=is_question,
        asr_boundary_word_indexes=asr_boundary_word_indexes,
    )


def _word_gap_seconds(current: Word, nxt: Word) -> float:
    return nxt.time_range.start_seconds - current.time_range.end_seconds


def _effective_word_gap_seconds(current: Word, nxt: Word) -> float:
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    maximum_speech_duration = max(
        len(_word_text(current)), 1
    ) * config.max_aligned_word_seconds_per_character
    estimated_end = min(
        current.time_range.end_seconds,
        current.time_range.start_seconds + maximum_speech_duration,
    )
    return nxt.time_range.start_seconds - estimated_end


def _words_text(words: tuple[Word, ...]) -> str:
    return "".join(_word_text(word) for word in words).strip()


def _asr_boundary_word_indexes(sentence: Sentence) -> frozenset[int]:
    """Map whitespace retained by ASR text back to aligned-word boundaries."""
    compact_cursor = 0
    text_boundaries: set[int] = set()
    for character in sentence.text:
        if character.isspace():
            if compact_cursor:
                text_boundaries.add(compact_cursor)
        else:
            compact_cursor += 1

    indexes: set[int] = set(sentence.asr_boundary_word_indexes)
    word_cursor = 0
    for index, word in enumerate(sentence.words[:-1]):
        word_cursor += len(_compact_text(_word_text(word)))
        if word_cursor in text_boundaries:
            indexes.add(index + 1)
    return frozenset(indexes)


def _slice_asr_boundaries(
    boundaries: tuple[int, ...],
    start: int,
    end: int,
) -> tuple[int, ...]:
    return tuple(index - start for index in boundaries if start < index < end)


def _reasonable_word_duration_seconds(word: Word) -> float:
    return max(
        min(
            word.time_range.duration_seconds,
            max(len(_word_text(word)), 1)
            * DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_aligned_word_seconds_per_character,
        ),
        0.1,
    )


def _word_text(word: Word) -> str:
    return unicodedata.normalize("NFKC", word.text).strip()


@dataclass(frozen=True, slots=True)
class _NumberingCandidate:
    word_index: int
    value: int
    start_seconds: float
    has_lexical_host: bool


@dataclass(frozen=True, slots=True)
class _OwnedSentence:
    owner: int
    sentence: Sentence


@dataclass(frozen=True, slots=True)
class _EmbeddedNumberingCandidate:
    entry_index: int
    word_index: int
    value: int
    start_seconds: float


def _structured_numbering_start_indexes(
    words: tuple[Word, ...],
    analyzer: JapaneseMorphologicalAnalyzer | None,
    asr_boundaries: frozenset[int],
) -> frozenset[int]:
    if analyzer is None or len(words) < 3:
        return frozenset()
    all_candidates = tuple(
        candidate
        for index, word in enumerate(words)
        if (
            candidate := _numbering_candidate(
                words,
                index,
                word,
                analyzer,
                asr_boundaries,
            )
        )
        is not None
    )
    candidates = tuple(
        candidate for candidate in all_candidates if not candidate.has_lexical_host
    )
    if len(candidates) < DEFAULT_SENTENCE_BOUNDARY_CONFIG.numbering_region_min_sequence_length:
        return frozenset()

    starts: set[int] = set()
    run_start = 0
    while run_start < len(candidates):
        run_end = run_start + 1
        while run_end < len(candidates) and _continues_numbering_run(
            candidates[run_end - 1],
            candidates[run_end],
            words,
        ):
            run_end += 1
        run = candidates[run_start:run_end]
        if len(run) >= DEFAULT_SENTENCE_BOUNDARY_CONFIG.numbering_region_min_sequence_length:
            starts.update(item.word_index for item in run if item.word_index > 0)
            restart = next(
                (
                    item
                    for item in all_candidates
                    if item.word_index > run[-1].word_index
                ),
                None,
            )
            if restart is not None and _is_numbering_restart(
                run,
                restart,
                words,
                asr_boundaries,
            ):
                starts.add(restart.word_index)
        run_start = run_end
    return frozenset(starts)


def _numbering_candidate(
    words: tuple[Word, ...],
    index: int,
    word: Word,
    analyzer: JapaneseMorphologicalAnalyzer,
    blocked_host_boundaries: frozenset[int] = frozenset(),
) -> _NumberingCandidate | None:
    normalized = _compact_text(_word_text(word))
    digits = "".join(character for character in normalized if character.isdecimal())
    if not digits or any(
        not character.isdecimal()
        and not unicodedata.category(character).startswith("P")
        for character in normalized
    ):
        return None
    value = int(digits)
    if value <= 0:
        return None
    return _NumberingCandidate(
        index,
        value,
        word.time_range.start_seconds,
        _number_has_lexical_host(
            words,
            index,
            analyzer,
            blocked_host_boundaries,
        ),
    )


def _number_has_lexical_host(
    words: tuple[Word, ...],
    index: int,
    analyzer: JapaneseMorphologicalAnalyzer,
    blocked_host_boundaries: frozenset[int] = frozenset(),
) -> bool:
    if index + 1 in blocked_host_boundaries:
        return False
    number_text = _compact_text(_word_text(words[index]))
    max_lookahead_words = 8
    for end in range(index + 1, min(len(words), index + max_lookahead_words) + 1):
        analyzed = analyzer.analyze(
            _compact_text(_words_text(words[index:end]))
        )
        host = _number_host_morpheme(analyzed, number_text)
        if host is not None:
            return _is_quantity_host(host)
    return False


def _number_host_morpheme(
    morphemes: tuple[JapaneseMorpheme, ...],
    number_text: str,
) -> JapaneseMorpheme | None:
    if not morphemes:
        return None
    first = morphemes[0]
    if _is_numeric_morpheme(first):
        return morphemes[1] if len(morphemes) > 1 else None

    # Sudachi can analyze compact quantities such as "1人" as one noun.
    if first.surface.startswith(number_text) and first.surface != number_text:
        return first
    return None


def _is_quantity_host(morpheme: JapaneseMorpheme) -> bool:
    if not morpheme.part_of_speech:
        return False
    major = morpheme.part_of_speech[0]
    minor = morpheme.part_of_speech[1] if len(morpheme.part_of_speech) > 1 else ""
    detail = morpheme.part_of_speech[2] if len(morpheme.part_of_speech) > 2 else ""
    return bool(
        (
            major == "接尾辞"
            and minor == "名詞的"
            and detail in {"助数詞", "一般"}
        )
        or (major == "名詞" and "助数詞可能" in morpheme.part_of_speech)
        or (
            major == "名詞"
            and minor == "普通名詞"
            and detail in {"副詞可能", "助数詞可能"}
            and any(character.isdecimal() for character in morpheme.surface)
            and not morpheme.surface.isdecimal()
        )
    )


def _continues_numbering_run(
    previous: _NumberingCandidate,
    current: _NumberingCandidate,
    words: tuple[Word, ...],
) -> bool:
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    body = _compact_text(
        _words_text(words[previous.word_index + 1 : current.word_index])
    )
    return bool(
        current.value == previous.value + 1
        and current.start_seconds - previous.start_seconds
        <= config.numbering_region_max_item_gap_seconds
        and len(body) >= config.numbering_region_min_body_characters
    )


def _is_numbering_restart(
    run: tuple[_NumberingCandidate, ...],
    candidate: _NumberingCandidate,
    words: tuple[Word, ...],
    asr_boundaries: frozenset[int],
) -> bool:
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    previous = run[-1]
    body = _compact_text(
        _words_text(words[previous.word_index + 1 : candidate.word_index])
    )
    if (
        candidate.value > run[0].value
        or candidate.start_seconds - previous.start_seconds
        > config.numbering_region_max_item_gap_seconds
        or len(body) < config.numbering_region_min_body_characters
    ):
        return False
    if candidate.word_index in asr_boundaries:
        return True
    prior = words[candidate.word_index - 1]
    current = words[candidate.word_index]
    return _effective_word_gap_seconds(prior, current) >= config.min_pause_seconds


def _resolve_cross_segment_numbering_bodies(
    segments: tuple[Segment, ...],
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> tuple[tuple[Segment, ...], tuple[SentenceBoundaryDecision, ...]]:
    if analyzer is None:
        return segments, ()
    entries = [
        _OwnedSentence(owner, sentence)
        for owner, segment in enumerate(segments)
        for sentence in segment.sentences
    ]
    decisions: list[SentenceBoundaryDecision] = []
    entries, quantity_decisions = _merge_cross_sentence_quantities(
        entries,
        segments,
        analyzer,
    )
    decisions.extend(quantity_decisions)
    entries, embedded_decisions = _split_confirmed_embedded_numbering(
        entries,
        segments,
        analyzer,
    )
    decisions.extend(embedded_decisions)
    entries, retrospective_decisions = _attach_confirmed_numbering_bodies(
        entries,
        segments,
        analyzer,
    )
    decisions.extend(retrospective_decisions)
    run: list[int] = []
    index = 0
    while index < len(entries):
        current = entries[index]
        numbered = _leading_structural_number_and_body(current.sentence, analyzer)
        if numbered is None:
            run.clear()
            index += 1
            continue
        value, body = numbered
        if run and value == run[-1] + 1:
            run.append(value)
        else:
            run = [value]

        if len(run) >= DEFAULT_SENTENCE_BOUNDARY_CONFIG.numbering_region_min_sequence_length:
            if not body and index + 1 < len(entries):
                nxt = entries[index + 1]
                if _can_attach_numbering_body(current.sentence, nxt.sentence, analyzer):
                    entries[index] = _OwnedSentence(
                        current.owner,
                        _join_numbering_sentences(current.sentence, nxt.sentence),
                    )
                    decisions.append(
                        _cross_numbering_decision(
                            segments,
                            current,
                            nxt.sentence,
                            "cross_asr_numbering_body",
                        )
                    )
                    del entries[index + 1]
                    current = entries[index]

            expected = run[-1] + 1
            if index + 1 < len(entries):
                nxt = entries[index + 1]
                candidate_index = _embedded_expected_number_index(
                    nxt.sentence.words,
                    expected,
                    analyzer,
                )
                if candidate_index is not None and _can_move_numbering_prefix(
                    current.sentence,
                    nxt.sentence,
                    candidate_index,
                    analyzer,
                ):
                    prefix = nxt.sentence.words[:candidate_index]
                    remainder = nxt.sentence.words[candidate_index:]
                    entries[index] = _OwnedSentence(
                        current.owner,
                        _join_numbering_sentences(
                            current.sentence,
                            _sentence_from_words(prefix),
                        ),
                    )
                    entries[index + 1] = _OwnedSentence(
                        nxt.owner,
                        _sentence_from_words(remainder),
                    )
                    decisions.append(
                        _cross_numbering_decision(
                            segments,
                            current,
                            entries[index + 1].sentence,
                            "cross_asr_numbering_prefix",
                        )
                    )
        index += 1
    return _rebuild_owned_segments(entries), tuple(decisions)


def _merge_cross_sentence_quantities(
    entries: list[_OwnedSentence],
    segments: tuple[Segment, ...],
    analyzer: JapaneseMorphologicalAnalyzer,
) -> tuple[list[_OwnedSentence], tuple[SentenceBoundaryDecision, ...]]:
    decisions: list[SentenceBoundaryDecision] = []
    index = 0
    while index + 1 < len(entries):
        left = entries[index]
        right = entries[index + 1]
        numbered = _leading_number_and_body(left.sentence.text)
        gap_seconds = (
            right.sentence.time_range.start_seconds
            - left.sentence.time_range.end_seconds
        )
        if (
            numbered is None
            or numbered[1]
            or len(left.sentence.words) != 1
            or not right.sentence.words
            or gap_seconds
            > DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_cross_segment_grammar_gap_seconds
        ):
            index += 1
            continue

        combined_words = (*left.sentence.words, *right.sentence.words)
        if not _number_has_lexical_host(combined_words, 0, analyzer):
            index += 1
            continue

        decisions.append(
            _cross_numbering_decision(
                segments,
                left,
                right.sentence,
                "cross_sentence_quantity_unit",
            )
        )
        entries[index] = _OwnedSentence(
            left.owner,
            _join_numbering_sentences(left.sentence, right.sentence),
        )
        del entries[index + 1]
    return entries, tuple(decisions)


def _split_confirmed_embedded_numbering(
    entries: list[_OwnedSentence],
    segments: tuple[Segment, ...],
    analyzer: JapaneseMorphologicalAnalyzer,
) -> tuple[list[_OwnedSentence], tuple[SentenceBoundaryDecision, ...]]:
    candidates = tuple(
        _EmbeddedNumberingCandidate(
            entry_index,
            word_index,
            candidate.value,
            candidate.start_seconds,
        )
        for entry_index, entry in enumerate(entries)
        for word_index, word in enumerate(entry.sentence.words)
        if (
            candidate := _numbering_candidate(
                entry.sentence.words,
                word_index,
                word,
                analyzer,
                _asr_boundary_word_indexes(entry.sentence),
            )
        ) is not None
        and not candidate.has_lexical_host
    )
    confirmed: set[tuple[int, int]] = set()
    run: list[_EmbeddedNumberingCandidate] = []
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    for item in (*candidates, None):
        continues = bool(
            item is not None
            and run
            and item.value == run[-1].value + 1
            and item.start_seconds - run[-1].start_seconds
            <= config.numbering_region_max_item_gap_seconds
            and len(_text_between_numbering_candidates(entries, run[-1], item))
            >= config.numbering_region_min_body_characters
        )
        if item is not None and (not run or continues):
            run.append(item)
            continue
        if len(run) >= config.numbering_region_min_sequence_length and run[0].value == 1:
            confirmed.update(
                (candidate.entry_index, candidate.word_index)
                for candidate in run
                if candidate.word_index > 0
            )
        run = [item] if item is not None else []

    decisions: list[SentenceBoundaryDecision] = []
    for entry_index, word_index in sorted(confirmed, reverse=True):
        entry = entries[entry_index]
        prefix = entry.sentence.words[:word_index]
        remainder = entry.sentence.words[word_index:]
        if not prefix or not remainder:
            continue
        if entry_index > 0 and _can_move_numbering_prefix(
            entries[entry_index - 1].sentence,
            entry.sentence,
            word_index,
            analyzer,
        ):
            previous = entries[entry_index - 1]
            decisions.append(
                _cross_numbering_decision(
                    segments,
                    previous,
                    _sentence_from_words(remainder),
                    "cross_asr_numbering_prefix",
                )
            )
            entries[entry_index - 1] = _OwnedSentence(
                previous.owner,
                _join_numbering_sentences(
                    previous.sentence,
                    _sentence_from_words(prefix),
                ),
            )
            entries[entry_index] = _OwnedSentence(
                entry.owner,
                _sentence_from_words(remainder),
            )
            continue
        entries[entry_index : entry_index + 1] = (
            _OwnedSentence(entry.owner, _sentence_from_words(prefix)),
            _OwnedSentence(entry.owner, _sentence_from_words(remainder)),
        )
    decisions.reverse()
    return entries, tuple(decisions)


def _text_between_numbering_candidates(
    entries: list[_OwnedSentence],
    left: _EmbeddedNumberingCandidate,
    right: _EmbeddedNumberingCandidate,
) -> str:
    if left.entry_index == right.entry_index:
        words = entries[left.entry_index].sentence.words[
            left.word_index + 1 : right.word_index
        ]
        return _compact_text(_words_text(words))
    parts = [
        _words_text(
            entries[left.entry_index].sentence.words[left.word_index + 1 :]
        )
    ]
    parts.extend(
        entries[index].sentence.text
        for index in range(left.entry_index + 1, right.entry_index)
    )
    parts.append(
        _words_text(entries[right.entry_index].sentence.words[: right.word_index])
    )
    return _compact_text("".join(parts))


def _attach_confirmed_numbering_bodies(
    entries: list[_OwnedSentence],
    segments: tuple[Segment, ...],
    analyzer: JapaneseMorphologicalAnalyzer,
) -> tuple[list[_OwnedSentence], tuple[SentenceBoundaryDecision, ...]]:
    candidates: list[tuple[int, int, bool]] = []
    index = 0
    while index < len(entries):
        numbered = _leading_structural_number_and_body(
            entries[index].sentence,
            analyzer,
        )
        if numbered is None:
            index += 1
            continue
        value, body = numbered
        standalone = not body
        candidates.append((index, value, standalone))
        has_separate_body = bool(
            standalone
            and index + 1 < len(entries)
            and _leading_number_and_body(entries[index + 1].sentence.text) is None
        )
        index += 2 if has_separate_body else 1

    confirmed_indexes: set[int] = set()
    run: list[tuple[int, int, bool]] = []
    for item in (*candidates, None):
        if item is not None and (not run or item[1] == run[-1][1] + 1):
            run.append(item)
            continue
        if (
            len(run)
            >= DEFAULT_SENTENCE_BOUNDARY_CONFIG.numbering_region_min_sequence_length
            and run[0][1] == 1
        ):
            confirmed_indexes.update(
                candidate_index
                for candidate_index, _value, standalone in run
                if standalone
            )
        run = [item] if item is not None else []

    decisions: list[SentenceBoundaryDecision] = []
    for candidate_index in sorted(confirmed_indexes, reverse=True):
        if candidate_index + 1 >= len(entries):
            continue
        number = entries[candidate_index]
        body = entries[candidate_index + 1]
        if not _can_attach_numbering_body(number.sentence, body.sentence, analyzer):
            continue
        entries[candidate_index] = _OwnedSentence(
            number.owner,
            _join_numbering_sentences(number.sentence, body.sentence),
        )
        decisions.append(
            _cross_numbering_decision(
                segments,
                number,
                body.sentence,
                "cross_asr_numbering_body",
            )
        )
        del entries[candidate_index + 1]
    decisions.reverse()
    return entries, tuple(decisions)


def _leading_number_and_body(text: str) -> tuple[int, str] | None:
    normalized = _compact_text(text)
    end = 0
    while end < len(normalized) and normalized[end].isdecimal():
        end += 1
    if end == 0:
        return None
    body = normalized[end:]
    while body and unicodedata.category(body[0]).startswith("P"):
        body = body[1:]
    return int(normalized[:end]), body


def _leading_structural_number_and_body(
    sentence: Sentence,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> tuple[int, str] | None:
    numbered = _leading_number_and_body(sentence.text)
    if numbered is None:
        return None
    number_text = str(numbered[0])
    host = _number_host_morpheme(
        analyzer.analyze(_compact_text(sentence.text)),
        number_text,
    )
    if host is not None and _is_quantity_host(host):
        return None
    return numbered


def _can_attach_numbering_body(
    number: Sentence,
    body: Sentence,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    if (
        not body.words
        or _leading_number_and_body(body.text) is not None
        or body.time_range.start_seconds - number.time_range.end_seconds
        > config.numbering_region_max_item_gap_seconds
    ):
        return False
    analysis = analyzer.analyze(_compact_text(body.text))
    return bool(
        analysis
        and not _starts_with_functional_fragment(analysis[0])
        and not _starts_with_unfinished_conjunction(analysis)
    )


def _embedded_expected_number_index(
    words: tuple[Word, ...],
    expected: int,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> int | None:
    for index, word in enumerate(words[1:], start=1):
        candidate = _numbering_candidate(words, index, word, analyzer)
        if candidate is not None and candidate.value == expected:
            return index
    return None


def _can_move_numbering_prefix(
    previous: Sentence,
    current: Sentence,
    number_index: int,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    prefix_words = current.words[:number_index]
    remainder_words = current.words[number_index:]
    if not prefix_words or not remainder_words:
        return False
    prefix = analyzer.analyze(_compact_text(_words_text(prefix_words)))
    left = analyzer.analyze(_compact_text(previous.text))
    remainder = analyzer.analyze(_compact_text(_words_text(remainder_words)))
    if not left or not prefix or not remainder:
        return False
    if not (
        _crosses_adnominal_dependency(left[-1], prefix[0])
        or _forms_contextual_adnominal_dependency(
            previous.text,
            _words_text(prefix_words),
            analyzer,
        )
    ):
        return False
    combined = analyzer.analyze(
        _compact_text(f"{previous.text}{_words_text(prefix_words)}")
    )
    return bool(
        combined
        and _is_valid_prefix_partition(combined, remainder)
        and _leading_number_and_body(_words_text(remainder_words)) is not None
    )


def _forms_contextual_adnominal_dependency(
    left_text: str,
    prefix_text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    left = analyzer.analyze(_compact_text(left_text))
    combined = analyzer.analyze(_compact_text(f"{left_text}{prefix_text}"))
    if not left or not combined:
        return False
    boundary = len(_compact_text(left_text))
    cursor = 0
    contextual_left: JapaneseMorpheme | None = None
    contextual_right: JapaneseMorpheme | None = None
    for index, morpheme in enumerate(combined):
        cursor += len(morpheme.surface)
        if cursor == boundary:
            contextual_left = morpheme
            if index + 1 < len(combined):
                contextual_right = combined[index + 1]
            break
        if cursor > boundary:
            return False
    return bool(
        contextual_left is not None
        and contextual_right is not None
        and contextual_right.part_of_speech
        and _is_adnominal_host(contextual_right)
        and contextual_left.dictionary_form == left[-1].dictionary_form
        and contextual_left.conjugation_type
        not in {"助動詞-ダ", "助動詞-デス"}
        and "連体形" in contextual_left.conjugation_form
        and "連体形" not in left[-1].conjugation_form
    )


def _is_adnominal_host(morpheme: JapaneseMorpheme) -> bool:
    if not morpheme.part_of_speech:
        return False
    major = morpheme.part_of_speech[0]
    if major in {"名詞", "代名詞"}:
        return True
    return bool(
        major == "形状詞"
        and len(morpheme.part_of_speech) > 1
        and morpheme.part_of_speech[1] == "助動詞語幹"
    )


def _join_numbering_sentences(left: Sentence, right: Sentence) -> Sentence:
    return Sentence(
        text=f"{left.text}{right.text}",
        time_range=TimeRange(
            left.time_range.start_seconds,
            right.time_range.end_seconds,
        ),
        words=(*left.words, *right.words),
        is_question=left.is_question,
        asr_boundary_word_indexes=(
            *left.asr_boundary_word_indexes,
            len(left.words),
            *(len(left.words) + item for item in right.asr_boundary_word_indexes),
        ),
    )


def _cross_numbering_decision(
    segments: tuple[Segment, ...],
    owned: _OwnedSentence,
    right: Sentence,
    reason: str,
) -> SentenceBoundaryDecision:
    left = owned.sentence
    return SentenceBoundaryDecision(
        segment_position=segments[owned.owner].position,
        sentence_index=0,
        word_index=max(len(left.words) - 1, 0),
        gap_seconds=max(
            right.time_range.start_seconds - left.time_range.end_seconds,
            0.0,
        ),
        reason=reason,
        left_text=left.text,
        right_text=right.text,
    )


def _rebuild_owned_segments(
    entries: list[_OwnedSentence],
) -> tuple[Segment, ...]:
    grouped: list[tuple[int, list[Sentence]]] = []
    for entry in entries:
        if grouped and grouped[-1][0] == entry.owner:
            grouped[-1][1].append(entry.sentence)
        else:
            grouped.append((entry.owner, [entry.sentence]))
    return tuple(
        Segment(
            position=position,
            text="".join(item.text for item in sentences),
            time_range=TimeRange(
                sentences[0].time_range.start_seconds,
                sentences[-1].time_range.end_seconds,
            ),
            sentences=tuple(sentences),
        )
        for position, (_owner, sentences) in enumerate(grouped)
    )


def _is_extended_alignment_word(word: Word) -> bool:
    text = _word_text(word)
    duration = word.time_range.duration_seconds
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    return (
        duration >= config.extended_word_duration_seconds
        and duration / max(len(text), 1)
        >= config.extended_word_seconds_per_character
    )


def _has_question_boundary_evidence(
    words: tuple[Word, ...],
    chunk_start: int,
    word_index: int,
    sentence_final_suffixes: tuple[str, ...],
) -> bool:
    preceding_clause = _words_text(words[chunk_start:word_index])
    return (
        word_index + 1 < len(words)
        and _effective_word_gap_seconds(words[word_index], words[word_index + 1])
        >= DEFAULT_SENTENCE_BOUNDARY_CONFIG.min_pause_seconds
        and _looks_sentence_final(preceding_clause, sentence_final_suffixes)
    )


def _trim_extended_boundary_word(words: tuple[Word, ...]) -> tuple[Word, ...]:
    """Exclude alignment-held trailing silence from a sentence boundary word."""
    last = words[-1]
    text = _word_text(last)
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    maximum_duration = max(
        len(text),
        1,
    ) * config.max_aligned_word_seconds_per_character
    if last.time_range.duration_seconds <= maximum_duration:
        return words

    return (
        *words[:-1],
        Word(
            text=last.text,
            time_range=TimeRange(
                last.time_range.start_seconds,
                last.time_range.start_seconds + maximum_duration,
            ),
            confidence=last.confidence,
        ),
    )


def _speaker_turn_candidate_reason(
    words: tuple[Word, ...],
    chunk_start: int,
    word_index: int,
    is_asr_boundary: bool,
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> str | None:
    if analyzer is None:
        return None
    left_text = _words_text(words[chunk_start : word_index + 1])
    right_text = _words_text(words[word_index + 1 :])
    if not left_text or not right_text:
        return None
    left = analyzer.analyze(_compact_text(left_text))
    right = analyzer.analyze(_compact_text(right_text))
    if not left or not right:
        return None
    combined_text = _compact_text(f"{left_text}{right_text}")
    if not _has_morpheme_boundary_at(
        combined_text,
        len(_compact_text(left_text)),
        analyzer,
    ):
        return None

    normalized_right = _compact_text(right_text)
    if normalized_right.startswith(_RESPONSE_STARTS):
        return "independent_response_start"
    if _is_short_response(left_text, left) and _starts_independent_clause(right):
        return "response_handoff"

    gap_seconds = _effective_word_gap_seconds(
        words[word_index],
        words[word_index + 1],
    )
    if (
        _is_interrogative_clause(left)
        and (
            _starts_independent_clause(right)
            or _looks_like_short_asr_response(
                words[word_index + 1 :],
                right,
                gap_seconds,
                is_asr_boundary,
            )
        )
        and (is_asr_boundary or gap_seconds >= 0.3)
    ):
        return "question_answer_transition"
    if (
        gap_seconds >= 0.6
        and _starts_independent_clause(right)
    ):
        return "pause_supported_turn"
    return None


def _speaker_turn_supports_sentence_boundary(
    left_text: str,
    right_text: str,
    reason: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    left = analyzer.analyze(_compact_text(left_text))
    right = analyzer.analyze(_compact_text(right_text))
    if not left or not right:
        return False
    has_dependency = _has_cross_boundary_morphological_dependency(
        left_text,
        right_text,
        analyzer,
    )
    if reason == "response_handoff":
        return bool(
            _is_short_response(left_text, left)
            and _starts_topic_restart(right_text, right)
            and _is_complete_clause(right)
        )
    if reason == "question_answer_transition":
        return bool(
            not has_dependency
            and (
                _starts_independent_clause(right)
                or _is_short_terminal_response_analysis(right)
            )
        )
    if not _starts_independent_clause(right):
        return False
    if reason == "independent_response_start":
        return bool(
            not has_dependency
            and (
                _is_complete_clause(left)
                or _compact_text(left_text).endswith(_ELLIPTICAL_TURN_ENDS)
            )
        )
    if reason == "pause_supported_turn":
        left_last = left[-1]
        is_terminal_particle = bool(
            len(left_last.part_of_speech) > 1
            and left_last.part_of_speech[0] == "助詞"
            and left_last.part_of_speech[1] == "終助詞"
        )
        return bool(
            is_terminal_particle
            or _compact_text(left_text).endswith(_ELLIPTICAL_TURN_ENDS)
        )
    return False


def _looks_like_short_asr_response(
    words: tuple[Word, ...],
    analyzed: tuple[JapaneseMorpheme, ...],
    gap_seconds: float,
    is_asr_boundary: bool,
) -> bool:
    if not words or not is_asr_boundary or gap_seconds < 0.3:
        return False
    text = _words_text(words)
    duration = words[-1].time_range.end_seconds - words[0].time_range.start_seconds
    return bool(
        len(_compact_text(text)) <= 8
        and duration <= 1.5
        and _is_short_terminal_response_analysis(analyzed)
    )


def _is_short_terminal_response_analysis(
    analyzed: tuple[JapaneseMorpheme, ...],
) -> bool:
    if len(analyzed) < 2:
        return False
    last = analyzed[-1]
    return bool(
        last.part_of_speech[:2] == ("助詞", "終助詞")
        and any(
            item.part_of_speech
            and item.part_of_speech[0] in {"感動詞", "助動詞", "代名詞", "名詞"}
            for item in analyzed[:-1]
        )
    )


def _is_short_response(
    text: str,
    analyzed: tuple[JapaneseMorpheme, ...],
) -> bool:
    normalized = _compact_text(text)
    del analyzed
    return normalized in _SHORT_RESPONSE_UTTERANCES


def _starts_topic_restart(
    text: str,
    analyzed: tuple[JapaneseMorpheme, ...],
) -> bool:
    normalized = _compact_text(text)
    if (
        not analyzed
        or not analyzed[0].part_of_speech
        or analyzed[0].part_of_speech[0] != "接続詞"
    ):
        return False
    if normalized.startswith(f"{analyzed[0].surface}、"):
        return True
    following = next(
        (
            morpheme
            for morpheme in analyzed[1:]
            if not (
                morpheme.part_of_speech
                and morpheme.part_of_speech[0] == "補助記号"
            )
        ),
        None,
    )
    return bool(
        following
        and following.part_of_speech
        and following.part_of_speech[0] in {"副詞", "感動詞"}
    )


def _has_high_confidence_clause_restart(
    left_text: str,
    right_text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
    sentence_final_suffixes: tuple[str, ...],
) -> bool:
    left = analyzer.analyze(_compact_text(left_text))
    right = analyzer.analyze(_compact_text(right_text))
    if not left or not right or not _starts_independent_clause(right):
        return False

    left_last = left[-1]
    is_terminal_particle = bool(
        len(left_last.part_of_speech) > 1
        and left_last.part_of_speech[0] == "助詞"
        and left_last.part_of_speech[1] == "終助詞"
    )
    is_terminal_auxiliary = bool(
        left_last.part_of_speech
        and left_last.part_of_speech[0] == "助動詞"
        and any(
            form in left_last.conjugation_form
            for form in ("終止形", "意志推量形")
        )
    )
    if _looks_sentence_final(left_text, sentence_final_suffixes) or is_terminal_particle:
        return True
    return bool(
        is_terminal_auxiliary
        and not _forms_contextual_adnominal_dependency(
            left_text,
            right_text,
            analyzer,
        )
    )


def _looks_sentence_final(text: str, suffixes: tuple[str, ...]) -> bool:
    normalized = _trim_closing_quotes(_compact_text(text))
    if not normalized:
        return False

    return any(normalized.endswith(suffix) for suffix in suffixes)


def _ends_with_terminal_mark(text: str, terminal_marks: tuple[str, ...]) -> bool:
    normalized = _trim_closing_quotes(_compact_text(text))
    if not normalized:
        return False

    return normalized.endswith(terminal_marks)


def _ends_with_question_mark(text: str) -> bool:
    normalized = _trim_closing_quotes(_compact_text(text))
    return normalized.endswith(("?", "？"))


def _trim_closing_quotes(text: str) -> str:
    normalized = text
    while normalized and normalized[-1] in _CLOSING_QUOTES:
        normalized = normalized[:-1]
    return normalized


def _compact_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        character for character in normalized.strip() if not character.isspace()
    )


def _is_ordered_subsequence(source: str, output: str) -> bool:
    output_index = 0
    for character in source:
        output_index = output.find(character, output_index)
        if output_index < 0:
            return False
        output_index += 1
    return True


def _starts_with_sentence_final_particle(text: str) -> bool:
    normalized = _compact_text(text)
    return bool(normalized) and normalized[0] in _SENTENCE_FINAL_PARTICLES


def _starts_with_dependent_continuation(text: str) -> bool:
    normalized = _compact_text(text)
    return normalized.startswith(
        DEFAULT_SENTENCE_BOUNDARY_CONFIG.dependent_continuation_prefixes
    )


def _reattach_syntactic_prefixes(
    segments: tuple[Segment, ...],
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> tuple[Segment, ...]:
    """Move a grammatically dependent ASR prefix back to the preceding segment."""
    if analyzer is None:
        return segments

    adjusted = list(segments)
    for index in range(1, len(adjusted)):
        previous = adjusted[index - 1]
        current = adjusted[index]
        if not previous.sentences or not current.sentences:
            continue
        left = previous.sentences[-1]
        right = current.sentences[0]
        if not right.words:
            continue
        boundary_gap = right.time_range.start_seconds - left.time_range.end_seconds
        if (
            boundary_gap < 0
            or boundary_gap > DEFAULT_SENTENCE_BOUNDARY_CONFIG.min_pause_seconds
            or _ends_with_terminal_mark(left.text, ("。", "？", "！", "?", "!"))
        ):
            continue

        prefix_count = _prefix_before_reliable_pause(right.words)
        if prefix_count <= 0 or prefix_count >= len(right.words):
            continue
        prefix_words = right.words[:prefix_count]
        if not _is_morphological_continuation(
            left.text,
            _words_text(prefix_words),
            analyzer,
        ):
            continue

        remaining_words = right.words[prefix_count:]
        attached_analysis = analyzer.analyze(
            _compact_text(f"{left.text}{_words_text(prefix_words)}")
        )
        remaining_analysis = analyzer.analyze(
            _compact_text(_words_text(remaining_words))
        )
        if not _is_valid_prefix_partition(
            attached_analysis,
            remaining_analysis,
            has_morpheme_boundary=_has_morpheme_boundary_at(
                _compact_text(
                    f"{left.text}{_words_text(prefix_words)}"
                    f"{_words_text(remaining_words)}"
                ),
                len(_compact_text(f"{left.text}{_words_text(prefix_words)}")),
                analyzer,
            ),
        ):
            continue
        attached = Sentence(
            text=f"{left.text}{_words_text(prefix_words)}",
            time_range=TimeRange(
                left.time_range.start_seconds,
                prefix_words[-1].time_range.end_seconds,
            ),
            words=(*left.words, *prefix_words),
            asr_boundary_word_indexes=(
                *left.asr_boundary_word_indexes,
                len(left.words),
                *(
                    len(left.words) + boundary
                    for boundary in right.asr_boundary_word_indexes
                    if boundary < prefix_count
                ),
            ),
        )
        previous_sentences = (*previous.sentences[:-1], attached)
        adjusted[index - 1] = Segment(
            position=previous.position,
            text="".join(sentence.text for sentence in previous_sentences),
            time_range=TimeRange(
                previous.time_range.start_seconds,
                prefix_words[-1].time_range.end_seconds,
            ),
            sentences=previous_sentences,
        )

        remaining = _sentence_from_words(
            remaining_words,
            asr_boundary_word_indexes=tuple(
                boundary - prefix_count
                for boundary in right.asr_boundary_word_indexes
                if boundary > prefix_count
            ),
        )
        current_sentences = (remaining, *current.sentences[1:])
        adjusted[index] = Segment(
            position=current.position,
            text="".join(sentence.text for sentence in current_sentences),
            time_range=TimeRange(
                remaining.time_range.start_seconds,
                current.time_range.end_seconds,
            ),
            sentences=current_sentences,
        )

    return tuple(
        Segment(
            position=index,
            text=segment.text,
            time_range=segment.time_range,
            sentences=segment.sentences,
        )
        for index, segment in enumerate(adjusted)
    )


def _prefix_before_reliable_pause(words: tuple[Word, ...]) -> int:
    for index in range(len(words) - 1):
        if _effective_word_gap_seconds(words[index], words[index + 1]) >= (
            DEFAULT_SENTENCE_BOUNDARY_CONFIG.min_pause_seconds
        ):
            return index + 1
    return 0


def _is_morphological_continuation(
    left_text: str,
    prefix_text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    left = analyzer.analyze(_compact_text(left_text))
    right = analyzer.analyze(_compact_text(prefix_text))
    if not left or not right:
        return False

    left_last = left[-1]
    right_first = right[0]
    left_pos = left_last.part_of_speech[0] if left_last.part_of_speech else ""
    right_pos = right_first.part_of_speech[0] if right_first.part_of_speech else ""
    functional_right = right_pos in {"助動詞", "助詞", "接尾辞"}
    predicate_right = right_pos in {"動詞", "形容詞", "助動詞"}
    inflected_left = left_pos in {"動詞", "形容詞", "助動詞"} and any(
        form in left_last.conjugation_form for form in ("未然形", "連用形")
    )
    return (
        functional_right
        or (inflected_left and (functional_right or predicate_right))
        or (left_pos in {"副詞", "助詞"} and predicate_right)
    )


def _contextual_right_morpheme(
    left_text: str,
    prefix_text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> JapaneseMorpheme | None:
    compact_left = _compact_text(left_text)
    combined = analyzer.analyze(f"{compact_left}{_compact_text(prefix_text)}")
    cursor = 0
    for morpheme in combined:
        end = cursor + len(morpheme.surface)
        if cursor >= len(compact_left) or end > len(compact_left):
            return morpheme
        cursor = end
    return None


def _merge_high_confidence_cross_segment_continuations(
    segments: tuple[Segment, ...],
    analyzer: JapaneseMorphologicalAnalyzer | None,
    *,
    max_fragment_gap_seconds: float | None = None,
) -> tuple[tuple[Segment, ...], tuple[CrossSegmentMergeDecision, ...]]:
    """Remove only ASR boundaries with strong lexical or grammatical evidence."""
    if analyzer is None:
        return segments, ()

    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    merged: list[Segment] = []
    decisions: list[CrossSegmentMergeDecision] = []
    for segment in segments:
        if not merged or not segment.sentences:
            merged.append(segment)
            continue
        previous = merged[-1]
        if not previous.sentences:
            merged.append(segment)
            continue

        left = previous.sentences[-1]
        right = segment.sentences[0]
        gap_seconds = right.time_range.start_seconds - left.time_range.end_seconds
        left_analysis = (
            analyzer.analyze(_compact_text(left.text))
            if analyzer is not None
            else ()
        )
        right_analysis = (
            analyzer.analyze(_compact_text(right.text))
            if analyzer is not None
            else ()
        )
        has_fragment_reconstruction = bool(
            left_analysis
            and right_analysis
            and _reconstructs_cross_segment_fragment(
                _compact_text(left.text),
                _compact_text(right.text),
                left_analysis,
                analyzer,
            )
        )
        expected_number = _expected_numbering_restart(left, analyzer)
        has_dependent_quotative_right = _starts_dependent_quotative_continuation(
            right.text
        )
        has_coordinated_condition = _forms_coordinated_condition(
            left_analysis,
            right_analysis,
            right.text,
        )
        if (
            not left.words
            or not right.words
            or not left_analysis
            or not right_analysis
            or _ends_with_terminal_mark(
                left.text,
                (*config.terminal_marks, "?", "!"),
            )
            or (
                _starts_independent_discourse(right.text)
                and not has_coordinated_condition
            )
            or _starts_topic_shift_expression(right_analysis)
            or (
                _starts_cross_segment_response(right.text, right_analysis)
                and not has_fragment_reconstruction
            )
            or _has_cross_segment_speaker_turn_veto(
                left.text,
                right.text,
                left_analysis,
                right_analysis,
                gap_seconds,
            )
            or (
                _is_functional_continuation(right_analysis[0])
                and not has_fragment_reconstruction
                and not has_dependent_quotative_right
            )
            or (
                expected_number is not None
                and _embedded_expected_number_index(
                    right.words,
                    expected_number,
                    analyzer,
                )
                is not None
            )
        ):
            merged.append(segment)
            continue
        score, evidence, has_fragment_reconstruction = _cross_segment_merge_score(
            left,
            right,
            gap_seconds,
            analyzer,
        )
        allowed_gap = (
            (
                max_fragment_gap_seconds
                if max_fragment_gap_seconds is not None
                else config.max_cross_segment_fragment_gap_seconds
            )
            if has_fragment_reconstruction
            else config.max_cross_segment_grammar_gap_seconds
        )
        if (
            gap_seconds < 0
            or gap_seconds > allowed_gap
            or score < config.cross_segment_merge_score_threshold
        ):
            merged.append(segment)
            continue

        prefix_count = _minimal_dependent_prefix_length(left, right, analyzer)
        merge_right = right
        remainder: Sentence | None = None
        if prefix_count and prefix_count < len(right.words):
            merge_right = _sentence_from_words(
                right.words[:prefix_count],
                asr_boundary_word_indexes=tuple(
                    boundary
                    for boundary in right.asr_boundary_word_indexes
                    if boundary < prefix_count
                ),
            )
            remainder = _sentence_from_words(
                right.words[prefix_count:],
                asr_boundary_word_indexes=tuple(
                    boundary - prefix_count
                    for boundary in right.asr_boundary_word_indexes
                    if boundary > prefix_count
                ),
            )

        combined = _sentence_from_words(
            (*left.words, *merge_right.words),
            asr_boundary_word_indexes=(
                *left.asr_boundary_word_indexes,
                len(left.words),
                *(
                    len(left.words) + boundary
                    for boundary in merge_right.asr_boundary_word_indexes
                ),
            ),
        )
        sentences = (
            *previous.sentences[:-1],
            combined,
            *((remainder,) if remainder is not None else ()),
            *segment.sentences[1:],
        )
        merged[-1] = Segment(
            position=previous.position,
            text="".join(sentence.text for sentence in sentences),
            time_range=TimeRange(
                previous.time_range.start_seconds,
                max(previous.time_range.end_seconds, segment.time_range.end_seconds),
            ),
            sentences=sentences,
        )
        decisions.append(
            CrossSegmentMergeDecision(
                left_segment_position=previous.position,
                right_segment_position=segment.position,
                word_index=max(len(left.words) - 1, 0),
                left_end_seconds=left.time_range.end_seconds,
                right_start_seconds=right.time_range.start_seconds,
                gap_seconds=gap_seconds,
                score=score,
                reason=(
                    "cross_asr_word_fragment_reconstruction"
                    if has_fragment_reconstruction
                    else "cross_asr_syntactic_continuation"
                ),
                left_text=left.text,
                right_text=merge_right.text,
                evidence=evidence,
            )
        )

    return (
        tuple(
            Segment(
                position=position,
                text=segment.text,
                time_range=segment.time_range,
                sentences=segment.sentences,
            )
            for position, segment in enumerate(merged)
        ),
        tuple(decisions),
    )


def _deduplicate_cross_segment_merges(
    decisions: tuple[CrossSegmentMergeDecision, ...],
) -> tuple[CrossSegmentMergeDecision, ...]:
    unique: list[CrossSegmentMergeDecision] = []
    seen: set[tuple[str, str, str, float]] = set()
    for decision in decisions:
        key = (
            decision.left_text,
            decision.right_text,
            decision.reason,
            round(decision.gap_seconds, 6),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(decision)
    return tuple(unique)


def _merge_high_confidence_adjacent_sentences(
    segments: tuple[Segment, ...],
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> tuple[tuple[Segment, ...], tuple[CrossSegmentMergeDecision, ...]]:
    """Repair false language boundaries already materialized inside a segment."""
    if analyzer is None:
        return segments, ()
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    adjusted: list[Segment] = []
    decisions: list[CrossSegmentMergeDecision] = []
    for segment in segments:
        sentences: list[Sentence] = []
        for sentence in segment.sentences:
            if not sentences:
                sentences.append(sentence)
                continue
            left = sentences[-1]
            right_analysis = analyzer.analyze(_compact_text(sentence.text))
            gap_seconds = sentence.time_range.start_seconds - left.time_range.end_seconds
            score, evidence, has_fragment = _cross_segment_merge_score(
                left,
                sentence,
                gap_seconds,
                analyzer,
            )
            allowed_gap = (
                config.max_cross_segment_fragment_gap_seconds
                if has_fragment
                else config.max_cross_segment_grammar_gap_seconds
            )
            if (
                not left.words
                or not sentence.words
                or not right_analysis
                or gap_seconds < 0
                or gap_seconds > allowed_gap
                or score < config.cross_segment_merge_score_threshold
                or _ends_with_terminal_mark(
                    left.text,
                    (*config.terminal_marks, "?", "!"),
                )
                or _starts_independent_discourse(sentence.text)
                or _starts_topic_shift_expression(right_analysis)
                or (
                    _starts_cross_segment_response(sentence.text, right_analysis)
                    and not has_fragment
                )
                or _has_cross_segment_speaker_turn_veto(
                    left.text,
                    sentence.text,
                    analyzer.analyze(_compact_text(left.text)),
                    right_analysis,
                    gap_seconds,
                )
            ):
                sentences.append(sentence)
                continue

            sentences[-1] = _sentence_from_words(
                (*left.words, *sentence.words),
                asr_boundary_word_indexes=(
                    *left.asr_boundary_word_indexes,
                    len(left.words),
                    *(
                        len(left.words) + boundary
                        for boundary in sentence.asr_boundary_word_indexes
                    ),
                ),
            )
            decisions.append(
                CrossSegmentMergeDecision(
                    left_segment_position=segment.position,
                    right_segment_position=segment.position,
                    word_index=max(len(left.words) - 1, 0),
                    left_end_seconds=left.time_range.end_seconds,
                    right_start_seconds=sentence.time_range.start_seconds,
                    gap_seconds=gap_seconds,
                    score=score,
                    reason=(
                        "cross_asr_word_fragment_reconstruction"
                        if has_fragment
                        else "cross_asr_syntactic_continuation"
                    ),
                    left_text=left.text,
                    right_text=sentence.text,
                    evidence=evidence,
                )
            )
        adjusted.append(
            Segment(
                position=segment.position,
                text="".join(sentence.text for sentence in sentences),
                time_range=segment.time_range,
                sentences=tuple(sentences),
            )
        )
    return tuple(adjusted), tuple(decisions)


def _cross_segment_merge_score(
    left: Sentence,
    right: Sentence,
    gap_seconds: float,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> tuple[int, tuple[CrossSegmentMergeEvidence, ...], bool]:
    evidence: list[CrossSegmentMergeEvidence] = []
    left_text = _compact_text(left.text)
    right_text = _compact_text(right.text)
    left_analysis = analyzer.analyze(left_text)
    right_analysis = analyzer.analyze(right_text)
    if not left_analysis or not right_analysis:
        return 0, (), False

    if _ends_with_terminal_mark(
        left.text,
        (*DEFAULT_SENTENCE_BOUNDARY_CONFIG.terminal_marks, "?", "!"),
    ):
        evidence.append(CrossSegmentMergeEvidence("terminal_punctuation", -6))

    has_fragment_reconstruction = _reconstructs_cross_segment_fragment(
        left_text,
        right_text,
        left_analysis,
        analyzer,
    )
    if has_fragment_reconstruction:
        evidence.append(
            CrossSegmentMergeEvidence("word_fragment_reconstruction", 6)
        )

    left_last = left_analysis[-1]
    if _is_strongly_incomplete_predicate(left_last):
        evidence.append(
            CrossSegmentMergeEvidence("incomplete_predicate_chain", 4)
        )
    elif _ends_with_conditional_clause(left_last):
        evidence.append(CrossSegmentMergeEvidence("conditional_clause_tail", 4))
    elif _ends_with_conjunctive_particle_chain(left_analysis):
        evidence.append(CrossSegmentMergeEvidence("conjunctive_clause_tail", 4))
    elif _ends_with_causal_connective_clause(left_analysis):
        evidence.append(CrossSegmentMergeEvidence("causal_clause_tail", 4))
    elif _ends_with_suspended_object(
        left_last,
        right_analysis,
        gap_seconds,
    ):
        evidence.append(CrossSegmentMergeEvidence("suspended_object", 4))
    elif _ends_with_quotative_topic(
        left_last,
        right_analysis,
        gap_seconds,
    ):
        evidence.append(CrossSegmentMergeEvidence("quotative_topic", 4))
    elif _ends_with_topic_location_tail(left_analysis):
        evidence.append(CrossSegmentMergeEvidence("incomplete_topic_location", 4))
    elif _ends_with_suspended_topic(left_analysis, right_analysis, gap_seconds):
        evidence.append(CrossSegmentMergeEvidence("suspended_topic_predicate", 4))
    elif _ends_with_suspended_subject(left_last, right_analysis, gap_seconds):
        evidence.append(CrossSegmentMergeEvidence("tight_subject_predicate", 4))
    elif _is_dependent_formal_noun_tail(left_analysis):
        evidence.append(CrossSegmentMergeEvidence("dependent_formal_noun", 4))

    if (
        _starts_independent_response(right_analysis)
        and not has_fragment_reconstruction
    ):
        evidence.append(CrossSegmentMergeEvidence("independent_response", -5))

    if _starts_dependent_quotative_continuation(right_text):
        evidence.append(CrossSegmentMergeEvidence("dependent_quotative_right", 4))
    elif _forms_coordinated_condition(left_analysis, right_analysis, right_text):
        evidence.append(CrossSegmentMergeEvidence("coordinated_condition", 4))

    if gap_seconds > DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_cross_segment_grammar_gap_seconds:
        evidence.append(CrossSegmentMergeEvidence("long_cross_segment_gap", -2))
    elif gap_seconds <= DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_dependent_continuation_gap_seconds:
        evidence.append(CrossSegmentMergeEvidence("tight_timing", 1))

    return sum(item.score for item in evidence), tuple(evidence), has_fragment_reconstruction


def _reconstructs_cross_segment_fragment(
    left_text: str,
    right_text: str,
    left_analysis: tuple[JapaneseMorpheme, ...],
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    combined = analyzer.analyze(f"{left_text}{right_text}")
    if not combined:
        return False

    cursor = 0
    boundary_morpheme: JapaneseMorpheme | None = None
    following_morpheme: JapaneseMorpheme | None = None
    has_boundary = False
    for index, morpheme in enumerate(combined):
        cursor += len(morpheme.surface)
        if cursor == len(left_text):
            has_boundary = True
            boundary_morpheme = morpheme
            if index + 1 < len(combined):
                following_morpheme = combined[index + 1]
            break
        if cursor > len(left_text):
            boundary_morpheme = morpheme
            break
    if boundary_morpheme is None:
        return False
    separate_last = left_analysis[-1]
    if not has_boundary:
        return bool(
            separate_last.part_of_speech
            and separate_last.part_of_speech[0] in {"感動詞", "接頭辞"}
            and len(separate_last.surface) <= 3
            and (
                left_text.endswith(("ー", "っ", "ゃ", "ゅ", "ょ"))
                or (
                    analyzer.analyze(right_text)
                    and _is_functional_continuation(analyzer.analyze(right_text)[0])
                )
            )
        )

    if (
        _is_strongly_incomplete_predicate(separate_last)
        and following_morpheme is not None
        and _is_functional_continuation(following_morpheme)
    ):
        return True

    separate_right = analyzer.analyze(right_text)
    if (
        has_boundary
        and following_morpheme is not None
        and separate_right
        and separate_right[0].part_of_speech
        and separate_right[0].part_of_speech[0] == "感動詞"
        and len(separate_right[0].surface) <= 2
        and following_morpheme.surface == separate_right[0].surface
        and _is_functional_continuation(following_morpheme)
        and following_morpheme.part_of_speech[:2]
        != separate_right[0].part_of_speech[:2]
    ):
        return True

    return bool(
        boundary_morpheme.surface == separate_last.surface
        and separate_last.part_of_speech
        and separate_last.part_of_speech[0] == "感動詞"
        and len(separate_last.surface) <= 3
        and (
            boundary_morpheme.dictionary_form != separate_last.dictionary_form
            or boundary_morpheme.part_of_speech[:2]
            != separate_last.part_of_speech[:2]
            or boundary_morpheme.conjugation_form
            != separate_last.conjugation_form
        )
    )


def _starts_cross_segment_response(
    text: str,
    analysis: tuple[JapaneseMorpheme, ...],
) -> bool:
    normalized = _compact_text(text)
    return bool(
        _starts_independent_response(analysis)
        or _is_compact_terminal_response(analysis, normalized)
        or normalized.startswith((*_RESPONSE_STARTS, "わかりました", "どういうこと"))
    )


def _is_compact_terminal_response(
    analysis: tuple[JapaneseMorpheme, ...],
    normalized_text: str,
) -> bool:
    if not analysis or len(normalized_text) > 8:
        return False
    first = analysis[0]
    last = analysis[-1]
    if (
        not first.part_of_speech
        or first.part_of_speech[0] not in {"副詞", "感動詞"}
        or last.part_of_speech[:2] != ("助詞", "終助詞")
    ):
        return False
    return bool(
        len(analysis) == 2
        or (
            len(analysis) == 3
            and analysis[1].part_of_speech
            and analysis[1].part_of_speech[0] == "助動詞"
            and analysis[1].conjugation_type in {"助動詞-ダ", "助動詞-デス"}
        )
    )


def _starts_topic_shift_expression(
    analysis: tuple[JapaneseMorpheme, ...],
) -> bool:
    if not (
        len(analysis) >= 2
        and analysis[0].part_of_speech
        and analysis[0].part_of_speech[0] == "代名詞"
        and analysis[1].part_of_speech
        and analysis[1].part_of_speech[0] == "助詞"
        and analysis[1].dictionary_form in {"より", "では"}
    ):
        return False
    if analysis[1].dictionary_form == "では":
        return True
    return bool(
        len(analysis) >= 3
        and analysis[2].part_of_speech
        and analysis[2].part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
    )


def _is_strongly_incomplete_predicate(morpheme: JapaneseMorpheme) -> bool:
    return bool(
        morpheme.part_of_speech
        and morpheme.part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
        and any(
            form in morpheme.conjugation_form
            for form in ("未然形", "連用形", "接続形")
        )
    )


def _ends_with_conditional_clause(morpheme: JapaneseMorpheme) -> bool:
    return bool(
        morpheme.part_of_speech
        and morpheme.part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
        and "仮定形" in morpheme.conjugation_form
    )


def _ends_with_conjunctive_particle(morpheme: JapaneseMorpheme) -> bool:
    return bool(
        morpheme.part_of_speech[:2] == ("助詞", "接続助詞")
        and morpheme.dictionary_form in {"て", "で"}
    )


def _ends_with_suspended_object(
    morpheme: JapaneseMorpheme,
    right: tuple[JapaneseMorpheme, ...],
    gap_seconds: float,
) -> bool:
    return bool(
        gap_seconds
        <= DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_cross_segment_grammar_gap_seconds
        and morpheme.part_of_speech[:2] == ("助詞", "格助詞")
        and morpheme.dictionary_form == "を"
        and _has_early_predicate(right)
    )


def _ends_with_quotative_topic(
    morpheme: JapaneseMorpheme,
    right: tuple[JapaneseMorpheme, ...],
    gap_seconds: float,
) -> bool:
    return bool(
        gap_seconds
        <= DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_cross_segment_grammar_gap_seconds
        and morpheme.part_of_speech
        and morpheme.part_of_speech[0] == "助詞"
        and morpheme.dictionary_form == "って"
        and _has_early_predicate(right)
    )


def _has_early_predicate(morphemes: tuple[JapaneseMorpheme, ...]) -> bool:
    return any(
        morpheme.part_of_speech
        and morpheme.part_of_speech[0]
        in {"動詞", "形容詞", "形状詞", "助動詞"}
        for morpheme in morphemes[:4]
    )


def _starts_dependent_quotative_continuation(text: str) -> bool:
    return _compact_text(text).startswith(("っていう", "という"))


def _forms_coordinated_condition(
    left: tuple[JapaneseMorpheme, ...],
    right: tuple[JapaneseMorpheme, ...],
    right_text: str,
) -> bool:
    if not left or not right or not _compact_text(right_text).startswith("そして"):
        return False
    left_last = left[-1]
    return bool(
        left_last.dictionary_form == "たい"
        and left_last.part_of_speech
        and left_last.part_of_speech[0] == "助動詞"
        and _ends_with_conditional_clause(right[-1])
    )


def _is_dependent_formal_noun_tail(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    if len(morphemes) < 2 or morphemes[-1].dictionary_form != "ほか":
        return False
    previous = morphemes[-2]
    return bool(
        previous.part_of_speech
        and previous.part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
        and "連体形" in previous.conjugation_form
    )


def _ends_with_topic_location_tail(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    return bool(
        len(morphemes) >= 2
        and morphemes[-2].part_of_speech[:2] == ("助詞", "格助詞")
        and morphemes[-2].dictionary_form == "で"
        and morphemes[-1].part_of_speech[:2] == ("助詞", "係助詞")
        and morphemes[-1].dictionary_form == "は"
    )


def _ends_with_suspended_topic(
    left: tuple[JapaneseMorpheme, ...],
    right: tuple[JapaneseMorpheme, ...],
    gap_seconds: float,
) -> bool:
    return bool(
        gap_seconds
        <= DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_cross_segment_grammar_gap_seconds
        and len(left) >= 2
        and left[-1].part_of_speech[:2] == ("助詞", "係助詞")
        and left[-1].dictionary_form == "は"
        and left[-2].part_of_speech
        and left[-2].part_of_speech[0] in {"名詞", "代名詞"}
        and _has_early_predicate(right)
    )


def _ends_with_suspended_subject(
    left_last: JapaneseMorpheme,
    right: tuple[JapaneseMorpheme, ...],
    gap_seconds: float,
) -> bool:
    return bool(
        gap_seconds
        <= DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_cross_segment_grammar_gap_seconds
        and left_last.part_of_speech[:2] == ("助詞", "格助詞")
        and left_last.dictionary_form == "が"
        and _has_early_predicate(right)
    )


def _merge_adjacent_dependent_continuations(
    segments: tuple[Segment, ...],
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> tuple[Segment, ...]:
    """Join cross-segment continuations supported by timing and grammar."""
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    merged: list[Segment] = []
    for segment in segments:
        if not merged or not segment.sentences:
            merged.append(segment)
            continue

        previous = merged[-1]
        if not previous.sentences:
            merged.append(segment)
            continue

        left = previous.sentences[-1]
        right = segment.sentences[0]
        left_analysis = (
            analyzer.analyze(_compact_text(left.text))
            if analyzer is not None
            else ()
        )
        right_analysis = (
            analyzer.analyze(_compact_text(right.text))
            if analyzer is not None
            else ()
        )
        gap_seconds = (
            right.time_range.start_seconds - left.time_range.end_seconds
        )
        expected_number = _expected_numbering_restart(left, analyzer)
        preserves_numbering_restart = bool(
            analyzer is not None
            and expected_number is not None
            and _embedded_expected_number_index(
                right.words,
                expected_number,
                analyzer,
            )
            is not None
        )
        if (
            not left.words
            or not right.words
            or gap_seconds < 0
            or gap_seconds > config.max_continuation_candidate_gap_seconds
            or preserves_numbering_restart
            or _ends_with_terminal_mark(
                left.text,
                (*config.terminal_marks, "?", "!"),
            )
            or (
                analyzer is not None
                and _starts_cross_segment_response(right.text, right_analysis)
            )
            or (
                analyzer is not None
                and _has_cross_segment_speaker_turn_veto(
                    left.text,
                    right.text,
                    left_analysis,
                    right_analysis,
                    gap_seconds,
                )
            )
            or (
                analyzer is not None
                and _starts_topic_shift_expression(right_analysis)
            )
            or (
                analyzer is not None
                and _is_short_response(left.text, left_analysis)
                and _starts_independent_clause(right_analysis)
            )
            or _continuation_evidence_score(
                left,
                right,
                gap_seconds,
                analyzer,
            )
            < config.continuation_score_threshold
        ):
            merged.append(segment)
            continue

        prefix_count = _minimal_dependent_prefix_length(left, right, analyzer)
        prefix_words = right.words[:prefix_count]
        if prefix_count and prefix_count < len(right.words):
            attached = _sentence_from_words(
                (*left.words, *prefix_words),
                asr_boundary_word_indexes=(
                    *left.asr_boundary_word_indexes,
                    len(left.words),
                    *(
                        len(left.words) + boundary
                        for boundary in right.asr_boundary_word_indexes
                        if boundary < prefix_count
                    ),
                ),
            )
            previous_sentences = (*previous.sentences[:-1], attached)
            merged[-1] = Segment(
                position=previous.position,
                text="".join(sentence.text for sentence in previous_sentences),
                time_range=TimeRange(
                    previous.time_range.start_seconds,
                    attached.time_range.end_seconds,
                ),
                sentences=previous_sentences,
            )
            remainder = _sentence_from_words(
                right.words[prefix_count:],
                asr_boundary_word_indexes=tuple(
                    boundary - prefix_count
                    for boundary in right.asr_boundary_word_indexes
                    if boundary > prefix_count
                ),
            )
            remaining_sentences = (remainder, *segment.sentences[1:])
            merged.append(
                Segment(
                    position=segment.position,
                    text="".join(sentence.text for sentence in remaining_sentences),
                    time_range=TimeRange(
                        remainder.time_range.start_seconds,
                        segment.time_range.end_seconds,
                    ),
                    sentences=remaining_sentences,
                )
            )
            continue

        combined = Sentence(
            text=f"{left.text}{right.text}",
            time_range=TimeRange(
                left.time_range.start_seconds,
                right.time_range.end_seconds,
            ),
            words=(*left.words, *right.words),
            asr_boundary_word_indexes=(
                *left.asr_boundary_word_indexes,
                len(left.words),
                *(
                    len(left.words) + boundary
                    for boundary in right.asr_boundary_word_indexes
                ),
            ),
        )
        sentences = (
            *previous.sentences[:-1],
            combined,
            *segment.sentences[1:],
        )
        merged[-1] = Segment(
            position=previous.position,
            text="".join(sentence.text for sentence in sentences),
            time_range=TimeRange(
                previous.time_range.start_seconds,
                segment.time_range.end_seconds,
            ),
            sentences=sentences,
        )

    return tuple(
        Segment(
            position=position,
            text=segment.text,
            time_range=segment.time_range,
            sentences=segment.sentences,
        )
        for position, segment in enumerate(merged)
    )


def _expected_numbering_restart(
    sentence: Sentence,
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> int | None:
    if analyzer is None:
        return None
    starts = _structured_numbering_start_indexes(
        sentence.words,
        analyzer,
        _asr_boundary_word_indexes(sentence),
    )
    if not starts:
        return None
    last_index = max(starts)
    candidate = _numbering_candidate(
        sentence.words,
        last_index,
        sentence.words[last_index],
        analyzer,
        _asr_boundary_word_indexes(sentence),
    )
    return candidate.value + 1 if candidate is not None else None


def _minimal_dependent_prefix_length(
    left: Sentence,
    right: Sentence,
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> int:
    """Return the shortest prefix that completes the left without consuming a new clause."""
    if analyzer is None or len(right.words) < 2:
        return len(right.words)
    for count in range(1, len(right.words)):
        combined_text = _compact_text(
            f"{left.text}{_words_text(right.words[:count])}"
        )
        combined = analyzer.analyze(combined_text)
        remainder_text = _words_text(right.words[count:])
        remainder = analyzer.analyze(_compact_text(remainder_text))
        starts_independent_response = _starts_cross_segment_response(
            remainder_text,
            remainder,
        )
        if (
            combined
            and remainder
            and _is_complete_clause(combined)
            and (
                starts_independent_response
                or _starts_independent_clause(remainder)
            )
            and (
                starts_independent_response
                or not _has_cross_boundary_morphological_dependency(
                    combined_text,
                    remainder_text,
                    analyzer,
                )
            )
            and _is_valid_prefix_partition(
                combined,
                remainder,
                has_morpheme_boundary=_has_morpheme_boundary_at(
                    f"{combined_text}{_compact_text(_words_text(right.words[count:]))}",
                    len(combined_text),
                    analyzer,
                ),
            )
            and not _next_single_character_extends_inflection(
                combined_text,
                right.words[count],
                analyzer,
            )
        ):
            return count
    return len(right.words)


def _is_valid_prefix_partition(
    attached: tuple[JapaneseMorpheme, ...],
    remainder: tuple[JapaneseMorpheme, ...],
    *,
    has_morpheme_boundary: bool = True,
) -> bool:
    """Require both sides of a migrated prefix to remain grammatical units."""
    if not attached or not remainder or not has_morpheme_boundary:
        return False
    first = remainder[0]
    if _starts_with_functional_fragment(first):
        return False
    if _starts_with_unfinished_conjunction(remainder):
        return False
    if _crosses_adnominal_dependency(attached[-1], first):
        return False
    return bool(
        _is_complete_clause(remainder)
        or _is_independent_response_start(first)
        or _starts_structural_restart(remainder)
        or _starts_independent_clause(remainder)
    )


def _has_morpheme_boundary_at(
    text: str,
    offset: int,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    cursor = 0
    for morpheme in analyzer.analyze(text):
        cursor += len(morpheme.surface)
        if cursor == offset:
            return True
        if cursor > offset:
            return False
    return False


def _starts_with_functional_fragment(morpheme: JapaneseMorpheme) -> bool:
    return bool(
        morpheme.part_of_speech
        and morpheme.part_of_speech[0] in {"助詞", "助動詞", "接尾辞"}
    )


def _starts_with_unfinished_conjunction(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    first = morphemes[0]
    return bool(
        first.part_of_speech
        and first.part_of_speech[0] == "接続詞"
        and not _is_complete_clause(morphemes)
    )


def _crosses_adnominal_dependency(
    left: JapaneseMorpheme,
    right: JapaneseMorpheme,
) -> bool:
    return bool(
        left.part_of_speech
        and right.part_of_speech
        and left.part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
        and "連体形" in left.conjugation_form
        and right.part_of_speech[0] in {"名詞", "代名詞"}
    )


def _next_single_character_extends_inflection(
    prefix_text: str,
    next_word: Word,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    character = _word_text(next_word)
    if len(character) != 1 or unicodedata.category(character).startswith("P"):
        return False
    before = analyzer.analyze(prefix_text)
    after = analyzer.analyze(f"{prefix_text}{character}")
    if not before or not after:
        return False
    previous_last = before[-1]
    extended_last = after[-1]
    return bool(
        previous_last.part_of_speech
        and extended_last.part_of_speech
        and previous_last.part_of_speech[0] == extended_last.part_of_speech[0]
        and previous_last.dictionary_form
        and previous_last.dictionary_form == extended_last.dictionary_form
        and previous_last.conjugation_form
        and previous_last.conjugation_form == extended_last.conjugation_form
        and extended_last.surface == f"{previous_last.surface}{character}"
    )


def _is_complete_clause(morphemes: tuple[JapaneseMorpheme, ...]) -> bool:
    last = morphemes[-1]
    if not last.part_of_speech:
        return False
    major = last.part_of_speech[0]
    minor = last.part_of_speech[1] if len(last.part_of_speech) > 1 else ""
    if major == "補助記号":
        return minor == "句点"
    if major == "助詞" and minor == "終助詞":
        return True
    return major in {"動詞", "形容詞", "助動詞"} and (
        "終止形" in last.conjugation_form or "意志推量形" in last.conjugation_form
    )


def _starts_independent_clause(morphemes: tuple[JapaneseMorpheme, ...]) -> bool:
    return bool(morphemes) and _is_strong_independent_start(morphemes[0])


def _is_strong_independent_start(morpheme: JapaneseMorpheme) -> bool:
    return bool(morpheme.part_of_speech) and morpheme.part_of_speech[0] in {
        "感動詞", "接続詞", "副詞", "代名詞", "名詞", "連体詞", "接頭辞",
        "形状詞",
    }


def _ends_in_connective_form(morphemes: tuple[JapaneseMorpheme, ...]) -> bool:
    last = morphemes[-1]
    return bool(
        last.part_of_speech
        and last.part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
        and any(
            form in last.conjugation_form
            for form in ("連用形", "接続形")
        )
    )


def _starts_independent_response(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    first = morphemes[0]
    return bool(
        first.part_of_speech
        and (
            first.part_of_speech[0] == "感動詞"
            or (
                first.part_of_speech[0] == "名詞"
                and len(first.part_of_speech) > 1
                and first.part_of_speech[1] == "感動詞的"
            )
        )
    )


def _starts_structural_restart(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    if not morphemes:
        return False
    first = morphemes[0]
    if _is_numeric_morpheme(first):
        return len(morphemes) == 1 or _is_enumeration_delimiter(morphemes[1])
    return bool(
        len(morphemes) >= 2
        and _is_restart_label(first)
        and _is_numeric_morpheme(morphemes[1])
    )


def _is_restart_label(morpheme: JapaneseMorpheme) -> bool:
    return bool(
        morpheme.part_of_speech
        and morpheme.part_of_speech[0] in {"名詞", "接頭辞"}
        and not _is_numeric_morpheme(morpheme)
    )


def _is_numeric_morpheme(morpheme: JapaneseMorpheme) -> bool:
    return bool(
        morpheme.surface.isdecimal()
        or (
            len(morpheme.part_of_speech) > 1
            and morpheme.part_of_speech[0] == "名詞"
            and morpheme.part_of_speech[1] == "数詞"
        )
    )


def _is_enumeration_delimiter(morpheme: JapaneseMorpheme) -> bool:
    return bool(
        morpheme.part_of_speech
        and morpheme.part_of_speech[0] == "補助記号"
        and len(morpheme.part_of_speech) > 1
        and morpheme.part_of_speech[1] in {"読点", "括弧開", "括弧閉"}
    )


def _is_question_clause(morphemes: tuple[JapaneseMorpheme, ...]) -> bool:
    last = morphemes[-1]
    return bool(
        last.part_of_speech
        and last.part_of_speech[0] == "助詞"
        and len(last.part_of_speech) > 1
        and last.part_of_speech[1] == "終助詞"
    )


def _is_interrogative_clause(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    return _is_question_clause(morphemes) and morphemes[-1].surface in {"か", "の"}


def _continuation_evidence_score(
    left: Sentence,
    right: Sentence,
    gap_seconds: float,
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> int:
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    score = 0
    right_text = _compact_text(right.text)
    if gap_seconds <= config.max_dependent_continuation_gap_seconds:
        score += config.close_timing_evidence_score
    if right_text.startswith(config.dependent_continuation_prefixes):
        score += config.dependent_prefix_evidence_score
    if analyzer is None:
        return score

    left_analysis = analyzer.analyze(_compact_text(left.text))
    right_analysis = analyzer.analyze(_compact_text(right.text))
    if not left_analysis or not right_analysis:
        return score
    left_last = left_analysis[-1]
    right_first = right_analysis[0]
    contextual_right = _contextual_right_morpheme(
        left.text,
        right.text,
        analyzer,
    )
    dependency_first = contextual_right or right_first
    if _is_functional_continuation(dependency_first):
        score += config.functional_continuation_evidence_score
    if _forms_adverb_predicate_continuation(left_last, dependency_first):
        score += config.dependent_prefix_evidence_score
    if _forms_adverb_clause_continuation(left_last, right_analysis):
        score += config.functional_continuation_evidence_score
    if _forms_adverbial_nonfinite_dependency(left_analysis, right_analysis):
        score += config.functional_continuation_evidence_score
    if _forms_contextual_adnominal_dependency(left.text, right.text, analyzer):
        score += config.dependent_prefix_evidence_score
    if (
        contextual_right is not None
        and _is_functional_continuation(contextual_right)
        and not _is_functional_continuation(right_first)
    ):
        score += config.functional_continuation_evidence_score
    if _is_incomplete_left_context(left_last) or _forms_auxiliary_chain(
        left_last,
        dependency_first,
    ):
        score += config.incomplete_left_evidence_score
    elif _has_clause_completing_prefix(left, right, analyzer):
        score += config.incomplete_left_evidence_score
    return score


def _has_clause_completing_prefix(
    left: Sentence,
    right: Sentence,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    for count in range(1, len(right.words) + 1):
        combined = analyzer.analyze(
            _compact_text(f"{left.text}{_words_text(right.words[:count])}")
        )
        if combined and _is_complete_clause(combined):
            return True
    return False


def _is_functional_continuation(morpheme: JapaneseMorpheme) -> bool:
    return bool(morpheme.part_of_speech) and morpheme.part_of_speech[0] in {
        "助詞",
        "助動詞",
        "接尾辞",
    }


def _is_incomplete_left_context(morpheme: JapaneseMorpheme) -> bool:
    if not morpheme.part_of_speech:
        return False
    major = morpheme.part_of_speech[0]
    if major in {
        "名詞",
        "代名詞",
        "副詞",
        "連体詞",
        "接頭辞",
    }:
        return True
    return major in {"動詞", "形容詞", "助動詞"} and any(
        form in morpheme.conjugation_form
        for form in ("未然形", "連用形", "連体形")
    )


def _forms_auxiliary_chain(
    left: JapaneseMorpheme,
    right: JapaneseMorpheme,
) -> bool:
    return bool(
        left.part_of_speech
        and right.part_of_speech
        and left.part_of_speech[0] == "助動詞"
        and right.part_of_speech[0] == "助動詞"
    )


def _forms_adverb_predicate_continuation(
    left: JapaneseMorpheme,
    right: JapaneseMorpheme,
) -> bool:
    return bool(
        left.part_of_speech
        and right.part_of_speech
        and left.part_of_speech[0] == "副詞"
        and right.part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
    )


def _forms_adverb_clause_continuation(
    left: JapaneseMorpheme,
    right: tuple[JapaneseMorpheme, ...],
) -> bool:
    return bool(
        left.part_of_speech
        and left.part_of_speech[0] == "副詞"
        and right
        and _starts_independent_clause(right)
        and any(
            item.part_of_speech
            and item.part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
            for item in right
        )
    )


def _has_cross_boundary_morphological_dependency(
    left_text: str,
    right_text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    if _forms_contextual_adnominal_dependency(left_text, right_text, analyzer):
        return True
    left = analyzer.analyze(_compact_text(left_text))
    right = analyzer.analyze(_compact_text(right_text))
    contextual_right = _contextual_right_morpheme(
        left_text,
        right_text,
        analyzer,
    )
    if not left or not right or contextual_right is None:
        return False
    contextual_dependency = bool(
        _is_functional_continuation(contextual_right)
        and not _is_functional_continuation(right[0])
        and left[-1].part_of_speech
        and left[-1].part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
    )
    return contextual_dependency or _forms_adverbial_nonfinite_dependency(
        left,
        right,
    )


def _has_conjunctive_predicate_continuation(
    left_text: str,
    right_text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    left = analyzer.analyze(_compact_text(left_text))
    right = analyzer.analyze(_compact_text(right_text))
    return bool(
        left
        and right
        and _ends_with_conjunctive_particle_chain(left)
        and right[0].part_of_speech
        and right[0].part_of_speech[0]
        in {"動詞", "形容詞", "助動詞"}
    )


def _ends_with_conjunctive_particle_chain(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    if not morphemes:
        return False
    if _ends_with_conjunctive_particle(morphemes[-1]):
        return True
    return bool(
        len(morphemes) >= 2
        and morphemes[-1].part_of_speech[:2] == ("助詞", "係助詞")
        and morphemes[-1].dictionary_form == "も"
        and _ends_with_conjunctive_particle(morphemes[-2])
    )


def _ends_with_causal_connective_clause(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    if not morphemes:
        return False
    last = morphemes[-1]
    return bool(
        last.part_of_speech[:2] == ("助詞", "接続助詞")
        and last.dictionary_form in {"から", "ので"}
    )


def _forms_adverbial_nonfinite_dependency(
    left: tuple[JapaneseMorpheme, ...],
    right: tuple[JapaneseMorpheme, ...],
) -> bool:
    if not left or not right:
        return False
    left_last = left[-1]
    right_first = right[0]
    if not right_first.part_of_speech or right_first.part_of_speech[0] not in {
        "動詞",
        "形容詞",
        "助動詞",
    }:
        return False
    if not any(
        form in right_first.conjugation_form
        for form in ("未然形", "連用形", "接続形")
    ):
        return False
    if not left_last.part_of_speech:
        return False
    if left_last.part_of_speech[0] == "副詞":
        return True
    if left_last.conjugation_type in {"助動詞-ダ", "助動詞-デス"}:
        return bool(
            len(left) > 1
            and left[-2].part_of_speech
            and left[-2].part_of_speech[0] == "形状詞"
            and any(
                form in left_last.conjugation_form
                for form in ("連用形", "接続形")
            )
        )
    return bool(
        left_last.part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
        and any(
            form in left_last.conjugation_form
            for form in ("連用形", "接続形")
        )
    )


def _merge_adjacent_connective_continuations(
    segments: tuple[Segment, ...],
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> tuple[Segment, ...]:
    """Join a connective te-form segment to its following main clause."""
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    merged: list[Segment] = []
    for segment in segments:
        if not merged or not segment.sentences:
            merged.append(segment)
            continue

        previous = merged[-1]
        if not previous.sentences:
            merged.append(segment)
            continue

        left = previous.sentences[-1]
        right = segment.sentences[0]
        left_analysis = (
            analyzer.analyze(_compact_text(left.text))
            if analyzer is not None
            else ()
        )
        right_analysis = (
            analyzer.analyze(_compact_text(right.text))
            if analyzer is not None
            else ()
        )
        gap_seconds = right.time_range.start_seconds - left.time_range.end_seconds
        if (
            not left.words
            or not right.words
            or gap_seconds < 0
            or gap_seconds > config.max_connective_continuation_gap_seconds
            or _starts_independent_discourse(right.text)
            or (
                analyzer is not None
                and _starts_cross_segment_response(right.text, right_analysis)
            )
            or (
                analyzer is not None
                and _has_cross_segment_speaker_turn_veto(
                    left.text,
                    right.text,
                    left_analysis,
                    right_analysis,
                    gap_seconds,
                )
            )
            or (
                analyzer is not None
                and _starts_topic_shift_expression(right_analysis)
            )
            or (
                analyzer is not None
                and _is_short_response(left.text, left_analysis)
                and _starts_independent_clause(right_analysis)
            )
            or not _has_connective_tail(left.text, analyzer)
            or not _connective_merge_has_score_advantage(
                left,
                right,
                gap_seconds,
                analyzer,
            )
        ):
            merged.append(segment)
            continue

        joined_left = (
            _append_comma(left)
            if gap_seconds >= config.min_pause_seconds
            else left
        )
        combined = Sentence(
            text=f"{joined_left.text}{right.text}",
            time_range=TimeRange(
                joined_left.time_range.start_seconds,
                right.time_range.end_seconds,
            ),
            words=(*joined_left.words, *right.words),
            asr_boundary_word_indexes=(
                *joined_left.asr_boundary_word_indexes,
                len(joined_left.words),
                *(
                    len(joined_left.words) + boundary
                    for boundary in right.asr_boundary_word_indexes
                ),
            ),
        )
        sentences = (
            *previous.sentences[:-1],
            combined,
            *segment.sentences[1:],
        )
        merged[-1] = Segment(
            position=previous.position,
            text="".join(sentence.text for sentence in sentences),
            time_range=TimeRange(
                previous.time_range.start_seconds,
                segment.time_range.end_seconds,
            ),
            sentences=sentences,
        )

    return tuple(
        Segment(
            position=position,
            text=segment.text,
            time_range=segment.time_range,
            sentences=segment.sentences,
        )
        for position, segment in enumerate(merged)
    )


def _connective_merge_has_score_advantage(
    left: Sentence,
    right: Sentence,
    gap_seconds: float,
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> bool:
    if analyzer is None:
        return True
    left_analysis = analyzer.analyze(_compact_text(left.text))
    right_analysis = analyzer.analyze(_compact_text(right.text))
    if not left_analysis or not right_analysis:
        return False
    continuation_score = _connective_continuation_score(
        left_analysis,
        right_analysis,
        gap_seconds,
    )
    transition_score = _turn_transition_score(
        left_analysis,
        right_analysis,
        gap_seconds,
    )
    return continuation_score >= (
        transition_score
        + DEFAULT_SENTENCE_BOUNDARY_CONFIG.connective_merge_score_margin
    )


def _connective_continuation_score(
    left: tuple[JapaneseMorpheme, ...],
    right: tuple[JapaneseMorpheme, ...],
    gap_seconds: float,
) -> int:
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    score = 0
    left_last = left[-1]
    right_first = right[0]
    if _is_connective_morpheme(left_last) or _is_nonfinal_particle(left_last):
        score += 4
    elif _is_incomplete_left_context(left_last):
        score += 2
    if _is_functional_continuation(right_first) or _is_non_independent_predicate(
        right_first
    ):
        score += 4
    elif (
        not _is_independent_response_start(right_first)
        and not _is_independent_predicate(right_first)
        and not _is_interrogative_clause(right)
    ):
        score += 2
    if gap_seconds <= config.max_dependent_continuation_gap_seconds:
        score += 1
    return score


def _turn_transition_score(
    left: tuple[JapaneseMorpheme, ...],
    right: tuple[JapaneseMorpheme, ...],
    gap_seconds: float,
) -> int:
    config = DEFAULT_SENTENCE_BOUNDARY_CONFIG
    del left
    score = 0
    right_first = right[0]
    if _is_independent_response_start(right_first):
        score += 5
    if _is_complete_clause(right):
        score += 2
    if _is_interrogative_clause(right):
        score += 3
    if _is_independent_predicate(right_first) and _is_complete_clause(right):
        score += 3
    if gap_seconds >= config.min_pause_seconds:
        score += 1
    return score


def _is_connective_morpheme(morpheme: JapaneseMorpheme) -> bool:
    return bool(
        morpheme.part_of_speech
        and (
            (
                morpheme.part_of_speech[0] == "助詞"
                and len(morpheme.part_of_speech) > 1
                and morpheme.part_of_speech[1] == "接続助詞"
            )
            or (
                morpheme.part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
                and any(
                    form in morpheme.conjugation_form
                    for form in ("連用形", "接続形")
                )
            )
        )
    )


def _is_nonfinal_particle(morpheme: JapaneseMorpheme) -> bool:
    return bool(
        len(morpheme.part_of_speech) > 1
        and morpheme.part_of_speech[0] == "助詞"
        and morpheme.part_of_speech[1] != "終助詞"
    )


def _is_non_independent_predicate(morpheme: JapaneseMorpheme) -> bool:
    return bool(
        len(morpheme.part_of_speech) > 1
        and morpheme.part_of_speech[0] in {"動詞", "形容詞"}
        and morpheme.part_of_speech[1] == "非自立可能"
    )


def _is_independent_predicate(morpheme: JapaneseMorpheme) -> bool:
    return bool(
        morpheme.part_of_speech
        and morpheme.part_of_speech[0] in {"動詞", "形容詞"}
        and not _is_non_independent_predicate(morpheme)
    )


def _is_independent_response_start(morpheme: JapaneseMorpheme) -> bool:
    if not morpheme.part_of_speech:
        return False
    major = morpheme.part_of_speech[0]
    if major in {"感動詞", "接続詞", "副詞"}:
        return True
    return bool(
        major == "名詞"
        and len(morpheme.part_of_speech) > 1
        and morpheme.part_of_speech[1] == "感動詞的"
    )


def _ends_with_connective_te(text: str) -> bool:
    normalized = _compact_text(text)
    if not normalized or normalized.endswith(("、", ",")):
        return False
    return normalized.endswith(("て", "で"))


def _has_connective_tail(
    text: str,
    analyzer: JapaneseMorphologicalAnalyzer | None,
) -> bool:
    if _ends_with_connective_te(text):
        return True
    if analyzer is None:
        return False
    analyzed = analyzer.analyze(_compact_text(text))
    return bool(
        analyzed
        and not _is_complete_clause(analyzed)
        and (
            _is_connective_morpheme(analyzed[-1])
            or _is_nonfinal_particle(analyzed[-1])
        )
    )


def _starts_independent_discourse(text: str) -> bool:
    return _compact_text(text).startswith(_INDEPENDENT_DISCOURSE_STARTS)


def _has_cross_segment_speaker_turn_veto(
    left_text: str,
    right_text: str,
    left: tuple[JapaneseMorpheme, ...],
    right: tuple[JapaneseMorpheme, ...],
    gap_seconds: float,
) -> bool:
    if not left or not right:
        return False
    normalized_right = _compact_text(right_text)
    if normalized_right.startswith("すいません"):
        return True
    if _is_compact_acknowledgement(right, normalized_right):
        return True
    if _starts_first_person_agreement(right) and (
        _contains_first_person_subject(left)
        or _is_addressee_request(left)
    ):
        return True
    if _starts_disclosure_restart(right):
        return True
    if (
        gap_seconds >= DEFAULT_SENTENCE_BOUNDARY_CONFIG.min_pause_seconds
        and _is_deictic_elliptical_subject(left)
        and _is_complete_clause(right)
    ):
        return True
    return _ends_with_negative_request(left) and _repeats_addressee(left, right)


def _is_compact_acknowledgement(
    morphemes: tuple[JapaneseMorpheme, ...],
    normalized_text: str,
) -> bool:
    return bool(
        len(normalized_text) <= 8
        and _is_complete_clause(morphemes)
        and any(
            morpheme.part_of_speech
            and morpheme.part_of_speech[0] == "動詞"
            and morpheme.dictionary_form == "分かる"
            for morpheme in morphemes
        )
    )


def _starts_first_person_agreement(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    surface_text = "".join(morpheme.surface for morpheme in morphemes)
    return bool(
        len(morphemes) >= 2
        and len(surface_text) <= 8
        and morphemes[0].part_of_speech
        and morphemes[0].part_of_speech[0] == "代名詞"
        and morphemes[0].dictionary_form in {"私", "僕", "俺"}
        and morphemes[1].part_of_speech[:2] == ("助詞", "係助詞")
        and morphemes[1].dictionary_form == "も"
    )


def _contains_first_person_subject(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    return any(
        morpheme.part_of_speech
        and morpheme.part_of_speech[0] == "代名詞"
        and morpheme.dictionary_form in {"私", "僕", "俺"}
        for morpheme in morphemes
    )


def _is_addressee_request(morphemes: tuple[JapaneseMorpheme, ...]) -> bool:
    return bool(
        _ends_with_conjunctive_particle_chain(morphemes)
        and morphemes[0].part_of_speech
        and morphemes[0].part_of_speech[0] == "名詞"
        and any(
            morpheme.part_of_speech
            and morpheme.part_of_speech[0] == "代名詞"
            and morpheme.dictionary_form in {"何", "どれ", "どこ", "誰"}
            for morpheme in morphemes[1:4]
        )
    )


def _starts_disclosure_restart(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    return bool(
        len(morphemes) >= 2
        and morphemes[0].dictionary_form == "実"
        and morphemes[1].part_of_speech[:2] == ("助詞", "係助詞")
        and morphemes[1].dictionary_form == "は"
    )


def _is_deictic_elliptical_subject(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    return bool(
        len(morphemes) <= 4
        and morphemes[0].part_of_speech
        and morphemes[0].part_of_speech[0] == "連体詞"
        and morphemes[-1].part_of_speech[:2] == ("助詞", "格助詞")
        and morphemes[-1].dictionary_form == "が"
    )


def _ends_with_negative_request(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    return bool(
        len(morphemes) >= 2
        and morphemes[-1].part_of_speech[:2] == ("助詞", "接続助詞")
        and morphemes[-1].dictionary_form == "で"
        and morphemes[-2].dictionary_form == "ない"
    )


def _repeats_addressee(
    left: tuple[JapaneseMorpheme, ...],
    right: tuple[JapaneseMorpheme, ...],
) -> bool:
    left_names = {
        morpheme.dictionary_form
        for morpheme in left
        if morpheme.part_of_speech[:2] == ("名詞", "固有名詞")
    }
    return bool(
        left_names
        and any(
            morpheme.dictionary_form in left_names
            for morpheme in right[:2]
        )
    )


def _append_comma(sentence: Sentence) -> Sentence:
    words = sentence.words
    if words:
        last = words[-1]
        words = (
            *words[:-1],
            Word(
                text=f"{_word_text(last)}、",
                time_range=last.time_range,
                confidence=last.confidence,
            ),
        )
    return Sentence(
        text=f"{sentence.text.rstrip()}、",
        time_range=sentence.time_range,
        words=words,
        asr_boundary_word_indexes=sentence.asr_boundary_word_indexes,
    )


__all__ = [
    "DEFAULT_SENTENCE_BOUNDARY_FINAL_SUFFIXES",
    "DEFAULT_SENTENCE_BOUNDARY_MIN_PAUSE_SECONDS",
    "DEFAULT_SENTENCE_BOUNDARY_TERMINAL_MARKS",
    "JapaneseSentenceBoundaryResolver",
]
