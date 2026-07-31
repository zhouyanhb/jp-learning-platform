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
    SentenceBoundaryDecision,
    SentenceBoundaryResolution,
    SentenceBoundaryResolutionRequest,
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
        for segment in segments:
            resolved_segment, segment_decisions = self._resolve_segment(segment)
            resolved_segments.append(resolved_segment)
            decisions.extend(segment_decisions)

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
        )

    def _resolve_segment(
        self,
        segment: Segment,
    ) -> tuple[Segment, tuple[SentenceBoundaryDecision, ...]]:
        sentences = segment.sentences or (
            Sentence(
                text=segment.text,
                time_range=segment.time_range,
                words=(),
            ),
        )

        resolved_sentences: list[Sentence] = []
        decisions: list[SentenceBoundaryDecision] = []
        for sentence_index, sentence in enumerate(sentences):
            sentence_parts, sentence_decisions = self._split_sentence(
                segment.position,
                sentence_index,
                sentence,
            )
            resolved_sentences.extend(sentence_parts)
            decisions.extend(sentence_decisions)

        if tuple(resolved_sentences) == sentences:
            return segment, tuple(decisions)

        return (
            Segment(
                position=segment.position,
                text="".join(sentence.text for sentence in resolved_sentences),
                time_range=segment.time_range,
                sentences=tuple(resolved_sentences),
            ),
            tuple(decisions),
        )

    def _split_sentence(
        self,
        segment_position: int,
        sentence_index: int,
        sentence: Sentence,
    ) -> tuple[tuple[Sentence, ...], tuple[SentenceBoundaryDecision, ...]]:
        if len(sentence.words) < 2:
            return (sentence,), ()

        boundaries: list[int] = []
        asr_boundaries = _asr_boundary_word_indexes(sentence)
        structured_numbering_starts = _structured_numbering_start_indexes(
            sentence.words,
            self.morphological_analyzer,
            asr_boundaries,
        )
        decisions: list[SentenceBoundaryDecision] = []
        chunk_start = 0
        for word_index in range(len(sentence.words) - 1):
            if word_index + 1 in structured_numbering_starts:
                reason = "structured_numbering_sequence"
            else:
                reason = self._boundary_reason(
                    sentence.words,
                    chunk_start,
                    word_index,
                    word_index + 1 in asr_boundaries,
                )
            if reason is None:
                continue

            left_text = _words_text(sentence.words[chunk_start : word_index + 1])
            right_text = _words_text(sentence.words[word_index + 1 :])
            if not left_text or not right_text:
                continue

            gap_seconds = _effective_word_gap_seconds(
                sentence.words[word_index],
                sentence.words[word_index + 1],
            )
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
            return (sentence,), ()

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
        return tuple(parts), tuple(decisions)

    def _boundary_reason(
        self,
        words: tuple[Word, ...],
        chunk_start: int,
        word_index: int,
        is_asr_boundary: bool = False,
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
        if _starts_with_sentence_final_particle(right_text):
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

        if (
            self.morphological_analyzer is not None
            and gap_seconds
            <= DEFAULT_SENTENCE_BOUNDARY_CONFIG.max_syntactic_dependency_gap_seconds
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
        if (candidate := _numbering_candidate(words, index, word, analyzer))
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
        _number_has_lexical_host(words, index, analyzer),
    )


def _number_has_lexical_host(
    words: tuple[Word, ...],
    index: int,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> bool:
    lookahead = _compact_text(_words_text(words[index : index + 4]))
    analyzed = analyzer.analyze(lookahead)
    if len(analyzed) < 2 or not _is_numeric_morpheme(analyzed[0]):
        return False
    following = analyzed[1]
    if not following.part_of_speech:
        return False
    major = following.part_of_speech[0]
    minor = following.part_of_speech[1] if len(following.part_of_speech) > 1 else ""
    return bool(
        major in {"助詞", "助動詞"}
        or (
            major == "接尾辞"
            and minor == "名詞的"
            and "助数詞" in following.part_of_speech
        )
        or (major == "名詞" and "助数詞可能" in following.part_of_speech)
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
    run: list[int] = []
    index = 0
    while index < len(entries):
        current = entries[index]
        numbered = _leading_number_and_body(current.sentence.text)
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
            gap_seconds < 0
            or gap_seconds > config.max_continuation_candidate_gap_seconds
            or preserves_numbering_restart
            or _ends_with_terminal_mark(
                left.text,
                (*config.terminal_marks, "?", "!"),
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
        remainder = analyzer.analyze(_compact_text(_words_text(right.words[count:])))
        if (
            combined
            and remainder
            and _is_complete_clause(combined)
            and _starts_independent_clause(remainder)
            and not _has_cross_boundary_morphological_dependency(
                combined_text,
                _words_text(right.words[count:]),
                analyzer,
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
        gap_seconds = right.time_range.start_seconds - left.time_range.end_seconds
        if (
            gap_seconds < 0
            or gap_seconds > config.max_connective_continuation_gap_seconds
            or not _ends_with_connective_te(left.text)
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
    if _is_connective_morpheme(left_last):
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
        and not _is_question_clause(right)
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
    if _is_question_clause(right):
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
