"""Sudachi-backed normalization of aligned ASR tokens into learning words."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.workflow.word_normalization_stage import (
    WordNormalization,
    WordNormalizationRequest,
)

DEFAULT_LOCAL_REANALYSIS_MAX_MORPHEMES = 3


class WordNormalizerDependencyError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class JapaneseMorpheme:
    surface: str
    part_of_speech: tuple[str, ...]
    dictionary_form: str = ""
    normalized_form: str = ""
    conjugation_type: str = ""
    conjugation_form: str = ""


class JapaneseMorphologicalAnalyzer(Protocol):
    def analyze(self, text: str) -> tuple[JapaneseMorpheme, ...]: ...


class SudachiMorphologicalAnalyzer:
    def __init__(self) -> None:
        try:
            from sudachipy import dictionary, tokenizer
        except ImportError as error:
            raise WordNormalizerDependencyError(
                "Word normalization requires sudachipy and sudachidict-core."
            ) from error
        self._tokenizer = dictionary.Dictionary().create()
        self._mode = tokenizer.Tokenizer.SplitMode.C

    def analyze(self, text: str) -> tuple[JapaneseMorpheme, ...]:
        return tuple(
            JapaneseMorpheme(
                surface=item.surface(),
                part_of_speech=tuple(item.part_of_speech()),
                dictionary_form=item.dictionary_form(),
                normalized_form=item.normalized_form(),
                conjugation_type=item.part_of_speech()[4],
                conjugation_form=item.part_of_speech()[5],
            )
            for item in self._tokenizer.tokenize(text, self._mode)
            if item.surface().strip()
        )


@dataclass(frozen=True, slots=True)
class _LearningUnit:
    text: str
    start: int
    end: int
    pos: tuple[str, ...]


class JapaneseLearningWordNormalizer:
    """Use morphology rules, not sentence-specific replacements."""

    def __init__(self, analyzer: JapaneseMorphologicalAnalyzer | None = None) -> None:
        self._analyzer = analyzer or SudachiMorphologicalAnalyzer()

    def normalize(self, request: WordNormalizationRequest) -> WordNormalization:
        return WordNormalization(
            source_path=request.source_path,
            segments=tuple(self._normalize_segment(segment) for segment in request.segments),
        )

    def _normalize_segment(self, segment: Segment) -> Segment:
        return Segment(
            position=segment.position,
            text=segment.text,
            time_range=segment.time_range,
            speaker_id=segment.speaker_id,
            sentences=tuple(self._normalize_sentence(item) for item in segment.sentences),
        )

    def _normalize_sentence(self, sentence: Sentence) -> Sentence:
        morphemes = self._analyzer.analyze(sentence.text)
        units = self._learning_units(morphemes)
        if not units:
            return sentence
        total_chars = max(units[-1].end, 1)
        words = tuple(
            self._make_word(unit, sentence, total_chars) for unit in units
        )
        return Sentence(
            text=sentence.text,
            time_range=sentence.time_range,
            words=words,
            speaker_id=sentence.speaker_id,
        )

    def _learning_units(self, morphemes: tuple[JapaneseMorpheme, ...]) -> tuple[_LearningUnit, ...]:
        raw: list[_LearningUnit] = []
        cursor = 0
        for item in morphemes:
            start, end = cursor, cursor + len(item.surface)
            cursor = end
            # でも is one functional learning unit, independently of its host word.
            if self._continues_compound_particle(item, raw):
                previous = raw.pop()
                raw.append(_LearningUnit("でも", previous.start, end, item.part_of_speech))
            # サ変可能名詞＋「する」の活用を一つの主要動詞にする。
            elif self._continues_sahen_verb(item, raw):
                previous = raw.pop()
                raw.append(
                    _LearningUnit(
                        previous.text + item.surface,
                        previous.start,
                        end,
                        item.part_of_speech,
                    )
                )
            # 接続助詞の「て/で」は直前の活用語に付ける（聞いて、話して）。
            elif self._continues_te_form(item, raw):
                previous = raw.pop()
                raw.append(_LearningUnit(previous.text + item.surface, previous.start, end, previous.pos))
            # 活用助動詞を主要動詞・形容詞または補助動詞自身に付ける。
            elif self._continues_inflection(item, raw):
                previous = raw.pop()
                raw.append(_LearningUnit(previous.text + item.surface, previous.start, end, previous.pos))
            # 「高く＋ない」のような非自立形容詞は直前の形容詞に付ける。
            elif self._continues_adjective_inflection(item, raw):
                previous = raw.pop()
                raw.append(_LearningUnit(previous.text + item.surface, previous.start, end, previous.pos))
            elif item.part_of_speech[0] == "補助記号" and raw:
                previous = raw.pop()
                raw.append(_LearningUnit(previous.text + item.surface, previous.start, end, previous.pos))
            else:
                raw.append(_LearningUnit(item.surface, start, end, item.part_of_speech))
        return self._merge_reanalyzed_nominals(raw)

    def _merge_reanalyzed_nominals(
        self,
        units: list[_LearningUnit],
    ) -> tuple[_LearningUnit, ...]:
        merged: list[_LearningUnit] = []
        index = 0
        while index < len(units):
            selected: _LearningUnit | None = None
            selected_size = 1
            maximum_size = min(
                DEFAULT_LOCAL_REANALYSIS_MAX_MORPHEMES,
                len(units) - index,
            )
            for size in range(maximum_size, 1, -1):
                candidates = units[index : index + size]
                if not all(self._is_nominal_unit(item) for item in candidates):
                    continue
                combined_text = "".join(item.text for item in candidates)
                analysis = self._analyzer.analyze(combined_text)
                if not self._is_complete_noun(combined_text, analysis):
                    continue
                selected = _LearningUnit(
                    text=combined_text,
                    start=candidates[0].start,
                    end=candidates[-1].end,
                    pos=analysis[0].part_of_speech,
                )
                selected_size = size
                break

            merged.append(selected or units[index])
            index += selected_size
        return tuple(merged)

    @staticmethod
    def _is_nominal_unit(unit: _LearningUnit) -> bool:
        if not unit.pos:
            return False
        if unit.pos[0] == "名詞":
            return True
        return (
            unit.pos[0] == "接尾辞"
            and len(unit.pos) > 1
            and unit.pos[1] == "名詞的"
        )

    @staticmethod
    def _is_complete_noun(
        combined_text: str,
        analysis: tuple[JapaneseMorpheme, ...],
    ) -> bool:
        return (
            len(analysis) == 1
            and analysis[0].surface == combined_text
            and bool(analysis[0].part_of_speech)
            and analysis[0].part_of_speech[0] == "名詞"
        )

    @staticmethod
    def _continues_compound_particle(
        item: JapaneseMorpheme,
        units: list[_LearningUnit],
    ) -> bool:
        return (
            item.surface == "も"
            and bool(units)
            and units[-1].text == "で"
            and units[-1].pos[0] == "助詞"
        )

    @staticmethod
    def _continues_sahen_verb(
        item: JapaneseMorpheme,
        units: list[_LearningUnit],
    ) -> bool:
        return (
            bool(units)
            and item.part_of_speech[0] == "動詞"
            and item.dictionary_form == "する"
            and len(units[-1].pos) > 2
            and units[-1].pos[0] == "名詞"
            and units[-1].pos[2] == "サ変可能"
        )

    @staticmethod
    def _continues_te_form(
        item: JapaneseMorpheme,
        units: list[_LearningUnit],
    ) -> bool:
        return (
            item.surface in {"て", "で"}
            and len(item.part_of_speech) > 1
            and item.part_of_speech[1] == "接続助詞"
            and bool(units)
            and units[-1].pos[0] in {"動詞", "形容詞"}
        )

    @staticmethod
    def _continues_inflection(
        item: JapaneseMorpheme,
        units: list[_LearningUnit],
    ) -> bool:
        return (
            item.part_of_speech[0] == "助動詞"
            and bool(units)
            and units[-1].pos[0] in {"動詞", "形容詞", "形状詞"}
        )

    @staticmethod
    def _continues_adjective_inflection(
        item: JapaneseMorpheme,
        units: list[_LearningUnit],
    ) -> bool:
        return (
            bool(units)
            and item.part_of_speech[0] == "形容詞"
            and len(item.part_of_speech) > 1
            and item.part_of_speech[1] == "非自立可能"
            and units[-1].pos[0] == "形容詞"
        )

    def _make_word(self, unit: _LearningUnit, sentence: Sentence, total_chars: int) -> Word:
        source_words = sentence.words
        if source_words:
            overlaps = self._overlapping_words(unit, source_words, total_chars)
            start = self._time_at(unit.start, source_words, total_chars)
            end = self._time_at(unit.end, source_words, total_chars)
            confidences = [word.confidence for word in overlaps if word.confidence is not None]
            speakers = {word.speaker_id for word in overlaps if word.speaker_id is not None}
            confidence = min(confidences) if confidences else None
            speaker = next(iter(speakers)) if len(speakers) == 1 else sentence.speaker_id
        else:
            duration = sentence.time_range.duration_seconds
            start = sentence.time_range.start_seconds + duration * unit.start / total_chars
            end = sentence.time_range.start_seconds + duration * unit.end / total_chars
            confidence, speaker = None, sentence.speaker_id
        return Word(unit.text, TimeRange(start, end), confidence, speaker)

    @staticmethod
    def _source_spans(words: tuple[Word, ...], total_chars: int) -> tuple[tuple[int, int, Word], ...]:
        lengths = [max(len(word.text.replace(" ", "")), 1) for word in words]
        source_total = sum(lengths)
        spans, cursor = [], 0
        for length, word in zip(lengths, words, strict=True):
            end = cursor + length
            spans.append((round(cursor * total_chars / source_total), round(end * total_chars / source_total), word))
            cursor = end
        return tuple(spans)

    def _overlapping_words(self, unit: _LearningUnit, words: tuple[Word, ...], total_chars: int) -> tuple[Word, ...]:
        return tuple(word for start, end, word in self._source_spans(words, total_chars) if start < unit.end and unit.start < end)

    def _time_at(self, offset: int, words: tuple[Word, ...], total_chars: int) -> float:
        spans = self._source_spans(words, total_chars)
        if offset <= 0:
            return words[0].time_range.start_seconds
        if offset >= total_chars:
            return words[-1].time_range.end_seconds
        for start, end, word in spans:
            if start <= offset <= end and end > start:
                fraction = (offset - start) / (end - start)
                return word.time_range.start_seconds + word.time_range.duration_seconds * fraction
        return words[-1].time_range.end_seconds


__all__ = [
    "JapaneseLearningWordNormalizer",
    "JapaneseMorpheme",
    "JapaneseMorphologicalAnalyzer",
    "SudachiMorphologicalAnalyzer",
    "WordNormalizerDependencyError",
]
