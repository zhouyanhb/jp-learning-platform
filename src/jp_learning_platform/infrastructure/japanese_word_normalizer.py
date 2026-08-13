"""Sudachi-backed normalization of aligned ASR tokens into learning words."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from jp_learning_platform.domain import (
    LearningWord,
    Segment,
    Sentence,
    TimeRange,
    Word,
)
from jp_learning_platform.workflow.word_normalization_stage import (
    WordNormalization,
    WordNormalizationRequest,
)
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_SENTENCE_BOUNDARY_CONFIG,
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


def morphological_particle_chain_penalty(
    text: str,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> int:
    """Count unlikely dependent-predicate and particle chains."""
    morphemes = tuple(
        item
        for item in analyzer.analyze(text)
        if item.part_of_speech and item.part_of_speech[0] != "補助記号"
    )
    penalty = 0
    for first, second, third, fourth in zip(
        morphemes,
        morphemes[1:],
        morphemes[2:],
        morphemes[3:],
    ):
        if (
            len(first.part_of_speech) > 1
            and first.part_of_speech[0] in {"動詞", "形容詞"}
            and first.part_of_speech[1] == "非自立可能"
            and second.part_of_speech[:2] == ("助詞", "終助詞")
            and third.part_of_speech[:2] == ("助詞", "副助詞")
            and fourth.part_of_speech[:2] == ("助詞", "係助詞")
        ):
            penalty += 1
    return penalty


@dataclass(frozen=True, slots=True)
class _LearningUnit:
    text: str
    start: int
    end: int
    pos: tuple[str, ...]
    prefixed: bool = False
    suffixed: bool = False
    auxiliary_stem: bool = False
    is_structure: bool = False


@dataclass(frozen=True, slots=True)
class _LearningWordStructureContext:
    numbered_sentence_keys: frozenset[tuple[int, int]] = frozenset()


class JapaneseLearningWordNormalizer:
    """Use morphology rules, not sentence-specific replacements."""

    def __init__(self, analyzer: JapaneseMorphologicalAnalyzer | None = None) -> None:
        self._analyzer = analyzer or SudachiMorphologicalAnalyzer()

    def normalize(self, request: WordNormalizationRequest) -> WordNormalization:
        structure_context = self._structure_context(request.segments)
        return WordNormalization(
            source_path=request.source_path,
            segments=tuple(
                self._normalize_segment(segment, structure_context)
                for segment in request.segments
            ),
        )

    def _normalize_segment(
        self,
        segment: Segment,
        structure_context: _LearningWordStructureContext,
    ) -> Segment:
        return Segment(
            position=segment.position,
            text=segment.text,
            time_range=segment.time_range,
            sentences=tuple(
                self._normalize_sentence(
                    item,
                    (segment.position, index)
                    in structure_context.numbered_sentence_keys,
                )
                for index, item in enumerate(segment.sentences)
            ),
        )

    def _normalize_sentence(
        self,
        sentence: Sentence,
        has_structural_number: bool,
    ) -> Sentence:
        morphemes = self._analyzer.analyze(sentence.text)
        morphemes = self._repair_contextual_functional_boundaries(morphemes)
        units = self._learning_units(morphemes, has_structural_number)
        if not units:
            return sentence
        total_chars = max(sum(len(item.surface) for item in morphemes), 1)
        learning_words = tuple(
            self._make_learning_word(unit, sentence, total_chars) for unit in units
        )
        return Sentence(
            text=sentence.text,
            time_range=sentence.time_range,
            words=sentence.words,
            learning_words=learning_words,
            is_question=sentence.is_question,
            asr_boundary_word_indexes=sentence.asr_boundary_word_indexes,
        )

    def _repair_contextual_functional_boundaries(
        self,
        morphemes: tuple[JapaneseMorpheme, ...],
    ) -> tuple[JapaneseMorpheme, ...]:
        repaired: list[JapaneseMorpheme] = []
        index = 0
        while index < len(morphemes):
            if index + 1 >= len(morphemes):
                repaired.append(morphemes[index])
                break
            left, right = morphemes[index : index + 2]
            if not (
                left.part_of_speech
                and left.part_of_speech[0] == "接続詞"
                and right.part_of_speech
                and right.part_of_speech[0] == "名詞"
            ):
                repaired.append(left)
                index += 1
                continue
            combined = left.surface + right.surface
            try:
                local = self._analyzer.analyze(combined)
            except LookupError:
                local = ()
            if self._is_functional_boundary_reanalysis(combined, local):
                repaired.extend(local)
                index += 2
                continue
            repaired.append(left)
            index += 1
        return tuple(repaired)

    @staticmethod
    def _is_functional_boundary_reanalysis(
        combined: str,
        local: tuple[JapaneseMorpheme, ...],
    ) -> bool:
        return bool(
            len(local) == 2
            and "".join(item.surface for item in local) == combined
            and local[0].part_of_speech
            and local[0].part_of_speech[0] == "接続詞"
            and local[1].part_of_speech[:2] == ("助詞", "終助詞")
        )

    @staticmethod
    def _structure_context(
        segments: tuple[Segment, ...],
    ) -> _LearningWordStructureContext:
        numbered: list[tuple[tuple[int, int], int, float]] = []
        pattern = re.compile(r"^\s*([1-9]\d{0,2})")
        for segment in segments:
            for index, sentence in enumerate(segment.sentences):
                match = pattern.match(sentence.text)
                has_structural_boundary = bool(
                    match
                    and sentence.words
                    and sentence.asr_boundary_word_indexes
                    and sentence.asr_boundary_word_indexes[0] == 1
                    and sentence.words[0].text.strip().isdecimal()
                )
                has_explicit_separator = bool(
                    match
                    and match.end() < len(sentence.text)
                    and (
                        sentence.text[match.end()].isspace()
                        or unicodedata.category(sentence.text[match.end()]).startswith("P")
                    )
                )
                if match and (has_structural_boundary or has_explicit_separator):
                    numbered.append(
                        (
                            (segment.position, index),
                            int(match.group(1)),
                            sentence.time_range.start_seconds,
                        )
                    )

        structural_keys: set[tuple[int, int]] = set()
        run: list[tuple[tuple[int, int], int, float]] = []
        for item in (*numbered, None):
            if item is not None and (
                not run
                or (
                    item[1] == run[-1][1] + 1
                    and item[2] - run[-1][2]
                    <= DEFAULT_SENTENCE_BOUNDARY_CONFIG.numbering_region_max_item_gap_seconds
                )
            ):
                run.append(item)
                continue
            if len(run) >= 2 and run[0][1] == 1:
                structural_keys.update(key for key, _value, _start in run)
            run = [item] if item is not None else []
        return _LearningWordStructureContext(frozenset(structural_keys))

    def _learning_units(
        self,
        morphemes: tuple[JapaneseMorpheme, ...],
        has_structural_number: bool = False,
    ) -> tuple[_LearningUnit, ...]:
        raw: list[_LearningUnit] = []
        cursor = 0
        for item in morphemes:
            start, end = cursor, cursor + len(item.surface)
            cursor = end
            # 接頭辞は後続する中心語と組み合わせる。
            if self._continues_prefix(item, raw):
                previous = raw.pop()
                raw.append(
                    _LearningUnit(
                        previous.text + item.surface,
                        previous.start,
                        end,
                        item.part_of_speech,
                        prefixed=True,
                    )
                )
            # でも is one functional learning unit, independently of its host word.
            elif self._continues_compound_particle(item, raw):
                previous = raw.pop()
                raw.append(_LearningUnit("でも", previous.start, end, item.part_of_speech))
            # 名詞性語基＋非独立の「なさる」命令形は一つの学習表現にする。
            elif self._continues_dependent_imperative(item, raw):
                previous = raw.pop()
                raw.append(
                    _LearningUnit(
                        previous.text + item.surface,
                        previous.start,
                        end,
                        item.part_of_speech,
                        prefixed=previous.prefixed,
                    )
                )
            # サ変可能名詞＋「する」の活用を一つの主要動詞にする。
            elif self._continues_sahen_verb(item, raw):
                previous = raw.pop()
                raw.append(
                    _LearningUnit(
                        previous.text + item.surface,
                        previous.start,
                        end,
                        item.part_of_speech,
                        prefixed=previous.prefixed,
                    )
                )
            # 接続助詞の「て/で」は直前の活用語に付ける（聞いて、話して）。
            elif self._continues_te_form(item, raw):
                previous = raw.pop()
                raw.append(_LearningUnit(previous.text + item.surface, previous.start, end, previous.pos))
            # 非自立動詞に続く助動詞語幹を、後続する活用助動詞まで一つの鎖にする。
            elif self._continues_auxiliary_stem(item, raw):
                previous = raw.pop()
                raw.append(
                    _LearningUnit(
                        previous.text + item.surface,
                        previous.start,
                        end,
                        previous.pos,
                        auxiliary_stem=True,
                    )
                )
            # 活用語に続く助動詞と、連続する助動詞の活用鎖を一単位にする。
            elif self._continues_inflection(item, raw):
                previous = raw.pop()
                raw.append(
                    _LearningUnit(
                        previous.text + item.surface,
                        previous.start,
                        end,
                        previous.pos,
                        auxiliary_stem=previous.auxiliary_stem,
                    )
                )
            # 「高く＋ない」のような非自立形容詞は直前の形容詞に付ける。
            elif self._continues_adjective_inflection(item, raw):
                previous = raw.pop()
                raw.append(_LearningUnit(previous.text + item.surface, previous.start, end, previous.pos))
            elif self._is_numeric_enumeration_separator(item, raw):
                raw.append(_LearningUnit(item.surface, start, end, item.part_of_speech))
            elif item.part_of_speech[0] == "補助記号":
                raw.append(_LearningUnit(item.surface, start, end, item.part_of_speech))
            else:
                raw.append(
                    _LearningUnit(
                        item.surface,
                        start,
                        end,
                        item.part_of_speech,
                        is_structure=(
                            has_structural_number
                            and not raw
                            and item.surface.isdecimal()
                        ),
                    )
                )
        functional = self._merge_functional_units(raw)
        structural = self._merge_structural_units(functional)
        merged = self._merge_reanalyzed_nominals(structural)
        return tuple(unit for unit in merged if not self._is_pure_punctuation(unit))

    @staticmethod
    def _is_pure_punctuation(unit: _LearningUnit) -> bool:
        return bool(unit.text) and all(
            unicodedata.category(character).startswith("P")
            for character in unit.text
        )

    @classmethod
    def _merge_functional_units(
        cls,
        units: list[_LearningUnit],
    ) -> list[_LearningUnit]:
        merged: list[_LearningUnit] = []
        for unit in units:
            if merged and cls._forms_functional_learning_unit(merged[-1], unit):
                previous = merged.pop()
                merged.append(
                    _LearningUnit(
                        text=previous.text + unit.text,
                        start=previous.start,
                        end=unit.end,
                        pos=previous.pos,
                    )
                )
            else:
                merged.append(unit)
        return merged

    @classmethod
    def _forms_functional_learning_unit(
        cls,
        left: _LearningUnit,
        right: _LearningUnit,
    ) -> bool:
        if (
            left.is_structure
            or right.is_structure
            or cls._is_pure_punctuation(left)
            or cls._is_pure_punctuation(right)
        ):
            return False
        left_pos = left.pos[:2]
        right_pos = right.pos[:2]
        return bool(
            (
                left_pos == ("助詞", "副助詞")
                and right_pos == ("助詞", "係助詞")
            )
            or (
                left_pos == ("助詞", "終助詞")
                and right_pos == ("助詞", "終助詞")
            )
            or (
                left_pos == ("助詞", "準体助詞")
                and right.pos
                and right.pos[0] == "助動詞"
            )
        )

    def _merge_structural_units(
        self,
        units: list[_LearningUnit],
    ) -> list[_LearningUnit]:
        merged: list[_LearningUnit] = []
        for index, unit in enumerate(units):
            following = units[index + 1] if index + 1 < len(units) else None
            if merged and self._forms_structural_unit(merged[-1], unit, following):
                previous = merged.pop()
                merged.append(
                    _LearningUnit(
                        text=previous.text + unit.text,
                        start=previous.start,
                        end=unit.end,
                        pos=previous.pos,
                        prefixed=previous.prefixed,
                        suffixed=(
                            previous.suffixed
                            or self._is_nominal_suffix_unit(unit)
                        ),
                        auxiliary_stem=previous.auxiliary_stem or unit.auxiliary_stem,
                    )
                )
            else:
                merged.append(unit)
        return merged

    @classmethod
    def _forms_structural_unit(
        cls,
        left: _LearningUnit,
        right: _LearningUnit,
        following: _LearningUnit | None,
    ) -> bool:
        if left.is_structure or right.is_structure:
            return False
        return (
            cls._forms_ascii_identifier(left.text, right.text)
            or (
                not left.suffixed
                and cls._is_numeric_unit(left)
                and cls._is_counter_unit(right)
            )
            or cls._forms_nominal_suffix_unit(left, right)
            or cls._forms_person_title_unit(left, right, following)
            or (
                cls._is_nominal_unit(left)
                and cls._is_nominal_unit(right)
                and cls._is_katakana_text(left.text)
                and cls._is_katakana_text(right.text)
            )
        )

    @staticmethod
    def _is_numeric_enumeration_separator(
        item: JapaneseMorpheme,
        units: list[_LearningUnit],
    ) -> bool:
        return bool(
            units
            and JapaneseLearningWordNormalizer._is_numeric_unit(units[-1])
            and item.part_of_speech
            and item.part_of_speech[0] == "補助記号"
            and len(item.part_of_speech) > 1
            and item.part_of_speech[1] == "読点"
        )

    @staticmethod
    def _forms_ascii_identifier(left: str, right: str) -> bool:
        combined = left + right
        return (
            combined.isascii()
            and combined.isalnum()
            and any(character.isalpha() for character in combined)
            and any(character.isdigit() for character in combined)
        )

    @staticmethod
    def _is_numeric_unit(unit: _LearningUnit) -> bool:
        return unit.text.isdecimal() or (
            len(unit.pos) > 1
            and unit.pos[0] == "名詞"
            and unit.pos[1] == "数詞"
        )

    @staticmethod
    def _is_counter_unit(unit: _LearningUnit) -> bool:
        return unit.pos[0] == "名詞" and "助数詞可能" in unit.pos

    @classmethod
    def _forms_nominal_suffix_unit(
        cls,
        left: _LearningUnit,
        right: _LearningUnit,
    ) -> bool:
        return (
            not left.suffixed
            and
            len(right.pos) > 1
            and right.pos[0] == "接尾辞"
            and right.pos[1] == "名詞的"
            and (
                cls._is_numeric_unit(left)
                or left.prefixed
                or cls._is_person_reference(left)
                or cls._is_nominal_suffix_unit(left)
            )
        )

    @staticmethod
    def _is_nominal_suffix_unit(unit: _LearningUnit) -> bool:
        return bool(
            len(unit.pos) > 1
            and unit.pos[0] == "接尾辞"
            and unit.pos[1] == "名詞的"
        )

    @staticmethod
    def _is_person_reference(unit: _LearningUnit) -> bool:
        return bool(
            unit.pos
            and unit.pos[0] == "名詞"
            and (
                "副詞可能" in unit.pos
                or (
                    len(unit.pos) > 3
                    and unit.pos[1] == "固有名詞"
                    and unit.pos[2] == "人名"
                )
            )
        )

    @classmethod
    def _forms_person_title_unit(
        cls,
        left: _LearningUnit,
        right: _LearningUnit,
        following: _LearningUnit | None,
    ) -> bool:
        if not cls._is_person_name(left) or not cls._is_title_noun_candidate(right):
            return False
        return following is None or bool(
            following.pos
            and following.pos[0] in {"助詞", "助動詞", "補助記号"}
        )

    @staticmethod
    def _is_person_name(unit: _LearningUnit) -> bool:
        return bool(
            len(unit.pos) > 2
            and unit.pos[0] == "名詞"
            and unit.pos[1] == "固有名詞"
            and unit.pos[2] == "人名"
        )

    @staticmethod
    def _is_title_noun_candidate(unit: _LearningUnit) -> bool:
        return bool(
            len(unit.pos) > 2
            and unit.pos[0] == "名詞"
            and unit.pos[1] == "普通名詞"
            and unit.pos[2] == "一般"
            and unit.text
            and all(
                unicodedata.name(character, "").startswith("CJK UNIFIED IDEOGRAPH")
                for character in unit.text
            )
        )

    @staticmethod
    def _is_katakana_text(text: str) -> bool:
        return bool(text) and all(
            character in {"ー", "・"}
            or unicodedata.name(character, "").startswith("KATAKANA")
            for character in text
        )

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
                if any(item.is_structure for item in candidates):
                    continue
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
                    prefixed=any(item.prefixed for item in candidates),
                    suffixed=any(item.suffixed for item in candidates),
                    auxiliary_stem=any(item.auxiliary_stem for item in candidates),
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

    def _continues_prefix(
        self,
        item: JapaneseMorpheme,
        units: list[_LearningUnit],
    ) -> bool:
        if not (
            units
            and units[-1].pos
            and units[-1].pos[0] == "接頭辞"
            and item.part_of_speech
            and item.part_of_speech[0] in {
                "名詞",
                "代名詞",
                "動詞",
                "形容詞",
                "形状詞",
            }
        ):
            return False
        try:
            local = self._analyzer.analyze(f"{units[-1].text}{item.surface}")
        except LookupError:
            # Minimal test or plugin analyzers may only support whole inputs.
            return True
        combined_text = f"{units[-1].text}{item.surface}"
        if (
            len(local) == 1
            and local[0].surface == combined_text
            and local[0].part_of_speech
            and local[0].part_of_speech[0] != "補助記号"
        ):
            return True
        return bool(
            local
            and local[0].surface == units[-1].text
            and local[0].part_of_speech
            and local[0].part_of_speech[0] == "接頭辞"
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
        if not (
            units
            and item.part_of_speech
            and item.part_of_speech[0] == "動詞"
            and item.dictionary_form == "する"
        ):
            return False
        previous = units[-1]
        return bool(
            (
                len(previous.pos) > 2
                and previous.pos[0] == "名詞"
                and previous.pos[2] == "サ変可能"
            )
            or (
                previous.prefixed
                and previous.pos
                and previous.pos[0] == "動詞"
            )
        )

    @staticmethod
    def _continues_dependent_imperative(
        item: JapaneseMorpheme,
        units: list[_LearningUnit],
    ) -> bool:
        if not units or not item.part_of_speech or not units[-1].pos:
            return False
        previous = units[-1]
        return bool(
            item.part_of_speech[0] == "動詞"
            and len(item.part_of_speech) > 1
            and item.part_of_speech[1] == "非自立可能"
            and item.dictionary_form == "なさる"
            and "命令形" in item.conjugation_form
            and (
                previous.pos[0] in {"名詞", "形状詞"}
                or (
                    previous.pos[0] == "動詞"
                    and len(previous.pos) > 5
                    and "連用形" in previous.pos[5]
                )
            )
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
        if (
            item.conjugation_type == "助動詞-ダ"
            and bool(units)
            and units[-1].pos[0] != "形状詞"
            and not units[-1].auxiliary_stem
        ):
            return False
        return (
            item.part_of_speech[0] == "助動詞"
            and bool(units)
            and units[-1].pos[0] in {
                "動詞",
                "形容詞",
                "形状詞",
                "助動詞",
            }
        )

    @staticmethod
    def _continues_auxiliary_stem(
        item: JapaneseMorpheme,
        units: list[_LearningUnit],
    ) -> bool:
        if not units or not item.part_of_speech or not units[-1].pos:
            return False
        previous = units[-1]
        return bool(
            item.part_of_speech[0] == "形状詞"
            and len(item.part_of_speech) > 1
            and item.part_of_speech[1] == "助動詞語幹"
            and previous.pos[0] == "動詞"
            and len(previous.pos) > 1
            and previous.pos[1] == "非自立可能"
            and len(previous.pos) > 5
            and "連用形" in previous.pos[5]
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

    def _make_learning_word(
        self,
        unit: _LearningUnit,
        sentence: Sentence,
        total_chars: int,
    ) -> LearningWord:
        source_words = sentence.words
        if source_words:
            spans = self._source_spans(source_words, total_chars)
            aligned_indexes = tuple(
                index
                for index, (start, end, _word) in enumerate(spans)
                if start < unit.end and unit.start < end
            )
            start = self._time_at(unit.start, source_words, total_chars)
            end = self._time_at(unit.end, source_words, total_chars)
            timing_estimated = self._timing_is_estimated(
                unit,
                aligned_indexes,
                spans,
            )
        else:
            duration = sentence.time_range.duration_seconds
            start = sentence.time_range.start_seconds + duration * unit.start / total_chars
            end = sentence.time_range.start_seconds + duration * unit.end / total_chars
            aligned_indexes = ()
            timing_estimated = True
        return LearningWord(
            text=unit.text,
            start_char=unit.start,
            end_char=unit.end,
            aligned_word_indexes=aligned_indexes,
            time_range=TimeRange(start, end),
            timing_estimated=timing_estimated,
            is_structure=unit.is_structure,
        )

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

    @staticmethod
    def _timing_is_estimated(
        unit: _LearningUnit,
        aligned_indexes: tuple[int, ...],
        spans: tuple[tuple[int, int, Word], ...],
    ) -> bool:
        if not aligned_indexes:
            return True
        first_start = spans[aligned_indexes[0]][0]
        last_end = spans[aligned_indexes[-1]][1]
        return unit.start != first_start or unit.end != last_end

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
