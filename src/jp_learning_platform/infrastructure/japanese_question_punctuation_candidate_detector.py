"""Conservative morphology-only Japanese question candidate detector."""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from jp_learning_platform.domain import Sentence, TimeRange
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    JapaneseMorpheme,
    JapaneseMorphologicalAnalyzer,
)
from jp_learning_platform.workflow.question_punctuation_candidate_stage import (
    QuestionPunctuationCandidate,
)


_EXPLICIT_INTERROGATIVE_TERMINAL_PARTICLES = frozenset(("か", "の"))
_QUOTATIVE_PARTICLES = frozenset(("って", "と"))
_REFORMULATION_PREDICATES = frozenset(("言う",))
_QUOTED_SPEECH_PREDICATES = frozenset(("言う", "聞く", "尋ねる", "問う"))
_INTERROGATIVE_FORMS = frozenset(
    (
        "何",
        "何処",
        "どこ",
        "何方",
        "どちら",
        "何故",
        "なぜ",
        "如何",
        "どう",
        "どういう",
        "何時",
        "いつ",
        "いつ頃",
        "幾つ",
        "いくつ",
        "幾ら",
        "いくら",
        "誰",
        "だれ",
    )
)


@dataclass(frozen=True, slots=True)
class ConservativeJapaneseQuestionCandidateDetector:
    analyzer: JapaneseMorphologicalAnalyzer
    maximum_elliptical_duration_seconds: float = 2.0
    maximum_response_gap_seconds: float = 2.0

    def detect_all(
        self,
        segment_position: int,
        sentence_index: int,
        sentence: Sentence,
        following: Sentence | None,
    ) -> tuple[QuestionPunctuationCandidate, ...]:
        embedded = _embedded_quoted_question_candidate(
            segment_position,
            sentence_index,
            sentence,
            self.analyzer,
        )
        terminal = self.detect(
            segment_position,
            sentence_index,
            sentence,
            following,
        )
        return tuple(item for item in (embedded, terminal) if item is not None)

    def detect(
        self,
        segment_position: int,
        sentence_index: int,
        sentence: Sentence,
        following: Sentence | None,
    ) -> QuestionPunctuationCandidate | None:
        if _has_terminal_punctuation(sentence.text):
            return None
        morphemes = _meaningful(self.analyzer.analyze(sentence.text))
        if not morphemes:
            return None
        evidence: tuple[str, ...] = ()
        confidence = 0.0
        relation_evidence = _question_relation_evidence(
            sentence,
            morphemes,
            following,
            self.analyzer,
            self.maximum_response_gap_seconds,
        )
        if (
            sentence.is_question
            and _ends_in_terminal_particle(morphemes)
            and _has_complete_predicate(morphemes)
            and _has_no_forward_dependency(morphemes)
            and relation_evidence is not None
        ):
            evidence = (
                "semantic_question_boundary",
                "terminal_particle",
                "complete_predicate",
                relation_evidence,
                "no_forward_dependency",
            )
            confidence = 0.95
        elif _is_complete_polite_desu_ka(morphemes):
            polite_relation = (
                _adjacent_short_response_evidence(
                    sentence,
                    following,
                    self.analyzer,
                    self.maximum_response_gap_seconds,
                )
                if _is_negative_confirmation_tail(
                    _without_terminal_particles(morphemes)
                )
                else _following_independent_relation(
                    sentence,
                    following,
                    self.analyzer,
                    self.maximum_response_gap_seconds,
                    question_morphemes=morphemes,
                )
            )
            if polite_relation is not None:
                evidence = (
                    "polite_desu_ka_question",
                    "complete_predicate",
                    polite_relation,
                    "no_forward_dependency",
                )
                confidence = 0.95
        elif _is_complete_sentence_final_self_question(morphemes):
            self_question_relation = _following_independent_relation(
                sentence,
                following,
                self.analyzer,
                self.maximum_response_gap_seconds,
                question_morphemes=morphemes,
            )
            if (
                not _has_volitional_self_question_predicate(morphemes)
                and self_question_relation is not None
            ):
                evidence = (
                    "listener_directed_self_question_form",
                    "complete_predicate",
                    self_question_relation,
                    "no_forward_dependency",
                )
                confidence = 0.95
        elif _is_short_elliptical_question(
            sentence,
            morphemes,
            following,
            self.analyzer,
            self.maximum_elliptical_duration_seconds,
            self.maximum_response_gap_seconds,
        ):
            evidence = (
                "short_pronominal_case_phrase",
                "following_independent_response",
            )
            confidence = 0.95
        if not evidence:
            return None
        return QuestionPunctuationCandidate(
            segment_position=segment_position,
            sentence_index=sentence_index,
            time_range=sentence.time_range,
            text=sentence.text,
            confidence=confidence,
            evidence=evidence,
        )


def _meaningful(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> tuple[JapaneseMorpheme, ...]:
    return tuple(item for item in morphemes if item.part_of_speech[0] != "補助記号")


def _has_terminal_punctuation(text: str) -> bool:
    compact = text.rstrip()
    return bool(
        compact
        and unicodedata.category(compact[-1]).startswith("P")
    )


def _is_complete_polite_desu_ka(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    return bool(
        len(morphemes) >= 3
        and morphemes[-1].part_of_speech[:2] == ("助詞", "終助詞")
        and morphemes[-1].normalized_form == "か"
        and morphemes[-2].part_of_speech[0] == "助動詞"
        and morphemes[-2].normalized_form == "です"
        and _has_complete_predicate(morphemes)
        and _has_no_forward_dependency(morphemes)
    )


def _is_complete_sentence_final_self_question(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    core = _self_question_predicate_core(morphemes)
    return bool(
        core
        and _has_complete_predicate(core)
        and _has_no_forward_dependency(core)
    )


def _self_question_predicate_core(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> tuple[JapaneseMorpheme, ...]:
    # かな / のかな: Sudachi marks か and な as terminal particles.
    if (
        len(morphemes) >= 3
        and tuple(item.normalized_form for item in morphemes[-2:]) == ("か", "な")
        and all(
            item.part_of_speech[:2] == ("助詞", "終助詞")
            for item in morphemes[-2:]
        )
    ):
        core = morphemes[:-2]
        if (
            core
            and core[-1].normalized_form == "の"
            and core[-1].part_of_speech[0] == "助詞"
        ):
            core = core[:-1]
        return core
    # んだろう: explanatory の followed by volitional だ.
    if (
        len(morphemes) >= 3
        and morphemes[-1].part_of_speech[0] == "助動詞"
        and morphemes[-1].normalized_form == "だ"
        and "意志推量形" in morphemes[-1].conjugation_form
        and morphemes[-2].part_of_speech[0] == "助詞"
        and morphemes[-2].normalized_form == "の"
    ):
        return morphemes[:-2]
    return ()


def _has_volitional_self_question_predicate(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    core = _self_question_predicate_core(morphemes)
    return bool(
        core
        and core[-1].part_of_speech[0] in {"動詞", "助動詞"}
        and "意志推量形" in core[-1].conjugation_form
    )


def _following_independent_relation(
    sentence: Sentence,
    following: Sentence | None,
    analyzer: JapaneseMorphologicalAnalyzer,
    maximum_gap_seconds: float,
    question_morphemes: tuple[JapaneseMorpheme, ...] = (),
) -> str | None:
    if following is None or following.is_question:
        return None
    gap = following.time_range.start_seconds - sentence.time_range.end_seconds
    if not 0.0 <= gap <= maximum_gap_seconds:
        return None
    response = _meaningful(analyzer.analyze(following.text))
    if not response or _has_explicit_interrogative_terminal(response):
        return None
    if _is_short_standalone_response(response):
        return "adjacent_independent_response"
    if (
        _contains_interrogative_form(question_morphemes)
        and _has_complete_answer_clause(response)
        and _has_no_forward_dependency(response)
    ):
        return "following_complete_answer"
    if (
        _contains_interrogative_form(question_morphemes)
        and _is_entity_nominal_answer(response)
    ):
        return "following_entity_answer"
    if _is_independent_topic_restart(response):
        return "following_topic_restart"
    return None


def _is_entity_nominal_answer(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    if not morphemes or len(morphemes) > 8:
        return False
    allowed = {"名詞", "接頭辞", "接尾辞", "助詞"}
    if any(item.part_of_speech[0] not in allowed for item in morphemes):
        return False
    if morphemes[-1].part_of_speech[0] != "名詞":
        return False
    return any(
        item.part_of_speech[:2] == ("名詞", "固有名詞")
        for item in morphemes
    )


def _has_complete_answer_clause(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    if _has_complete_predicate(morphemes):
        return True
    core = _without_terminal_particles(morphemes)
    if (
        len(core) >= 2
        and core[-2].part_of_speech[0] == "形状詞"
        and core[-1].part_of_speech[0] == "助動詞"
        and core[-1].normalized_form == "です"
        and "終止形" in core[-1].conjugation_form
    ):
        return True
    if len(core) < 3:
        return False
    explanatory, copula = core[-2:]
    if not (
        explanatory.part_of_speech[0] == "助詞"
        and explanatory.normalized_form == "の"
        and copula.part_of_speech[0] == "助動詞"
        and copula.normalized_form == "だ"
        and "終止形" in copula.conjugation_form
    ):
        return False
    predicate = core[-3]
    return bool(
        predicate.part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
        and any(
            form in predicate.conjugation_form
            for form in ("終止形", "連体形")
        )
    )


def _adjacent_short_response_evidence(
    sentence: Sentence,
    following: Sentence | None,
    analyzer: JapaneseMorphologicalAnalyzer,
    maximum_gap_seconds: float,
) -> str | None:
    if following is None or following.is_question:
        return None
    gap = following.time_range.start_seconds - sentence.time_range.end_seconds
    if not 0.0 <= gap <= maximum_gap_seconds:
        return None
    response = _meaningful(analyzer.analyze(following.text))
    if (
        response
        and not _has_explicit_interrogative_terminal(response)
        and _is_short_standalone_response(response)
    ):
        return "adjacent_independent_response"
    return None


def _is_independent_topic_restart(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    if not _has_complete_predicate(morphemes) or not _has_no_forward_dependency(
        morphemes
    ):
        return False
    if morphemes[0].part_of_speech[0] == "接続詞":
        return True
    return any(
        item.part_of_speech[0] == "助詞"
        and item.part_of_speech[1] == "係助詞"
        and item.normalized_form in {"は", "も"}
        for item in morphemes[:-1]
    )


def _ends_in_terminal_particle(morphemes: tuple[JapaneseMorpheme, ...]) -> bool:
    return morphemes[-1].part_of_speech[:2] == ("助詞", "終助詞")


def _without_terminal_particles(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> tuple[JapaneseMorpheme, ...]:
    end = len(morphemes)
    while end and morphemes[end - 1].part_of_speech[:2] == ("助詞", "終助詞"):
        end -= 1
    return morphemes[:end]


def _has_complete_predicate(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    core = _without_terminal_particles(morphemes)
    if not core:
        return False
    predicate = core[-1]
    major = predicate.part_of_speech[0]
    if major in {"動詞", "形容詞"}:
        return any(
            form in predicate.conjugation_form
            for form in ("終止形", "連体形", "意志推量形")
        )
    if major != "助動詞" or not any(
        form in predicate.conjugation_form
        for form in ("終止形", "連体形", "意志推量形")
    ):
        return False
    if len(core) < 2:
        return False
    host_major = core[-2].part_of_speech[0]
    return host_major in {"名詞", "代名詞", "動詞", "形容詞", "副詞"}


def _has_no_forward_dependency(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    core = _without_terminal_particles(morphemes)
    if not core:
        return False
    return bool(
        core[-1].part_of_speech[0] in {"動詞", "形容詞", "助動詞"}
        and not _has_discourse_reformulation_tail(core)
    )


def _has_discourse_reformulation_tail(
    core: tuple[JapaneseMorpheme, ...],
) -> bool:
    if len(core) < 2:
        return False
    linker, predicate = core[-2:]
    return bool(
        linker.part_of_speech[0] == "助詞"
        and linker.normalized_form in _QUOTATIVE_PARTICLES
        and predicate.part_of_speech[0] == "動詞"
        and predicate.normalized_form in _REFORMULATION_PREDICATES
        and "終止形" in predicate.conjugation_form
    )


def _question_relation_evidence(
    sentence: Sentence,
    morphemes: tuple[JapaneseMorpheme, ...],
    following: Sentence | None,
    analyzer: JapaneseMorphologicalAnalyzer,
    maximum_gap_seconds: float,
) -> str | None:
    if _has_explicit_interrogative_terminal(morphemes):
        return "explicit_interrogative_structure"
    if (
        morphemes[-1].normalized_form != "ね"
        and not _is_negative_confirmation_tail(
            _without_terminal_particles(morphemes)
        )
    ):
        return None
    if following is None or following.is_question:
        return None
    gap = following.time_range.start_seconds - sentence.time_range.end_seconds
    if not 0.0 <= gap <= maximum_gap_seconds:
        return None
    response = _meaningful(analyzer.analyze(following.text))
    if _has_explicit_interrogative_terminal(response):
        return None
    if not _is_short_standalone_response(response):
        return None
    return "adjacent_independent_response"


def _has_explicit_interrogative_terminal(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    if not _ends_in_terminal_particle(morphemes):
        return False
    terminal = morphemes[-1].normalized_form
    if terminal not in _EXPLICIT_INTERROGATIVE_TERMINAL_PARTICLES:
        return False
    if terminal == "の":
        return True
    core = _without_terminal_particles(morphemes)
    if _is_negative_confirmation_tail(core):
        return False
    return _contains_interrogative_form(core) or _ends_in_polite_predicate(core)


def _is_negative_confirmation_tail(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    if len(morphemes) < 2:
        return False
    polite = morphemes[-1]
    negative = morphemes[-2]
    return bool(
        polite.part_of_speech[0] == "助動詞"
        and polite.normalized_form == "です"
        and negative.part_of_speech[0] in {"形容詞", "助動詞"}
        and negative.normalized_form == "無い"
    )


def _contains_interrogative_form(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    return any(
        item.normalized_form in _INTERROGATIVE_FORMS
        for item in morphemes
        if item.part_of_speech[0] in {"名詞", "代名詞", "副詞", "連体詞"}
    )


def _ends_in_polite_predicate(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    return bool(
        morphemes
        and morphemes[-1].part_of_speech[0] == "助動詞"
        and morphemes[-1].normalized_form in {"です", "ます"}
    )


def _is_short_standalone_response(
    morphemes: tuple[JapaneseMorpheme, ...],
) -> bool:
    if not morphemes or len(morphemes) > 4:
        return False
    if morphemes[0].part_of_speech[0] == "感動詞":
        return True
    if morphemes[0].part_of_speech[0] != "副詞":
        return False
    return bool(
        _has_complete_predicate(morphemes)
        or _ends_in_terminal_particle(morphemes)
    )


def _embedded_quoted_question_candidate(
    segment_position: int,
    sentence_index: int,
    sentence: Sentence,
    analyzer: JapaneseMorphologicalAnalyzer,
) -> QuestionPunctuationCandidate | None:
    morphemes = _meaningful(analyzer.analyze(sentence.text))
    quote_index = next(
        (
            index
            for index in range(1, len(morphemes) - 1)
            if morphemes[index].part_of_speech[0] == "助詞"
            and morphemes[index].normalized_form in _QUOTATIVE_PARTICLES
            and morphemes[index + 1].part_of_speech[0] == "動詞"
            and morphemes[index + 1].normalized_form in _QUOTED_SPEECH_PREDICATES
            and _is_implicit_quoted_question(morphemes[:index])
        ),
        None,
    )
    if quote_index is None:
        return None
    quoted = morphemes[:quote_index]
    quote_start = _quoted_question_start_index(quoted)
    prefix_text = "".join(item.surface for item in quoted[:quote_start])
    quoted_text = "".join(item.surface for item in quoted[quote_start:])
    quoted_words = _words_for_text_span(sentence, prefix_text, quoted_text)
    if not quoted_words:
        return None
    return QuestionPunctuationCandidate(
        segment_position=segment_position,
        sentence_index=sentence_index,
        time_range=TimeRange(
            quoted_words[0].time_range.start_seconds,
            quoted_words[-1].time_range.end_seconds,
        ),
        text=quoted_text,
        confidence=0.95,
        evidence=(
            "embedded_quoted_question",
            "quotative_speech_predicate",
            "implicit_permission_question",
        ),
    )


def _is_implicit_quoted_question(
    quoted: tuple[JapaneseMorpheme, ...],
) -> bool:
    if len(quoted) < 2:
        return False
    predicate = quoted[-1]
    previous = quoted[-2]
    return bool(
        predicate.part_of_speech[0] == "形容詞"
        and predicate.normalized_form == "良い"
        and previous.part_of_speech[0] == "助詞"
        and previous.normalized_form == "て"
    )


def _quoted_question_start_index(
    quoted: tuple[JapaneseMorpheme, ...],
) -> int:
    predicate_index = next(
        (
            index
            for index in range(len(quoted) - 2, -1, -1)
            if quoted[index].part_of_speech[0] == "動詞"
        ),
        None,
    )
    if predicate_index is None:
        return 0
    boundary_index = next(
        (
            index
            for index in range(predicate_index - 1, -1, -1)
            if quoted[index].part_of_speech[:2] == ("助詞", "格助詞")
            and quoted[index].normalized_form in {"で", "では"}
        ),
        None,
    )
    if boundary_index is None:
        return 0
    between = quoted[boundary_index + 1 : predicate_index]
    if not any(
        item.part_of_speech[0] in {"名詞", "代名詞"}
        for item in between
    ):
        return 0
    return boundary_index + 1


def _words_for_text_span(
    sentence: Sentence,
    prefix_text: str,
    span_text: str,
) -> tuple:
    prefix_length = len(_compact(prefix_text))
    target_end = prefix_length + len(_compact(span_text))
    consumed = 0
    selected = []
    for word in sentence.words:
        word_start = consumed
        consumed += len(_compact(word.text))
        if word_start >= prefix_length and consumed <= target_end:
            selected.append(word)
        elif word_start < prefix_length < consumed:
            return ()
        elif word_start < target_end < consumed:
            return ()
        if consumed >= target_end:
            break
    return tuple(selected) if consumed == target_end and selected else ()


def _compact(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def _is_short_elliptical_question(
    sentence: Sentence,
    morphemes: tuple[JapaneseMorpheme, ...],
    following: Sentence | None,
    analyzer: JapaneseMorphologicalAnalyzer,
    maximum_duration_seconds: float,
    maximum_gap_seconds: float,
) -> bool:
    if following is None or not 2 <= len(morphemes) <= 3:
        return False
    gap = following.time_range.start_seconds - sentence.time_range.end_seconds
    following_morphemes = _meaningful(analyzer.analyze(following.text))
    return bool(
        sentence.time_range.duration_seconds <= maximum_duration_seconds
        and 0.0 <= gap <= maximum_gap_seconds
        and morphemes[0].part_of_speech[0] == "代名詞"
        and morphemes[-1].part_of_speech[:2] == ("助詞", "格助詞")
        and following_morphemes
        and following_morphemes[0].part_of_speech[0]
        in {"名詞", "代名詞", "動詞", "形容詞", "副詞", "接続詞", "感動詞"}
    )


__all__ = ["ConservativeJapaneseQuestionCandidateDetector"]
