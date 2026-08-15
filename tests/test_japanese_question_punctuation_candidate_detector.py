from pathlib import Path

from jp_learning_platform.domain import (
    Document,
    PipelineContext,
    Segment,
    Sentence,
    TimeRange,
    Word,
)
from jp_learning_platform.infrastructure.japanese_question_punctuation_candidate_detector import (
    ConservativeJapaneseQuestionCandidateDetector,
)
from jp_learning_platform.infrastructure.japanese_word_normalizer import (
    SudachiMorphologicalAnalyzer,
)
from jp_learning_platform.workflow.question_punctuation_candidate_stage import (
    QuestionPunctuationCandidateStage,
)


def _sentence(
    text: str,
    start: float,
    end: float,
    *,
    is_question: bool = False,
) -> Sentence:
    return Sentence(
        text,
        TimeRange(start, end),
        (Word(text, TimeRange(start, end), 0.9),),
        is_question=is_question,
    )


def _detector() -> ConservativeJapaneseQuestionCandidateDetector:
    return ConservativeJapaneseQuestionCandidateDetector(
        SudachiMorphologicalAnalyzer()
    )


def test_generates_short_elliptical_candidate_from_structure_and_response() -> None:
    sentence = _sentence("何を", 1.0, 1.6)
    following = _sentence("今回は行いません", 1.8, 3.0)

    candidate = _detector().detect(3, 0, sentence, following)

    assert candidate is not None
    assert candidate.text == "何を"
    assert candidate.evidence == (
        "short_pronominal_case_phrase",
        "following_independent_response",
    )


def test_generates_semantic_question_with_terminal_particle() -> None:
    candidate = _detector().detect(
        1,
        0,
        _sentence("そうですか", 1.0, 2.0, is_question=True),
        _sentence("はい", 2.2, 2.6),
    )

    assert candidate is not None
    assert candidate.evidence == (
        "semantic_question_boundary",
        "terminal_particle",
        "complete_predicate",
        "explicit_interrogative_structure",
        "no_forward_dependency",
    )


def test_semantic_question_requires_complete_predicate() -> None:
    detector = _detector()
    following = _sentence("次の話をします", 2.2, 3.2)

    for text in ("こちらでね", "そういうのってさ", "それぐらいかな", "でえーとですね"):
        assert detector.detect(
            0, 0, _sentence(text, 1.0, 2.0, is_question=True), following
        ) is None


def test_semantic_question_rejects_forward_dependent_ending() -> None:
    assert _detector().detect(
        0,
        0,
        _sentence("したけどね", 1.0, 2.0, is_question=True),
        _sentence("次の話をします", 2.2, 3.2),
    ) is None


def test_ambiguous_terminal_requires_adjacent_non_question_response() -> None:
    detector = _detector()
    sentence = _sentence("この席ですよね", 1.0, 2.0, is_question=True)

    assert detector.detect(0, 0, sentence, _sentence("ので", 2.2, 2.5)) is None
    assert detector.detect(0, 0, sentence, _sentence("はい", 4.2, 4.6)) is None
    assert detector.detect(
        0, 0, sentence, _sentence("そうですか", 2.2, 2.8, is_question=True)
    ) is None
    assert detector.detect(
        0, 0, sentence, _sentence("そうですか", 2.2, 2.8)
    ) is None


def test_ambiguous_terminal_accepts_adjacent_short_response() -> None:
    candidate = _detector().detect(
        0,
        0,
        _sentence("この席ですよね", 1.0, 2.0, is_question=True),
        _sentence("はい", 2.2, 2.6),
    )

    assert candidate is not None
    assert "adjacent_independent_response" in candidate.evidence


def test_informative_yo_is_not_promoted_by_following_response() -> None:
    assert _detector().detect(
        0,
        0,
        _sentence("私は先に行きますよ", 1.0, 2.0, is_question=True),
        _sentence("そうだね", 2.0, 2.5),
    ) is None


def test_rejects_quoted_reformulation_tail_as_question() -> None:
    assert _detector().detect(
        0,
        0,
        _sentence("何の集まりっていうか", 1.0, 2.0, is_question=True),
        None,
    ) is None


def test_reformulation_check_preserves_genuine_questions() -> None:
    detector = _detector()

    for text in ("彼が来るって本当ですか", "何を言いますか"):
        candidate = detector.detect(
            0,
            0,
            _sentence(text, 1.0, 2.0, is_question=True),
            None,
        )
        assert candidate is not None
        assert "explicit_interrogative_structure" in candidate.evidence


def test_plain_form_ka_requires_response_instead_of_assuming_question() -> None:
    detector = _detector()

    for text in ("うーん仕方ないか", "そうしたら並べるか"):
        assert detector.detect(
            0,
            0,
            _sentence(text, 1.0, 2.0, is_question=True),
            _sentence("じゃあそうしよう", 2.0, 3.0),
        ) is None


def test_ambiguous_plain_form_ka_is_conservatively_rejected_without_response() -> None:
    detector = _detector()

    for text in ("これって動くか", "どこに置くか"):
        assert detector.detect(
            0,
            0,
            _sentence(text, 1.0, 2.0, is_question=True),
            None,
        ) is None


def test_polite_interrogative_form_remains_explicit() -> None:
    candidate = _detector().detect(
        0,
        0,
        _sentence("どこに置きますか", 1.0, 2.0, is_question=True),
        None,
    )

    assert candidate is not None


def test_complete_desu_ka_requires_independent_response_or_topic_restart() -> None:
    detector = _detector()
    sentence = _sentence("よろしいですか", 1.0, 2.0)

    response_candidate = detector.detect(
        0, 0, sentence, _sentence("はい分かりました", 2.1, 2.8)
    )
    restart_candidate = detector.detect(
        0, 0, sentence, _sentence("次の予定は明日始まります", 2.1, 3.2)
    )

    assert response_candidate is not None
    assert "adjacent_independent_response" in response_candidate.evidence
    assert restart_candidate is not None
    assert "following_topic_restart" in restart_candidate.evidence
    assert detector.detect(0, 0, sentence, None) is None
    assert detector.detect(
        0, 0, sentence, _sentence("ので続けて説明します", 2.1, 3.2)
    ) is None


def test_sentence_final_self_question_requires_listener_response() -> None:
    detector = _detector()

    for text in ("同級生はいないのかな", "どうすればいいんだろう"):
        candidate = detector.detect(
            0,
            0,
            _sentence(text, 1.0, 2.0),
            _sentence("はいそうです", 2.1, 2.8),
        )
        assert candidate is not None
        assert "listener_directed_self_question_form" in candidate.evidence

    assert detector.detect(
        0, 0, _sentence("どうすればいいんだろう", 1.0, 2.0), None
    ) is None


def test_interrogative_self_question_accepts_complete_answer_clause() -> None:
    candidate = _detector().detect(
        0,
        0,
        _sentence("どうしたらいいかな", 1.0, 2.0),
        _sentence("僕の場合は友達に譲ります", 2.2, 3.2),
    )

    assert candidate is not None
    assert "following_complete_answer" in candidate.evidence


def test_interrogative_self_question_accepts_explanatory_and_negative_answers() -> None:
    detector = _detector()
    cases = (
        ("どうしたらいいかな", "僕の場合は友達に譲るんだ"),
        ("公衆電話ってどこにあるんだろう", "使ったことないから分かんないな"),
    )

    for question, answer in cases:
        candidate = detector.detect(
            0,
            0,
            _sentence(question, 1.0, 2.0),
            _sentence(answer, 2.1, 3.2),
        )
        assert candidate is not None
        assert "following_complete_answer" in candidate.evidence


def test_non_interrogative_kana_rejects_unrelated_complete_narration() -> None:
    assert _detector().detect(
        0,
        0,
        _sentence("主食になるのかな", 1.0, 2.0),
        _sentence("引き続き精進しようと心に誓った", 2.2, 3.2),
    ) is None


def test_volitional_kana_is_rejected_even_with_following_response() -> None:
    detector = _detector()
    following = _sentence("はいそうしましょう", 2.1, 2.8)

    for text in (
        "どうしようかな",
        "歩きながら話そうかな",
        "先生も一緒に遊ぼうかな",
        "ポッドキャストを撮ろうかな",
    ):
        assert detector.detect(
            0, 0, _sentence(text, 1.0, 2.0), following
        ) is None


def test_incomplete_kana_and_ambiguous_terminals_remain_rejected() -> None:
    detector = _detector()

    for text in (
        "それぐらいかな",
        "結婚しないの",
        "この席ですよね",
        "明日も来るでしょ",
    ):
        assert detector.detect(0, 0, _sentence(text, 1.0, 2.0), None) is None


def test_negative_polite_confirmation_requires_short_response() -> None:
    detector = _detector()
    sentence = _sentence("みんな許可を取るじゃないですか", 1.0, 2.0, is_question=True)

    assert detector.detect(
        0,
        0,
        sentence,
        _sentence("次の話を詳しく説明します", 2.0, 3.0),
    ) is None
    assert detector.detect(
        0,
        0,
        sentence,
        _sentence("はい", 2.0, 2.4),
    ) is not None


def test_unmarked_negative_polite_confirmation_rejects_topic_restart() -> None:
    detector = _detector()
    sentence = _sentence("できるかもしれないじゃないですか", 1.0, 2.0)

    assert detector.detect(
        0,
        0,
        sentence,
        _sentence("次の話は明日始まります", 2.1, 3.2),
    ) is None
    candidate = detector.detect(
        0,
        0,
        sentence,
        _sentence("はい", 2.1, 2.5),
    )
    assert candidate is not None
    assert "adjacent_independent_response" in candidate.evidence


def test_real_context_listener_question_regressions() -> None:
    detector = _detector()
    accepted = (
        (
            "国に帰るんで家具とか電気製品いらなくなるんだけどどうしたらいいかな",
            "僕の場合はアルバイト先の友達に譲るんだ",
            "following_complete_answer",
        ),
        (
            "公衆電話ってどこにあるんだろう",
            "使ったことないから分かんないな",
            "following_complete_answer",
        ),
        (
            "よろしいですか",
            "はい分かりました",
            "adjacent_independent_response",
        ),
        (
            "すいません何ですか",
            "グアテマラ南東部のオオアリクイ",
            "following_entity_answer",
        ),
        (
            "どういうことですか",
            "新しい生命ではなくてもう一度同じ人生をやり直すことでしたら可能です",
            "following_complete_answer",
        ),
    )

    for question, answer, expected_evidence in accepted:
        candidate = detector.detect(
            0,
            0,
            _sentence(question, 1.0, 2.0),
            _sentence(answer, 2.1, 3.2),
        )
        assert candidate is not None
        assert expected_evidence in candidate.evidence


def test_real_context_non_question_regressions() -> None:
    detector = _detector()
    rejected = (
        (
            "なんとか直りそうなんですが修理までに15日間半日かかると今日言われましてどうしようかな",
            "次の話を続けます",
        ),
        (
            "どうしようちょっと歩きながら話そうかな",
            "歩き始めます",
        ),
        (
            "今度は先生も一緒に遊ぼうかな",
            "はいそうしましょう",
        ),
        (
            "でもアリって小さいけど主食になるのかな",
            "引き続き精進しようと心に誓った",
        ),
        (
            "自分がやりたいことが出てくるかもしれないじゃないですか",
            "次は日本の生活について話します",
        ),
        ("すいません何ですか", "次の話"),
        ("すいません何ですか", "明日の予定"),
    )

    for text, following in rejected:
        assert detector.detect(
            0,
            0,
            _sentence(text, 1.0, 2.0),
            _sentence(following, 2.1, 3.2),
        ) is None


def test_explicit_question_accepts_auxiliary_volitional_form() -> None:
    candidate = _detector().detect(
        0,
        0,
        _sentence("最後の部分はどうでしょうか", 1.0, 2.0, is_question=True),
        None,
    )

    assert candidate is not None


def test_does_not_generate_for_embedded_question_or_existing_punctuation() -> None:
    detector = _detector()
    following = _sentence("答えます", 3.2, 4.0)

    assert detector.detect(
        0, 0, _sentence("何をするか考える", 1.0, 3.0), following
    ) is None
    assert detector.detect(
        0, 0, _sentence("何を?", 1.0, 2.0, is_question=True), following
    ) is None


def test_short_elliptical_candidate_requires_following_response() -> None:
    assert _detector().detect(
        0,
        0,
        _sentence("何を", 1.0, 1.6),
        None,
    ) is None


def test_stage_outputs_candidates_without_modifying_document() -> None:
    sentences = (
        _sentence("何を", 1.0, 1.6),
        _sentence("今回は行いません", 1.8, 3.0),
    )
    segment = Segment(0, "".join(x.text for x in sentences), TimeRange(1.0, 3.0), sentences)
    context = PipelineContext(
        "run-001",
        Document(Path("audio.mp3"), (segment,)),
        Path("work"),
    )

    result = QuestionPunctuationCandidateStage(_detector()).run(context)

    assert result.context is context
    assert tuple(item.text for item in result.data["candidates"]) == ("何を",)


def test_detects_embedded_permission_question_before_quotative_predicate() -> None:
    sentence = Sentence(
        "友達の家でトイレ借りていいって聞くのに似ています",
        TimeRange(1.0, 6.0),
        (
            Word("友達の家で", TimeRange(1.0, 2.0), 0.9),
            Word("トイレ借りていい", TimeRange(2.0, 3.5), 0.9),
            Word("って聞くのに似ています", TimeRange(3.5, 6.0), 0.9),
        ),
    )

    candidates = _detector().detect_all(2, 0, sentence, None)

    assert len(candidates) == 1
    assert candidates[0].text == "トイレ借りていい"
    assert candidates[0].time_range == TimeRange(2.0, 3.5)
    assert candidates[0].evidence[0] == "embedded_quoted_question"


def test_embedded_question_scope_keeps_location_argument() -> None:
    sentence = Sentence(
        "ここで話していいって聞きました",
        TimeRange(1.0, 4.0),
        (
            Word("ここで", TimeRange(1.0, 2.0), 0.9),
            Word("話していい", TimeRange(2.0, 3.0), 0.9),
            Word("って聞きました", TimeRange(3.0, 4.0), 0.9),
        ),
    )

    candidates = _detector().detect_all(0, 0, sentence, None)

    assert len(candidates) == 1
    assert candidates[0].text == "ここで話していい"
    assert candidates[0].time_range == TimeRange(1.0, 3.0)
