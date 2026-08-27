"""Constrained homophone semantic resolution adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol
import unicodedata

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_HOMOPHONE_CONFIDENCE_POLICY_CONFIG,
    DEFAULT_HOMOPHONE_PREFILTER_CONFIG,
)
from jp_learning_platform.workflow.homophone_stage import (
    HomophoneCandidateGenerationAudit,
    HomophoneCandidateScore,
    HomophoneResolution,
    HomophoneResolutionDecision,
    HomophoneResolutionRequest,
    HomophoneShadowCandidate,
    HomophoneShadowContextVerification,
    HomophoneShadowRankedCandidate,
)

DEFAULT_HOMOPHONE_MODEL_ID = "tohoku-nlp/bert-base-japanese-v3"
DEFAULT_HOMOPHONE_TOP_K = 80
DEFAULT_HOMOPHONE_SCORE_MARGIN = 0.0
DEFAULT_HOMOPHONE_MIN_CANDIDATE_SCORE = 0.0001
DEFAULT_HOMOPHONE_MIN_SCORE_RATIO = 20.0
DEFAULT_HOMOPHONE_MAX_ASR_CONFIDENCE = (
    DEFAULT_HOMOPHONE_CONFIDENCE_POLICY_CONFIG.high_asr_confidence
)
DEFAULT_HOMOPHONE_HIGH_CONFIDENCE_MIN_SCORE_RATIO = (
    DEFAULT_HOMOPHONE_CONFIDENCE_POLICY_CONFIG.high_confidence_min_score_ratio
)
DEFAULT_HOMOPHONE_MIN_TOKEN_CHARS = 2
DEFAULT_HOMOPHONE_MAX_CANDIDATE_PIECES = 3
DEFAULT_HOMOPHONE_MAX_LEXICAL_CANDIDATES = 64
DEFAULT_HOMOPHONE_MAX_TARGETS_PER_SENTENCE = (
    DEFAULT_HOMOPHONE_PREFILTER_CONFIG.max_targets_per_sentence
)
_MIN_CONTEXT_SCORE_DENOMINATOR = 1e-12
_DEFAULT_SUDACHI_SPLIT_MODE = "C"
_GENERATION_AUDIT_MAX_CONFIDENCE = 0.8
_SHADOW_CONTEXT_TOKENS_PER_SIDE = 4
_SHADOW_CONTEXT_VERIFICATION_LIMIT = 5
_CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞"}
_SKIPPED_SURFACES = {"する", "した", "して", "ある", "いる", "ます", "です"}
_PLACE_NAME_CONTINUATIONS = (
    "駅",
    "湖",
    "山",
    "川",
    "市",
    "町",
    "村",
    "区",
    "県",
    "府",
    "都",
    "線",
)
_DOCUMENT_PROPAGATION_REASONS = {
    "asr_confidence_too_high",
    "high_asr_confidence_requires_stronger_context",
    "candidate_score_ratio_too_low",
}


class HomophoneResolverDependencyError(RuntimeError):
    """Raised when optional homophone resolver dependencies are unavailable."""

    def __init__(self) -> None:
        super().__init__(
            "SudachiPy, sudachidict-core, transformers, torch, and the "
            "configured masked language model files are required for homophone "
            "semantic resolution. Install them with: "
            "python -m pip install -e '.[homophone]'"
        )


@dataclass(frozen=True, slots=True)
class HomophoneLanguageModelCandidate:
    """One masked-language-model candidate before same-reading filtering."""

    text: str
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")

        normalized_text = self.text.strip()
        if not normalized_text:
            raise ValueError("text must not be empty.")

        if isinstance(self.score, bool):
            raise TypeError("score must be a number.")

        object.__setattr__(self, "text", normalized_text)
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True, slots=True)
class HomophoneTarget:
    """A source token that may be replaceable by a same-reading candidate."""

    text: str
    reading: str
    part_of_speech: tuple[str, ...]
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class HomophoneReplacementScoringRequest:
    """One sentence target and its shadow replacements for batch scoring."""

    sentence_text: str
    target: HomophoneTarget
    replacements: tuple[str, ...]


class HomophoneCandidateGenerator(Protocol):
    """Candidate generator contract used by the resolver."""

    def candidates_for(
        self,
        sentence_text: str,
        target: HomophoneTarget,
    ) -> tuple[HomophoneLanguageModelCandidate, ...]:
        """Return language-model candidates for a masked target token."""

    def score_for(
        self,
        sentence_text: str,
        target: HomophoneTarget,
        replacement_text: str,
    ) -> float | None:
        """Score a concrete replacement in the same masked context."""


class HomophonePrefilterCandidateGenerator(HomophoneCandidateGenerator, Protocol):
    """Optional efficient target-prefilter capabilities."""

    def lexical_candidates_for(
        self,
        target: HomophoneTarget,
    ) -> tuple[str, ...]:
        """Return same-reading vocabulary candidates without model inference."""

    def original_scores_for(
        self,
        sentence_text: str,
        targets: tuple[HomophoneTarget, ...],
    ) -> tuple[float | None, ...]:
        """Score original targets in one contextual model batch."""

    def vocabulary_rank_for(self, text: str) -> float:
        """Return a normalized tokenizer vocabulary-rank frequency proxy."""

    def inflected_lexical_candidates_for(
        self,
        target: HomophoneTarget,
    ) -> tuple[str, ...]:
        """Return same-reading inflected candidates without model inference."""

    def scores_for_replacements(
        self,
        requests: tuple[HomophoneReplacementScoringRequest, ...],
    ) -> tuple[tuple[float | None, ...], ...]:
        """Batch-score original surfaces and shadow replacements."""


@dataclass(frozen=True, slots=True)
class _AnalyzedMorpheme:
    surface: str
    reading: str
    part_of_speech: tuple[str, ...]
    start: int
    end: int
    dictionary_form: str = ""


@dataclass(frozen=True, slots=True)
class _AcceptedChange:
    start: int
    end: int
    original_text: str
    selected_text: str


@dataclass(frozen=True, slots=True)
class _VocabularyPiece:
    surface: str
    reading: str
    part_of_speech: tuple[str, ...] = ()
    dictionary_form: str = ""


@dataclass(frozen=True, slots=True)
class _PrefilteredTarget:
    morpheme: _AnalyzedMorpheme
    target: HomophoneTarget
    lexical_candidate_count: int
    asr_confidence: float | None
    original_score: float | None
    vocabulary_rank: float


@dataclass(slots=True)
class SudachiReadingAnalyzer:
    """Analyze Japanese surfaces with Sudachi readings and POS metadata."""

    split_mode: str = _DEFAULT_SUDACHI_SPLIT_MODE
    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _mode: Any | None = field(default=None, init=False, repr=False)

    def analyze(self, text: str) -> tuple[_AnalyzedMorpheme, ...]:
        tokenizer, mode = self._load_tokenizer()
        cursor = 0
        morphemes: list[_AnalyzedMorpheme] = []
        for morpheme in tokenizer.tokenize(text, mode):
            surface = str(morpheme.surface())
            if not surface:
                continue

            start = text.find(surface, cursor)
            if start < 0:
                start = cursor
            end = start + len(surface)
            cursor = end
            reading = _normalize_reading(str(morpheme.reading_form()))
            if not reading:
                continue

            morphemes.append(
                _AnalyzedMorpheme(
                    surface=surface,
                    reading=reading,
                    part_of_speech=tuple(morpheme.part_of_speech()),
                    start=start,
                    end=end,
                    dictionary_form=str(morpheme.dictionary_form()),
                )
            )

        return tuple(morphemes)

    def analyze_single_token(self, text: str) -> _AnalyzedMorpheme | None:
        morphemes = self.analyze(text)
        if len(morphemes) != 1:
            return None

        morpheme = morphemes[0]
        if morpheme.surface != text:
            return None

        return morpheme

    def _load_tokenizer(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._mode is None:
            try:
                from sudachipy import dictionary
                from sudachipy import tokenizer
            except ImportError as error:
                raise HomophoneResolverDependencyError() from error

            mode_name = self.split_mode.strip().upper()
            try:
                self._mode = getattr(tokenizer.Tokenizer.SplitMode, mode_name)
            except AttributeError as error:
                raise ValueError(f"Unknown Sudachi split mode: {self.split_mode}") from error

            try:
                self._tokenizer = dictionary.Dictionary().create()
            except Exception as error:
                raise HomophoneResolverDependencyError() from error

        return self._tokenizer, self._mode


@dataclass(slots=True)
class BertMaskedLanguageHomophoneCandidateGenerator:
    """Generate and score candidates with a Japanese masked language model."""

    model_id: str = DEFAULT_HOMOPHONE_MODEL_ID
    device: str = "cpu"
    top_k: int = DEFAULT_HOMOPHONE_TOP_K
    analyzer: SudachiReadingAnalyzer | None = None
    max_candidate_pieces: int = DEFAULT_HOMOPHONE_MAX_CANDIDATE_PIECES
    max_lexical_candidates: int = DEFAULT_HOMOPHONE_MAX_LEXICAL_CANDIDATES
    _tokenizer: Any | None = field(default=None, init=False, repr=False)
    _model: Any | None = field(default=None, init=False, repr=False)
    _torch: Any | None = field(default=None, init=False, repr=False)
    _vocabulary_pieces: tuple[_VocabularyPiece, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _pieces_by_reading_initial: dict[str, tuple[_VocabularyPiece, ...]] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _inflected_candidates_by_suffix: dict[
        str,
        tuple[_AnalyzedMorpheme, ...],
    ] = field(default_factory=dict, init=False, repr=False)
    _lemma_pieces_by_inflection_class: dict[
        tuple[str, str, str],
        tuple[_VocabularyPiece, ...],
    ] | None = field(default=None, init=False, repr=False)
    _lemma_inflected_candidates: dict[
        tuple[str, str, str, str, str],
        tuple[_AnalyzedMorpheme, ...],
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise TypeError("top_k must be an integer.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.analyzer is None:
            self.analyzer = SudachiReadingAnalyzer()
        if isinstance(self.max_candidate_pieces, bool) or not isinstance(
            self.max_candidate_pieces,
            int,
        ):
            raise TypeError("max_candidate_pieces must be an integer.")
        if self.max_candidate_pieces <= 0:
            raise ValueError("max_candidate_pieces must be positive.")
        if isinstance(self.max_lexical_candidates, bool) or not isinstance(
            self.max_lexical_candidates,
            int,
        ):
            raise TypeError("max_lexical_candidates must be an integer.")
        if self.max_lexical_candidates <= 0:
            raise ValueError("max_lexical_candidates must be positive.")

    def candidates_for(
        self,
        sentence_text: str,
        target: HomophoneTarget,
    ) -> tuple[HomophoneLanguageModelCandidate, ...]:
        tokenizer, _, torch = self._load_model()
        logits = self._masked_logits(sentence_text, target)
        probabilities = torch.softmax(logits, dim=-1)
        limit = min(self.top_k, int(probabilities.shape[-1]))
        scores, token_ids = torch.topk(probabilities, k=limit)

        candidates: list[HomophoneLanguageModelCandidate] = []
        seen: set[str] = set()
        for token_id, score in zip(token_ids.tolist(), scores.tolist(), strict=True):
            text = self._token_text(tokenizer, token_id)
            if not text or text in seen:
                continue

            seen.add(text)
            candidates.append(
                HomophoneLanguageModelCandidate(
                    text=text,
                    score=float(score),
                )
            )

        for text in self._same_reading_vocabulary_candidates(target):
            if text in seen:
                continue

            score = self.score_for(sentence_text, target, text)
            if score is None:
                continue

            seen.add(text)
            candidates.append(
                HomophoneLanguageModelCandidate(
                    text=text,
                    score=score,
                )
            )

        return tuple(candidates)

    def score_for(
        self,
        sentence_text: str,
        target: HomophoneTarget,
        replacement_text: str,
    ) -> float | None:
        tokenizer, _, torch = self._load_model()
        token_ids = tokenizer.encode(replacement_text, add_special_tokens=False)
        if not token_ids:
            return None

        logits = self._masked_logits(
            sentence_text,
            target,
            mask_count=len(token_ids),
        )
        probabilities = torch.softmax(logits, dim=-1)
        if len(token_ids) == 1:
            return float(probabilities[token_ids[0]].item())

        log_probability = torch.tensor(0.0, device=probabilities.device)
        for index, token_id in enumerate(token_ids):
            probability = probabilities[index, token_id].clamp_min(1e-12)
            log_probability = log_probability + torch.log(probability)

        return float(torch.exp(log_probability / len(token_ids)).item())

    def lexical_candidates_for(
        self,
        target: HomophoneTarget,
    ) -> tuple[str, ...]:
        """Return same-reading candidates without a masked-model forward pass."""
        return self._same_reading_vocabulary_candidates(target)

    def original_scores_for(
        self,
        sentence_text: str,
        targets: tuple[HomophoneTarget, ...],
    ) -> tuple[float | None, ...]:
        """Compute contextual original-token probabilities in one model batch."""
        if not targets:
            return ()

        tokenizer, model, torch = self._load_model()
        mask_token = tokenizer.mask_token
        if not mask_token:
            raise HomophoneResolverDependencyError()

        masked_texts: list[str] = []
        original_token_ids: list[tuple[int, ...]] = []
        for target in targets:
            token_ids = tuple(
                tokenizer.encode(target.text, add_special_tokens=False)
            )
            original_token_ids.append(token_ids)
            masks = "".join(mask_token for _ in token_ids)
            masked_texts.append(
                f"{sentence_text[:target.start]}{masks}{sentence_text[target.end:]}"
            )

        inputs = tokenizer(masked_texts, return_tensors="pt", padding=True)
        if self.device != "cpu":
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

        model.eval()
        with torch.no_grad():
            logits = model(**inputs).logits

        scores: list[float | None] = []
        for row_index, token_ids in enumerate(original_token_ids):
            if not token_ids:
                scores.append(None)
                continue
            mask_positions = (
                inputs["input_ids"][row_index] == tokenizer.mask_token_id
            ).nonzero(as_tuple=False)
            if len(mask_positions) != len(token_ids):
                scores.append(None)
                continue

            log_probability = torch.tensor(0.0, device=logits.device)
            for position, token_id in zip(
                mask_positions,
                token_ids,
                strict=True,
            ):
                token_logits = logits[row_index, int(position.item())]
                probability = torch.softmax(token_logits, dim=-1)[token_id]
                log_probability = log_probability + torch.log(
                    probability.clamp_min(1e-12)
                )
            scores.append(
                float(torch.exp(log_probability / len(token_ids)).item())
            )

        return tuple(scores)

    def scores_for_replacements(
        self,
        requests: tuple[HomophoneReplacementScoringRequest, ...],
    ) -> tuple[tuple[float | None, ...], ...]:
        """Cheaply pre-rank all shadow replacements."""
        if not requests:
            return ()

        tokenizer, model, torch = self._load_model()
        mask_token = tokenizer.mask_token
        if not mask_token:
            raise HomophoneResolverDependencyError()

        scores: list[list[float | None]] = [
            [None] * len(request.replacements) for request in requests
        ]
        grouped: dict[int, list[tuple[int, int, str, tuple[int, ...]]]] = {}
        for request_index, request in enumerate(requests):
            for replacement_index, replacement in enumerate(request.replacements):
                token_ids = tuple(
                    tokenizer.encode(replacement, add_special_tokens=False)
                )
                if not token_ids:
                    continue
                masks = "".join(mask_token for _ in token_ids)
                masked_text = (
                    f"{request.sentence_text[:request.target.start]}"
                    f"{masks}{request.sentence_text[request.target.end:]}"
                )
                grouped.setdefault(len(token_ids), []).append(
                    (request_index, replacement_index, masked_text, token_ids)
                )

        batch_size = 32
        model.eval()
        for token_count, rows in grouped.items():
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset : offset + batch_size]
                inputs = tokenizer(
                    [row[2] for row in batch],
                    return_tensors="pt",
                    padding=True,
                )
                if self.device != "cpu":
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits
                for row_index, (
                    request_index,
                    replacement_index,
                    _,
                    token_ids,
                ) in enumerate(batch):
                    mask_positions = (
                        inputs["input_ids"][row_index] == tokenizer.mask_token_id
                    ).nonzero(as_tuple=False)
                    if len(mask_positions) != token_count:
                        continue
                    log_probability = torch.tensor(0.0, device=logits.device)
                    for position, token_id in zip(
                        mask_positions,
                        token_ids,
                        strict=True,
                    ):
                        probability = torch.softmax(
                            logits[row_index, int(position.item())],
                            dim=-1,
                        )[token_id]
                        log_probability += torch.log(probability.clamp_min(1e-12))
                    scores[request_index][replacement_index] = float(
                        torch.exp(log_probability / token_count).item()
                    )
        return tuple(tuple(values) for values in scores)

    def context_probe_scores_for_replacements(
        self,
        requests: tuple[HomophoneReplacementScoringRequest, ...],
    ) -> tuple[tuple[tuple[tuple[float, ...], tuple[float, ...]], ...], ...]:
        """Score fixed left and right context probes for shortlisted replacements."""
        tokenizer, model, torch = self._load_model()
        if tokenizer.mask_token_id is None:
            raise HomophoneResolverDependencyError()
        results: list[list[list[list[float]]]] = [
            [[[], []] for _ in request.replacements] for request in requests
        ]
        rows: list[tuple[int, int, int, list[int], int, int]] = []
        for request_index, request in enumerate(requests):
            prefix_ids = tokenizer.encode(
                request.sentence_text[: request.target.start],
                add_special_tokens=False,
            )
            suffix_ids = tokenizer.encode(
                request.sentence_text[request.target.end :],
                add_special_tokens=False,
            )
            left_indexes = range(
                max(0, len(prefix_ids) - _SHADOW_CONTEXT_TOKENS_PER_SIDE),
                len(prefix_ids),
            )
            right_indexes = range(
                min(len(suffix_ids), _SHADOW_CONTEXT_TOKENS_PER_SIDE)
            )
            for replacement_index, replacement in enumerate(request.replacements):
                replacement_ids = tokenizer.encode(
                    replacement,
                    add_special_tokens=False,
                )
                content_ids = [*prefix_ids, *replacement_ids, *suffix_ids]
                input_ids = tokenizer.build_inputs_with_special_tokens(content_ids)
                content_start = _subsequence_start(input_ids, content_ids)
                if content_start is None:
                    continue
                for prefix_index in left_indexes:
                    position = content_start + prefix_index
                    masked = list(input_ids)
                    masked[position] = tokenizer.mask_token_id
                    rows.append(
                        (
                            request_index,
                            replacement_index,
                            0,
                            masked,
                            position,
                            prefix_ids[prefix_index],
                        )
                    )
                suffix_start = content_start + len(prefix_ids) + len(replacement_ids)
                for suffix_index in right_indexes:
                    position = suffix_start + suffix_index
                    masked = list(input_ids)
                    masked[position] = tokenizer.mask_token_id
                    rows.append(
                        (
                            request_index,
                            replacement_index,
                            1,
                            masked,
                            position,
                            suffix_ids[suffix_index],
                        )
                    )
        model.eval()
        for offset in range(0, len(rows), 32):
            batch = rows[offset : offset + 32]
            inputs = tokenizer.pad(
                {"input_ids": [row[3] for row in batch]},
                padding=True,
                return_tensors="pt",
            )
            if self.device != "cpu":
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            for row_index, (
                request_index,
                replacement_index,
                side,
                _,
                position,
                token_id,
            ) in enumerate(batch):
                probability = torch.softmax(
                    logits[row_index, position],
                    dim=-1,
                )[token_id]
                results[request_index][replacement_index][side].append(
                    float(probability.item())
                )
        return tuple(
            tuple((tuple(sides[0]), tuple(sides[1])) for sides in replacements)
            for replacements in results
        )

    def token_counts_for_replacements(
        self,
        requests: tuple[HomophoneReplacementScoringRequest, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """Return tokenizer piece counts used by shadow score auditing."""
        tokenizer, _, _ = self._load_model()
        return tuple(
            tuple(
                len(tokenizer.encode(replacement, add_special_tokens=False))
                for replacement in request.replacements
            )
            for request in requests
        )

    def vocabulary_rank_for(self, text: str) -> float:
        """Use normalized token ids as a stable vocabulary-frequency proxy."""
        tokenizer, _, _ = self._load_model()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        vocabulary_size = max(len(tokenizer.get_vocab()), 1)
        if not token_ids:
            return 1.0
        return min(sum(token_ids) / len(token_ids) / vocabulary_size, 1.0)

    def inflected_lexical_candidates_for(
        self,
        target: HomophoneTarget,
    ) -> tuple[str, ...]:
        """Compose and validate vocabulary stems with the target inflection tail."""
        suffix = _trailing_hiragana(target.text)
        if not suffix:
            return ()

        tokenizer, _, _ = self._load_model()
        assert self.analyzer is not None
        if self._vocabulary_pieces is None:
            self._vocabulary_pieces = self._load_vocabulary_pieces(tokenizer)
        analyzed_candidates = self._inflected_candidates_by_suffix.get(suffix)
        if analyzed_candidates is None:
            generated: list[_AnalyzedMorpheme] = []
            seen: set[str] = set()
            for piece in self._vocabulary_pieces:
                if not _has_kanji(piece.surface) or _has_hiragana(piece.surface):
                    continue
                surface = f"{piece.surface}{suffix}"
                if surface in seen:
                    continue
                seen.add(surface)
                analyzed = self.analyzer.analyze_single_token(surface)
                if analyzed is not None:
                    generated.append(analyzed)
            analyzed_candidates = tuple(generated)
            self._inflected_candidates_by_suffix[suffix] = analyzed_candidates

        target_morpheme = self.analyzer.analyze_single_token(target.text)
        lemma_candidates: list[_AnalyzedMorpheme] = []
        if target_morpheme is not None and target_morpheme.dictionary_form:
            target_lemma = self.analyzer.analyze_single_token(
                target_morpheme.dictionary_form
            )
            if target_lemma is not None:
                inflection_class = _inflection_class(
                    target_morpheme.part_of_speech
                )
                cache_key = (
                    target_lemma.reading,
                    *inflection_class,
                    suffix,
                )
                cached = self._lemma_inflected_candidates.get(cache_key)
                if cached is None:
                    lemma_surfaces: set[str] = set()
                    generated_lemmas: list[_AnalyzedMorpheme] = []
                    for piece in self._lemma_pieces_for_inflection_class(
                        inflection_class
                    ):
                        if not _plausible_asr_reading_variant(
                            target_lemma.reading,
                            piece.reading,
                        ):
                            continue
                        surface = _inflected_surface_from_lemma(
                            target_morpheme.surface,
                            target_morpheme.dictionary_form,
                            piece.dictionary_form,
                        )
                        if not surface or surface in lemma_surfaces:
                            continue
                        lemma_surfaces.add(surface)
                        analyzed = self.analyzer.analyze_single_token(surface)
                        if (
                            analyzed is not None
                            and _plausible_asr_reading_variant(
                                target.reading,
                                analyzed.reading,
                            )
                        ):
                            generated_lemmas.append(analyzed)
                    cached = tuple(generated_lemmas)
                    self._lemma_inflected_candidates[cache_key] = cached
                lemma_candidates.extend(cached)

        candidates: list[str] = []
        seen_candidates: set[str] = set()
        lemma_candidate_surfaces = {
            analyzed.surface for analyzed in lemma_candidates
        }
        for analyzed in (*lemma_candidates, *analyzed_candidates):
            if analyzed.surface == target.text:
                continue
            if (
                analyzed.surface not in lemma_candidate_surfaces
                and analyzed.reading != target.reading
            ):
                continue
            if analyzed.surface in seen_candidates:
                continue
            if not _compatible_part_of_speech(
                target.part_of_speech,
                analyzed.part_of_speech,
            ):
                continue
            if not _compatible_script_change(target.text, analyzed.surface):
                continue
            seen_candidates.add(analyzed.surface)
            candidates.append(analyzed.surface)
            if len(candidates) >= self.max_lexical_candidates:
                break
        return tuple(candidates)

    def _lemma_pieces_for_inflection_class(
        self,
        inflection_class: tuple[str, str, str],
    ) -> tuple[_VocabularyPiece, ...]:
        if self._lemma_pieces_by_inflection_class is None:
            grouped: dict[tuple[str, str, str], list[_VocabularyPiece]] = {}
            assert self._vocabulary_pieces is not None
            for piece in self._vocabulary_pieces:
                if (
                    not piece.dictionary_form
                    or piece.surface != piece.dictionary_form
                ):
                    continue
                grouped.setdefault(
                    _inflection_class(piece.part_of_speech),
                    [],
                ).append(piece)
            self._lemma_pieces_by_inflection_class = {
                key: tuple(values) for key, values in grouped.items()
            }
        return self._lemma_pieces_by_inflection_class.get(inflection_class, ())

    def _masked_logits(
        self,
        sentence_text: str,
        target: HomophoneTarget,
        mask_count: int = 1,
    ) -> Any:
        tokenizer, model, torch = self._load_model()
        mask_token = tokenizer.mask_token
        if not mask_token:
            raise HomophoneResolverDependencyError()
        if mask_count <= 0:
            raise ValueError("mask_count must be positive.")

        masks = "".join(mask_token for _ in range(mask_count))
        masked_text = f"{sentence_text[:target.start]}{masks}{sentence_text[target.end:]}"
        inputs = tokenizer(masked_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=False)
        if len(mask_positions) != mask_count:
            raise RuntimeError("homophone language model mask count mismatch.")

        if self.device != "cpu":
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

        model.eval()
        with torch.no_grad():
            output = model(**inputs)

        mask_indexes = tuple(int(position.item()) for position in mask_positions)
        if mask_count == 1:
            return output.logits[0, mask_indexes[0]]

        return torch.stack(
            tuple(output.logits[0, mask_index] for mask_index in mask_indexes),
        )

    def _same_reading_vocabulary_candidates(
        self,
        target: HomophoneTarget,
    ) -> tuple[str, ...]:
        tokenizer, _, _ = self._load_model()
        assert self.analyzer is not None
        pieces_by_initial = self._vocabulary_pieces_by_initial(tokenizer)
        candidates: list[str] = []
        seen: set[str] = {target.text}

        # Prefer complete vocabulary entries with the same reading.  The old
        # depth-first composition could exhaust its candidate limit before a
        # common multi-kanji homophone was reached.
        if self._vocabulary_pieces is None:
            self._vocabulary_pieces = self._load_vocabulary_pieces(tokenizer)
        for piece in self._vocabulary_pieces:
            if piece.reading != target.reading or piece.surface in seen:
                continue
            seen.add(piece.surface)
            candidates.append(piece.surface)
            if len(candidates) >= self.max_lexical_candidates:
                return tuple(candidates)

        def visit(
            remaining_reading: str,
            surfaces: tuple[str, ...],
            piece_count: int,
        ) -> None:
            if len(candidates) >= self.max_lexical_candidates:
                return
            if not remaining_reading:
                surface = "".join(surfaces)
                if surface in seen:
                    return
                seen.add(surface)
                analyzed = self.analyzer.analyze_single_token(surface)
                if analyzed is None:
                    return
                if analyzed.reading != target.reading:
                    return
                if not _compatible_part_of_speech(
                    target.part_of_speech,
                    analyzed.part_of_speech,
                ):
                    return
                if not _compatible_script_change(target.text, analyzed.surface):
                    return
                if not tokenizer.encode(surface, add_special_tokens=False):
                    return
                candidates.append(surface)
                return

            if piece_count >= self.max_candidate_pieces:
                return

            for piece in pieces_by_initial.get(remaining_reading[0], ()):
                if not remaining_reading.startswith(piece.reading):
                    continue
                visit(
                    remaining_reading[len(piece.reading) :],
                    (*surfaces, piece.surface),
                    piece_count + 1,
                )

        visit(target.reading, (), 0)
        return tuple(candidates)

    def _vocabulary_pieces_by_initial(
        self,
        tokenizer: Any,
    ) -> dict[str, tuple[_VocabularyPiece, ...]]:
        if self._pieces_by_reading_initial is None:
            pieces = self._vocabulary_pieces
            if pieces is None:
                pieces = self._load_vocabulary_pieces(tokenizer)
            grouped: dict[str, list[_VocabularyPiece]] = {}
            for piece in pieces:
                grouped.setdefault(piece.reading[0], []).append(piece)

            self._vocabulary_pieces = pieces
            self._pieces_by_reading_initial = {
                initial: tuple(
                    sorted(values, key=lambda item: (len(item.reading), item.surface))
                )
                for initial, values in grouped.items()
            }

        return self._pieces_by_reading_initial

    def _load_vocabulary_pieces(self, tokenizer: Any) -> tuple[_VocabularyPiece, ...]:
        assert self.analyzer is not None
        pieces: list[_VocabularyPiece] = []
        seen: set[str] = set()
        for token in tokenizer.get_vocab():
            surface = str(token).replace("##", "").strip()
            surface = unicodedata.normalize("NFKC", surface).replace(" ", "")
            if not surface or surface in seen:
                continue
            if "[" in surface or "]" in surface:
                continue
            if not _has_japanese_text(surface):
                continue

            analyzed = self.analyzer.analyze_single_token(surface)
            if analyzed is None:
                continue

            seen.add(surface)
            pieces.append(
                _VocabularyPiece(
                    surface=surface,
                    reading=analyzed.reading,
                    part_of_speech=analyzed.part_of_speech,
                    dictionary_form=analyzed.dictionary_form,
                )
            )

        return tuple(pieces)

    def _token_text(self, tokenizer: Any, token_id: int) -> str:
        token = str(tokenizer.convert_ids_to_tokens(int(token_id))).strip()
        if not token or token in set(tokenizer.all_special_tokens):
            return ""

        text = token.replace("##", "").strip()
        text = unicodedata.normalize("NFKC", text).replace(" ", "")
        if not text or "[" in text or "]" in text:
            return ""

        return text

    def _load_model(self) -> tuple[Any, Any, Any]:
        if self._tokenizer is None or self._model is None or self._torch is None:
            try:
                import torch
                from transformers import AutoModelForMaskedLM, AutoTokenizer
            except ImportError as error:
                raise HomophoneResolverDependencyError() from error

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForMaskedLM.from_pretrained(self.model_id)
            except Exception as error:
                raise HomophoneResolverDependencyError() from error

            if self.device != "cpu":
                self._model.to(self.device)
            self._torch = torch

        return self._tokenizer, self._model, self._torch


@dataclass(slots=True)
class BertHomophoneResolver:
    """Replace only same-reading words that score better in sentence context."""

    candidate_generator: HomophoneCandidateGenerator | None = None
    analyzer: SudachiReadingAnalyzer | None = None
    model_id: str = DEFAULT_HOMOPHONE_MODEL_ID
    device: str = "cpu"
    top_k: int = DEFAULT_HOMOPHONE_TOP_K
    score_margin: float = DEFAULT_HOMOPHONE_SCORE_MARGIN
    min_candidate_score: float = DEFAULT_HOMOPHONE_MIN_CANDIDATE_SCORE
    min_score_ratio: float = DEFAULT_HOMOPHONE_MIN_SCORE_RATIO
    max_asr_confidence: float = DEFAULT_HOMOPHONE_MAX_ASR_CONFIDENCE
    high_confidence_min_score_ratio: float = (
        DEFAULT_HOMOPHONE_HIGH_CONFIDENCE_MIN_SCORE_RATIO
    )
    min_token_chars: int = DEFAULT_HOMOPHONE_MIN_TOKEN_CHARS
    max_targets_per_sentence: int = DEFAULT_HOMOPHONE_MAX_TARGETS_PER_SENTENCE
    require_original_score: bool = True
    enable_context_probe_audit: bool = False

    def __post_init__(self) -> None:
        if self.candidate_generator is None:
            self.candidate_generator = BertMaskedLanguageHomophoneCandidateGenerator(
                model_id=self.model_id,
                device=self.device,
                top_k=self.top_k,
            )
        if self.analyzer is None:
            self.analyzer = SudachiReadingAnalyzer()

        self.score_margin = float(self.score_margin)
        self.min_candidate_score = float(self.min_candidate_score)
        self.min_score_ratio = float(self.min_score_ratio)
        self.max_asr_confidence = float(self.max_asr_confidence)
        self.high_confidence_min_score_ratio = float(
            self.high_confidence_min_score_ratio
        )
        if self.min_candidate_score < 0:
            raise ValueError("min_candidate_score must be non-negative.")
        if self.min_score_ratio <= 1:
            raise ValueError("min_score_ratio must be greater than 1.0.")
        if not 0 <= self.max_asr_confidence <= 1:
            raise ValueError("max_asr_confidence must be between 0.0 and 1.0.")
        if self.high_confidence_min_score_ratio <= self.min_score_ratio:
            raise ValueError(
                "high_confidence_min_score_ratio must exceed min_score_ratio."
            )
        if isinstance(self.min_token_chars, bool) or not isinstance(
            self.min_token_chars,
            int,
        ):
            raise TypeError("min_token_chars must be an integer.")
        if self.min_token_chars < 1:
            raise ValueError("min_token_chars must be positive.")
        if isinstance(self.max_targets_per_sentence, bool) or not isinstance(
            self.max_targets_per_sentence,
            int,
        ):
            raise TypeError("max_targets_per_sentence must be an integer.")
        if self.max_targets_per_sentence < 1:
            raise ValueError("max_targets_per_sentence must be positive.")
        if not isinstance(self.enable_context_probe_audit, bool):
            raise TypeError("enable_context_probe_audit must be a bool.")

    def resolve(self, request: HomophoneResolutionRequest) -> HomophoneResolution:
        if not isinstance(request, HomophoneResolutionRequest):
            raise TypeError("request must be a HomophoneResolutionRequest.")

        resolved_segments: list[Segment] = []
        decisions: list[HomophoneResolutionDecision] = []
        generation_audits: list[HomophoneCandidateGenerationAudit] = []
        shadow_candidates: list[HomophoneShadowCandidate] = []
        for segment in request.segments:
            shadow_candidates.extend(self._shadow_candidates_for_segment(segment))
            resolved_segment, segment_decisions, segment_audits = (
                self._resolve_segment(segment)
            )
            resolved_segments.append(resolved_segment)
            decisions.extend(segment_decisions)
            generation_audits.extend(segment_audits)

        shadow_candidates = list(
            self._score_shadow_candidates(
                request.segments,
                tuple(shadow_candidates),
            )
        )

        confirmed = _unambiguous_confirmed_replacements(
            tuple(decisions),
            min_score_ratio=self.high_confidence_min_score_ratio,
        )
        propagated_decisions = tuple(
            self._propagate_decision(decision, confirmed)
            for decision in decisions
        )
        if propagated_decisions != tuple(decisions):
            resolved_segments = list(
                _apply_accepted_decisions(
                    request.segments,
                    propagated_decisions,
                )
            )

        return HomophoneResolution(
            source_path=request.source_path,
            segments=tuple(resolved_segments),
            decisions=propagated_decisions,
            candidate_generation_audits=tuple(generation_audits),
            shadow_candidates=tuple(shadow_candidates),
        )

    def _score_shadow_candidates(
        self,
        segments: tuple[Segment, ...],
        shadows: tuple[HomophoneShadowCandidate, ...],
    ) -> tuple[HomophoneShadowCandidate, ...]:
        assert self.candidate_generator is not None
        sentence_texts: dict[tuple[int, int], str] = {}
        for segment in segments:
            sentences = segment.sentences or (
                Sentence(segment.text, segment.time_range, ()),
            )
            for sentence_index, sentence in enumerate(sentences):
                sentence_texts[(segment.position, sentence_index)] = sentence.text

        scoreable_indexes: list[int] = []
        requests: list[HomophoneReplacementScoringRequest] = []
        for index, shadow in enumerate(shadows):
            if not shadow.candidates:
                continue
            sentence_text = sentence_texts.get(
                (shadow.segment_position, shadow.sentence_index)
            )
            if sentence_text is None:
                continue
            scoreable_indexes.append(index)
            requests.append(
                HomophoneReplacementScoringRequest(
                    sentence_text=sentence_text,
                    target=HomophoneTarget(
                        text=shadow.surface,
                        reading=shadow.reading,
                        part_of_speech=shadow.part_of_speech,
                        start=shadow.target_start,
                        end=shadow.target_end,
                    ),
                    replacements=(shadow.surface, *shadow.candidates),
                )
            )
        if not requests:
            return shadows

        batch_score = getattr(
            self.candidate_generator,
            "scores_for_replacements",
            None,
        )
        if callable(batch_score):
            score_rows = tuple(batch_score(tuple(requests)))
        else:
            score_rows = tuple(
                tuple(
                    self.candidate_generator.score_for(
                        request.sentence_text,
                        request.target,
                        replacement,
                    )
                    for replacement in request.replacements
                )
                for request in requests
            )
        token_count_lookup = getattr(
            self.candidate_generator,
            "token_counts_for_replacements",
            None,
        )
        if callable(token_count_lookup):
            token_count_rows = tuple(token_count_lookup(tuple(requests)))
        else:
            token_count_rows = tuple(
                tuple(None for _ in request.replacements)
                for request in requests
            )
        context_probe_lookup = (
            getattr(
                self.candidate_generator,
                "context_probe_scores_for_replacements",
                None,
            )
            if self.enable_context_probe_audit
            else None
        )

        scored = list(shadows)
        for shadow_index, request, scores, token_counts in zip(
            scoreable_indexes,
            requests,
            score_rows,
            token_count_rows,
            strict=True,
        ):
            shadow = shadows[shadow_index]
            original_score = scores[0] if scores else None
            candidate_scores = tuple(
                HomophoneCandidateScore(
                    text=text,
                    reading=shadow.reading,
                    score=score,
                )
                for text, score in zip(
                    request.replacements[1:],
                    scores[1:],
                    strict=True,
                )
            )
            ranked_candidates = _rank_shadow_candidates(
                candidate_scores,
                token_counts=token_counts[1:],
            )
            top_score = (
                ranked_candidates[0].score if ranked_candidates else None
            )
            if callable(context_probe_lookup):
                acceptance_status = "pending_context_verification"
                acceptance_reason = "shortlisted_by_cheap_score"
            else:
                acceptance_status = "not_evaluated"
                acceptance_reason = "context_probe_audit_disabled"
            scored[shadow_index] = replace(
                shadow,
                original_score=original_score,
                candidate_scores=candidate_scores,
                ranked_candidates=ranked_candidates,
                top_candidate=(
                    ranked_candidates[0].text if ranked_candidates else None
                ),
                top_score_margin=_top_score_margin(ranked_candidates),
                top_score_ratio_vs_original=_score_ratio(
                    top_score,
                    original_score,
                ),
                score_method=(
                    "cheap_prerank_with_context_probe_audit"
                    if callable(context_probe_lookup)
                    else "cheap_prerank_only"
                ),
                original_token_count=(token_counts[0] if token_counts else None),
                top_candidate_token_count=(
                    ranked_candidates[0].token_count
                    if ranked_candidates
                    else None
                ),
                relative_acceptance_status=acceptance_status,
                relative_acceptance_reason=acceptance_reason,
            )
        if not callable(context_probe_lookup):
            return tuple(scored)

        verification_indexes: list[int] = []
        verification_requests: list[HomophoneReplacementScoringRequest] = []
        for shadow_index in scoreable_indexes:
            shadow = scored[shadow_index]
            shortlisted = tuple(
                candidate.text
                for candidate in shadow.ranked_candidates[
                    :_SHADOW_CONTEXT_VERIFICATION_LIMIT
                ]
            )
            if not shortlisted:
                continue
            sentence_text = sentence_texts[
                (shadow.segment_position, shadow.sentence_index)
            ]
            verification_indexes.append(shadow_index)
            verification_requests.append(
                HomophoneReplacementScoringRequest(
                    sentence_text=sentence_text,
                    target=HomophoneTarget(
                        text=shadow.surface,
                        reading=shadow.reading,
                        part_of_speech=shadow.part_of_speech,
                        start=shadow.target_start,
                        end=shadow.target_end,
                    ),
                    replacements=(shadow.surface, *shortlisted),
                )
            )
        probe_rows = tuple(context_probe_lookup(tuple(verification_requests)))
        for shadow_index, request, replacement_probes in zip(
            verification_indexes,
            verification_requests,
            probe_rows,
            strict=True,
        ):
            original_probes = replacement_probes[0]
            verifications = tuple(
                _context_probe_verification(text, original_probes, probes)
                for text, probes in zip(
                    request.replacements[1:],
                    replacement_probes[1:],
                    strict=True,
                )
            )
            scored[shadow_index] = replace(
                scored[shadow_index],
                relative_acceptance_status="audit_only",
                relative_acceptance_reason="context_probes_are_non_decisive",
                accepted_candidate=None,
                context_verifications=verifications,
            )
        return tuple(scored)

    def _shadow_candidates_for_segment(
        self,
        segment: Segment,
    ) -> tuple[HomophoneShadowCandidate, ...]:
        assert self.analyzer is not None
        assert self.candidate_generator is not None
        lexical_lookup = getattr(
            self.candidate_generator,
            "lexical_candidates_for",
            None,
        )
        inflected_lookup = getattr(
            self.candidate_generator,
            "inflected_lexical_candidates_for",
            None,
        )
        if not callable(lexical_lookup):
            return ()

        sentences = segment.sentences or (
            Sentence(segment.text, segment.time_range, ()),
        )
        shadows: list[HomophoneShadowCandidate] = []
        for sentence_index, sentence in enumerate(sentences):
            morphemes = self.analyzer.analyze(sentence.text)
            for morpheme in morphemes:
                if _is_general_single_character_noun(morpheme):
                    target = _homophone_target(morpheme)
                    shadows.append(
                        _shadow_candidate(
                            self.analyzer,
                            segment.position,
                            sentence_index,
                            "single_character",
                            target,
                            tuple(lexical_lookup(target)),
                            (morpheme.surface,),
                        )
                    )
                if (
                    callable(inflected_lookup)
                    and _is_inflected_content_morpheme(morpheme)
                    and _has_kanji(morpheme.surface)
                    and bool(_trailing_hiragana(morpheme.surface))
                ):
                    target = _homophone_target(morpheme)
                    shadows.append(
                        _shadow_candidate(
                            self.analyzer,
                            segment.position,
                            sentence_index,
                            "inflected",
                            target,
                            tuple(inflected_lookup(target)),
                            (morpheme.surface,),
                        )
                    )
            for left, right in zip(morphemes, morphemes[1:]):
                if not (
                    _is_content_morpheme(left)
                    and _is_content_morpheme(right)
                    and len(left.surface) < self.min_token_chars
                    and len(right.surface) < self.min_token_chars
                ):
                    continue
                target = HomophoneTarget(
                    text=left.surface + right.surface,
                    reading=left.reading + right.reading,
                    part_of_speech=right.part_of_speech,
                    start=left.start,
                    end=right.end,
                )
                shadows.append(
                    _shadow_candidate(
                        self.analyzer,
                        segment.position,
                        sentence_index,
                        "cross_morpheme",
                        target,
                        tuple(lexical_lookup(target)),
                        (left.surface, right.surface),
                    )
                )
        return tuple(shadows)

    def _propagate_decision(
        self,
        decision: HomophoneResolutionDecision,
        confirmed: dict[str, str],
    ) -> HomophoneResolutionDecision:
        selected_text = confirmed.get(decision.original_text)
        if (
            selected_text is None
            or decision.reason not in _DOCUMENT_PROPAGATION_REASONS
        ):
            return decision
        selected = next(
            (
                candidate
                for candidate in decision.candidates
                if candidate.text == selected_text
            ),
            None,
        )
        if selected is None or selected.score is None:
            return decision
        if selected.score < self.min_candidate_score:
            return decision
        if decision.original_score is None:
            return decision
        if selected.score <= decision.original_score + self.score_margin:
            return decision
        return replace(
            decision,
            selected_text=selected_text,
            accepted=True,
            reason="accepted_document_consistency",
            selected_score=selected.score,
        )

    def _resolve_segment(
        self,
        segment: Segment,
    ) -> tuple[
        Segment,
        tuple[HomophoneResolutionDecision, ...],
        tuple[HomophoneCandidateGenerationAudit, ...],
    ]:
        sentences = segment.sentences or (
            Sentence(
                text=segment.text,
                time_range=segment.time_range,
                words=(),
            ),
        )

        resolved_sentences: list[Sentence] = []
        decisions: list[HomophoneResolutionDecision] = []
        generation_audits: list[HomophoneCandidateGenerationAudit] = []
        for sentence_index, sentence in enumerate(sentences):
            resolved_sentence, sentence_decisions, sentence_audits = self._resolve_sentence(
                segment.position,
                sentence_index,
                sentence,
            )
            resolved_sentences.append(resolved_sentence)
            decisions.extend(sentence_decisions)
            generation_audits.extend(sentence_audits)

        segment_text = "".join(sentence.text for sentence in resolved_sentences)
        start_seconds = min(
            segment.time_range.start_seconds,
            *(sentence.time_range.start_seconds for sentence in resolved_sentences),
        )
        end_seconds = max(
            segment.time_range.end_seconds,
            *(sentence.time_range.end_seconds for sentence in resolved_sentences),
        )
        return (
            Segment(
                position=segment.position,
                text=segment_text,
                time_range=TimeRange(start_seconds, end_seconds),
                sentences=tuple(resolved_sentences),
            ),
            tuple(decisions),
            tuple(generation_audits),
        )

    def _resolve_sentence(
        self,
        segment_position: int,
        sentence_index: int,
        sentence: Sentence,
    ) -> tuple[
        Sentence,
        tuple[HomophoneResolutionDecision, ...],
        tuple[HomophoneCandidateGenerationAudit, ...],
    ]:
        assert self.analyzer is not None
        morphemes = self.analyzer.analyze(sentence.text)
        selected_targets, original_scores, prefilter_audits = self._prefilter_targets(
            segment_position,
            sentence_index,
            sentence,
            morphemes,
        )
        generation_audits = list(prefilter_audits)
        decisions: list[HomophoneResolutionDecision] = []
        changes: list[_AcceptedChange] = []
        for morpheme in selected_targets:
            decision, target_audits = self._decision_for_target(
                segment_position=segment_position,
                sentence_index=sentence_index,
                sentence_text=sentence.text,
                sentence_words=sentence.words,
                morpheme=morpheme,
                prefetched_original_score=original_scores.get(
                    (morpheme.start, morpheme.end)
                ),
                has_prefetched_original_score=(
                    (morpheme.start, morpheme.end) in original_scores
                ),
            )
            generation_audits.extend(target_audits)
            if decision is None:
                continue

            decisions.append(decision)
            if decision.accepted and decision.selected_text != decision.original_text:
                changes.append(
                    _AcceptedChange(
                        start=morpheme.start,
                        end=morpheme.end,
                        original_text=decision.original_text,
                        selected_text=decision.selected_text,
                    )
                )

        if not changes:
            return sentence, tuple(decisions), tuple(generation_audits)

        text = _apply_text_changes(sentence.text, tuple(changes))
        words = _apply_word_changes(sentence.words, tuple(changes))
        return (
            Sentence(
                text=text,
                time_range=sentence.time_range,
                words=words,
                asr_boundary_word_indexes=sentence.asr_boundary_word_indexes,
            ),
            tuple(decisions),
            tuple(generation_audits),
        )

    def _prefilter_targets(
        self,
        segment_position: int,
        sentence_index: int,
        sentence: Sentence,
        morphemes: tuple[_AnalyzedMorpheme, ...],
    ) -> tuple[
        tuple[_AnalyzedMorpheme, ...],
        dict[tuple[int, int], float | None],
        tuple[HomophoneCandidateGenerationAudit, ...],
    ]:
        audits: list[HomophoneCandidateGenerationAudit] = []
        for morpheme in morphemes:
            if (
                _is_content_morpheme(morpheme)
                and len(morpheme.surface) < self.min_token_chars
                and (
                    _is_general_single_character_noun(morpheme)
                    or _should_record_generation_audit(
                        sentence.words,
                        morpheme.surface,
                        max_confidence=self.max_asr_confidence,
                    )
                )
            ):
                audits.append(
                    _generation_audit(
                        segment_position,
                        sentence_index,
                        morpheme,
                        "single_character_filtered",
                    )
                )
        for left, right in zip(morphemes, morphemes[1:]):
            if (
                _is_content_morpheme(left)
                and _is_content_morpheme(right)
                and len(left.surface) < self.min_token_chars
                and len(right.surface) < self.min_token_chars
            ):
                audits.append(
                    HomophoneCandidateGenerationAudit(
                        segment_position=segment_position,
                        sentence_index=sentence_index,
                        surface=left.surface + right.surface,
                        reading=left.reading + right.reading,
                        part_of_speech=left.part_of_speech,
                        target_start=left.start,
                        target_end=right.end,
                        reason="cross_morpheme_target_not_generated",
                        status="diagnostic_only",
                        morpheme_span=(left.surface, right.surface),
                    )
                )
        eligible = tuple(
            morpheme for morpheme in morphemes if self._should_consider(morpheme)
        )
        assert self.candidate_generator is not None
        lexical_lookup = getattr(
            self.candidate_generator,
            "lexical_candidates_for",
            None,
        )
        batch_score = getattr(
            self.candidate_generator,
            "original_scores_for",
            None,
        )
        vocabulary_rank = getattr(
            self.candidate_generator,
            "vocabulary_rank_for",
            None,
        )
        if not callable(lexical_lookup):
            return eligible[: self.max_targets_per_sentence], {}, tuple(audits)

        targets: list[HomophoneTarget] = []
        candidate_counts: list[int] = []
        filtered_morphemes: list[_AnalyzedMorpheme] = []
        for morpheme in eligible:
            target = HomophoneTarget(
                text=morpheme.surface,
                reading=morpheme.reading,
                part_of_speech=morpheme.part_of_speech,
                start=morpheme.start,
                end=morpheme.end,
            )
            candidates = tuple(lexical_lookup(target))
            if not candidates:
                if (
                    _is_inflected_content_morpheme(morpheme)
                    or _is_proper_noun(morpheme)
                    or _should_record_generation_audit(
                        sentence.words,
                        morpheme.surface,
                    )
                ):
                    reason = (
                        "inflected_candidate_not_generated"
                        if _is_inflected_content_morpheme(morpheme)
                        else "no_same_reading_lexical_candidate"
                    )
                    audits.append(
                        _generation_audit(
                            segment_position,
                            sentence_index,
                            morpheme,
                            reason,
                        )
                    )
                continue
            targets.append(target)
            candidate_counts.append(len(candidates))
            filtered_morphemes.append(morpheme)

        if not targets:
            return (), {}, tuple(audits)

        if callable(batch_score):
            scores = tuple(batch_score(sentence.text, tuple(targets)))
        else:
            scores = (None,) * len(targets)
        if len(scores) != len(targets):
            raise RuntimeError("homophone prefilter score count mismatch.")

        ranked: list[_PrefilteredTarget] = []
        for morpheme, target, count, original_score in zip(
            filtered_morphemes,
            targets,
            candidate_counts,
            scores,
            strict=True,
        ):
            rank = (
                float(vocabulary_rank(target.text))
                if callable(vocabulary_rank)
                else 0.0
            )
            ranked.append(
                _PrefilteredTarget(
                    morpheme=morpheme,
                    target=target,
                    lexical_candidate_count=count,
                    asr_confidence=_surface_confidence(
                        sentence.words,
                        target.text,
                    ),
                    original_score=original_score,
                    vocabulary_rank=rank,
                )
            )

        ranked.sort(key=_prefilter_sort_key)
        selected = tuple(ranked[: self.max_targets_per_sentence])
        for item in ranked[self.max_targets_per_sentence :]:
            audits.append(
                _generation_audit(
                    segment_position,
                    sentence_index,
                    item.morpheme,
                    "prefilter_target_limit",
                    candidate_count=item.lexical_candidate_count,
                )
            )
        return (
            tuple(item.morpheme for item in selected),
            {
                (item.morpheme.start, item.morpheme.end): item.original_score
                for item in selected
            },
            tuple(audits),
        )

    def _decision_for_target(
        self,
        *,
        segment_position: int,
        sentence_index: int,
        sentence_text: str,
        sentence_words: tuple[Word, ...],
        morpheme: _AnalyzedMorpheme,
        prefetched_original_score: float | None = None,
        has_prefetched_original_score: bool = False,
    ) -> tuple[
        HomophoneResolutionDecision | None,
        tuple[HomophoneCandidateGenerationAudit, ...],
    ]:
        if not self._should_consider(morpheme):
            return None, ()

        assert self.candidate_generator is not None
        assert self.analyzer is not None
        target = HomophoneTarget(
            text=morpheme.surface,
            reading=morpheme.reading,
            part_of_speech=morpheme.part_of_speech,
            start=morpheme.start,
            end=morpheme.end,
        )
        language_model_candidates = self.candidate_generator.candidates_for(
            sentence_text,
            target,
        )

        scored_candidates: list[HomophoneCandidateScore] = []
        candidate_part_of_speech: dict[str, tuple[str, ...]] = {}
        different_reading_candidates: list[str] = []
        for candidate in language_model_candidates:
            analyzed_candidate = self.analyzer.analyze_single_token(candidate.text)
            if analyzed_candidate is None:
                continue
            if analyzed_candidate.surface == morpheme.surface:
                continue
            if analyzed_candidate.reading != morpheme.reading:
                different_reading_candidates.append(analyzed_candidate.surface)
                continue
            if not _compatible_part_of_speech(
                morpheme.part_of_speech,
                analyzed_candidate.part_of_speech,
            ):
                continue
            if not _compatible_script_change(morpheme.surface, analyzed_candidate.surface):
                continue

            scored_candidates.append(
                HomophoneCandidateScore(
                    text=analyzed_candidate.surface,
                    reading=analyzed_candidate.reading,
                    score=candidate.score,
                )
            )
            candidate_part_of_speech[analyzed_candidate.surface] = (
                analyzed_candidate.part_of_speech
            )

        original_score = prefetched_original_score
        if not has_prefetched_original_score:
            original_score = self.candidate_generator.score_for(
                sentence_text,
                target,
                morpheme.surface,
            )
        if not scored_candidates:
            audits = ()
            if (
                different_reading_candidates
                and (
                    _is_inflected_content_morpheme(morpheme)
                    or _should_record_generation_audit(
                        sentence_words,
                        morpheme.surface,
                    )
                )
            ):
                reason = (
                    "inflected_candidate_not_generated"
                    if _is_inflected_content_morpheme(morpheme)
                    else "different_reading_candidate_filtered"
                )
                audits = (
                    _generation_audit(
                        segment_position,
                        sentence_index,
                        morpheme,
                        reason,
                        candidate_count=len(different_reading_candidates),
                        candidate_examples=tuple(different_reading_candidates[:5]),
                    ),
                )
            return (
                HomophoneResolutionDecision(
                    segment_position=segment_position,
                    sentence_index=sentence_index,
                    original_text=morpheme.surface,
                    selected_text=morpheme.surface,
                    reading=morpheme.reading,
                    accepted=False,
                    reason="no_same_reading_candidate",
                    original_score=original_score,
                    selected_score=None,
                    candidates=(),
                    target_start=morpheme.start,
                    target_end=morpheme.end,
                ),
                audits,
            )

        selected = max(
            scored_candidates,
            key=lambda candidate: candidate.score or 0.0,
        )
        selected_score = selected.score
        asr_confidence = _surface_confidence(
            sentence_words,
            morpheme.surface,
        )
        if _requires_external_person_name_evidence(
            sentence_text,
            morpheme,
            candidate_part_of_speech.get(selected.text, ()),
        ):
            accepted, reason = False, "person_name_requires_external_evidence"
        else:
            accepted, reason = self._accept_candidate(
                original_score=original_score,
                selected_score=selected_score,
                asr_confidence=asr_confidence,
            )
        return (
            HomophoneResolutionDecision(
                segment_position=segment_position,
                sentence_index=sentence_index,
                original_text=morpheme.surface,
                selected_text=selected.text if accepted else morpheme.surface,
                reading=morpheme.reading,
                accepted=accepted,
                reason=reason,
                original_score=original_score,
                selected_score=selected_score,
                candidates=tuple(scored_candidates),
                target_start=morpheme.start,
                target_end=morpheme.end,
                asr_confidence=asr_confidence,
                score_ratio=_score_ratio(selected_score, original_score),
            ),
            (),
        )

    def _should_consider(self, morpheme: _AnalyzedMorpheme) -> bool:
        if len(morpheme.surface) < self.min_token_chars:
            return False

        if morpheme.surface in _SKIPPED_SURFACES:
            return False

        if _pos(morpheme.part_of_speech, 0) not in _CONTENT_POS:
            return False

        if not _has_japanese_text(morpheme.surface):
            return False

        return True

    def _accept_candidate(
        self,
        *,
        original_score: float | None,
        selected_score: float | None,
        asr_confidence: float | None,
    ) -> tuple[bool, str]:
        if selected_score is None:
            return False, "missing_candidate_score"

        if selected_score < self.min_candidate_score:
            return False, "candidate_score_too_low"

        if asr_confidence is None:
            return False, "missing_asr_confidence"

        if original_score is None:
            if self.require_original_score:
                return False, "missing_original_score"
            return True, "accepted_same_reading_context"

        if selected_score <= original_score + self.score_margin:
            return False, "candidate_not_better_than_original"

        score_ratio = _score_ratio(selected_score, original_score)
        if score_ratio is not None and score_ratio < self.min_score_ratio:
            return False, "candidate_score_ratio_too_low"

        if (
            asr_confidence > self.max_asr_confidence
            and score_ratio is not None
            and score_ratio < self.high_confidence_min_score_ratio
        ):
            return False, "high_asr_confidence_requires_stronger_context"

        if asr_confidence > self.max_asr_confidence:
            return True, "accepted_high_asr_confidence_with_strong_context"

        return True, "accepted_same_reading_context"


def _score_ratio(
    selected_score: float | None,
    original_score: float | None,
) -> float | None:
    if selected_score is None or original_score is None:
        return None
    return selected_score / max(original_score, _MIN_CONTEXT_SCORE_DENOMINATOR)


def _rank_shadow_candidates(
    candidates: tuple[HomophoneCandidateScore, ...],
    *,
    token_counts: tuple[int | None, ...] = (),
) -> tuple[HomophoneShadowRankedCandidate, ...]:
    count_by_text = {
        candidate.text: token_count
        for candidate, token_count in zip(candidates, token_counts)
    }
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.score is None,
            -(candidate.score or 0.0),
            candidate.text,
        ),
    )
    return tuple(
        HomophoneShadowRankedCandidate(
            text=candidate.text,
            reading=candidate.reading,
            score=candidate.score,
            rank=index,
            token_count=count_by_text.get(candidate.text),
        )
        for index, candidate in enumerate(ordered, start=1)
    )


def _context_probe_verification(
    text: str,
    original: tuple[tuple[float, ...], tuple[float, ...]],
    candidate: tuple[tuple[float, ...], tuple[float, ...]],
) -> HomophoneShadowContextVerification:
    original_left, original_right = original
    candidate_left, candidate_right = candidate
    left_probes = min(len(original_left), len(candidate_left))
    right_probes = min(len(original_right), len(candidate_right))
    left_wins = sum(
        candidate_score > original_score
        for original_score, candidate_score in zip(
            original_left[:left_probes],
            candidate_left[:left_probes],
            strict=True,
        )
    )
    right_wins = sum(
        candidate_score > original_score
        for original_score, candidate_score in zip(
            original_right[:right_probes],
            candidate_right[:right_probes],
            strict=True,
        )
    )
    total_wins = left_wins + right_wins
    total_probes = left_probes + right_probes
    has_both_sides = left_probes > 0 and right_probes > 0
    left_majority = left_wins * 2 > left_probes
    right_majority = right_wins * 2 > right_probes
    total_majority = total_wins * 2 > total_probes
    bilateral_majority = (
        has_both_sides
        and left_majority
        and right_majority
        and total_majority
    )
    if not has_both_sides:
        reason = "missing_bilateral_context"
    elif not left_majority:
        reason = "left_probe_majority_missing"
    elif not right_majority:
        reason = "right_probe_majority_missing"
    elif not total_majority:
        reason = "overall_probe_majority_missing"
    else:
        reason = "bilateral_probe_majority"
    return HomophoneShadowContextVerification(
        text=text,
        left_wins=left_wins,
        left_probes=left_probes,
        right_wins=right_wins,
        right_probes=right_probes,
        total_wins=total_wins,
        total_probes=total_probes,
        bilateral_majority=bilateral_majority,
        reason=reason,
    )


def _top_score_margin(
    ranked: tuple[HomophoneShadowRankedCandidate, ...],
) -> float | None:
    if len(ranked) < 2:
        return None
    first = ranked[0].score
    second = ranked[1].score
    if first is None or second is None:
        return None
    return first - second


def _apply_text_changes(
    text: str,
    changes: tuple[_AcceptedChange, ...],
) -> str:
    updated = text
    for change in sorted(changes, key=lambda item: item.start, reverse=True):
        updated = f"{updated[:change.start]}{change.selected_text}{updated[change.end:]}"
    return updated


def _unambiguous_confirmed_replacements(
    decisions: tuple[HomophoneResolutionDecision, ...],
    *,
    min_score_ratio: float | None = None,
) -> dict[str, str]:
    candidates: dict[str, set[str]] = {}
    for decision in decisions:
        if not decision.accepted or decision.selected_text == decision.original_text:
            continue
        score_ratio = decision.score_ratio
        if score_ratio is None:
            score_ratio = _score_ratio(
                decision.selected_score,
                decision.original_score,
            )
        if (
            min_score_ratio is not None
            and (score_ratio is None or score_ratio < min_score_ratio)
        ):
            continue
        candidates.setdefault(decision.original_text, set()).add(
            decision.selected_text
        )
    return {
        original: next(iter(replacements))
        for original, replacements in candidates.items()
        if len(replacements) == 1
    }


def _is_person_name(part_of_speech: tuple[str, ...]) -> bool:
    return bool(
        len(part_of_speech) > 2
        and part_of_speech[0] == "名詞"
        and part_of_speech[1] == "固有名詞"
        and part_of_speech[2] == "人名"
    )


def _requires_external_person_name_evidence(
    sentence_text: str,
    morpheme: _AnalyzedMorpheme,
    candidate_part_of_speech: tuple[str, ...],
) -> bool:
    if not (
        _is_person_name(morpheme.part_of_speech)
        or _is_person_name(candidate_part_of_speech)
    ):
        return False
    following_text = sentence_text[morpheme.end :]
    return not following_text.startswith(_PLACE_NAME_CONTINUATIONS)


def _apply_accepted_decisions(
    segments: tuple[Segment, ...],
    decisions: tuple[HomophoneResolutionDecision, ...],
) -> tuple[Segment, ...]:
    decisions_by_sentence: dict[
        tuple[int, int],
        list[HomophoneResolutionDecision],
    ] = {}
    for decision in decisions:
        if not decision.accepted or decision.selected_text == decision.original_text:
            continue
        decisions_by_sentence.setdefault(
            (decision.segment_position, decision.sentence_index),
            [],
        ).append(decision)

    resolved_segments: list[Segment] = []
    for segment in segments:
        source_sentences = segment.sentences or (
            Sentence(
                text=segment.text,
                time_range=segment.time_range,
                words=(),
            ),
        )
        resolved_sentences: list[Sentence] = []
        for sentence_index, sentence in enumerate(source_sentences):
            changes = tuple(
                _AcceptedChange(
                    start=decision.target_start,
                    end=decision.target_end,
                    original_text=decision.original_text,
                    selected_text=decision.selected_text,
                )
                for decision in decisions_by_sentence.get(
                    (segment.position, sentence_index),
                    (),
                )
                if decision.target_start is not None and decision.target_end is not None
            )
            if not changes:
                resolved_sentences.append(sentence)
                continue
            resolved_sentences.append(
                Sentence(
                    text=_apply_text_changes(sentence.text, changes),
                    time_range=sentence.time_range,
                    words=_apply_word_changes(sentence.words, changes),
                    asr_boundary_word_indexes=sentence.asr_boundary_word_indexes,
                )
            )
        resolved_segments.append(
            Segment(
                position=segment.position,
                text="".join(sentence.text for sentence in resolved_sentences),
                time_range=segment.time_range,
                sentences=tuple(resolved_sentences),
            )
        )
    return tuple(resolved_segments)


def _prefilter_sort_key(
    target: _PrefilteredTarget,
) -> tuple[float, float, float, int, int]:
    context_probability = (
        target.original_score if target.original_score is not None else 1.0
    )
    asr_confidence = (
        target.asr_confidence if target.asr_confidence is not None else 1.0
    )
    return (
        context_probability,
        asr_confidence,
        -target.vocabulary_rank,
        -target.lexical_candidate_count,
        target.morpheme.start,
    )


def _surface_confidence(
    words: tuple[Word, ...],
    surface: str,
) -> float | None:
    for start_index in range(len(words)):
        combined = ""
        confidences: list[float] = []
        for word in words[start_index:]:
            combined += unicodedata.normalize("NFKC", word.text).strip()
            if word.confidence is not None:
                confidences.append(word.confidence)
            if combined == surface:
                return min(confidences) if confidences else None
            if not surface.startswith(combined):
                break
    return None


def _apply_word_changes(
    words: tuple[Word, ...],
    changes: tuple[_AcceptedChange, ...],
) -> tuple[Word, ...]:
    if not words:
        return words

    pending = list(changes)
    resolved_words: list[Word] = []
    word_index = 0
    while word_index < len(words):
        word = words[word_index]
        replacement = None
        for index, change in enumerate(pending):
            if word.text == change.original_text:
                replacement = pending.pop(index)
                break

            combined = ""
            for end_index in range(word_index, len(words)):
                combined += words[end_index].text
                if combined == change.original_text:
                    replacement = pending.pop(index)
                    matched_words = words[word_index : end_index + 1]
                    resolved_words.append(
                        Word(
                            text=change.selected_text,
                            time_range=TimeRange(
                                matched_words[0].time_range.start_seconds,
                                matched_words[-1].time_range.end_seconds,
                            ),
                            confidence=min(
                                (
                                    item.confidence
                                    for item in matched_words
                                    if item.confidence is not None
                                ),
                                default=None,
                            ),
                        )
                    )
                    word_index = end_index + 1
                    break
                if not change.original_text.startswith(combined):
                    break
            if replacement is not None:
                break

        if replacement is None:
            resolved_words.append(word)
            word_index += 1
            continue

        if word_index > 0 and resolved_words[-1].text == replacement.selected_text:
            continue

        resolved_words.append(
            Word(
                text=replacement.selected_text,
                time_range=word.time_range,
                confidence=word.confidence,
            )
        )
        word_index += 1

    return tuple(resolved_words)


def _compatible_part_of_speech(
    original: tuple[str, ...],
    candidate: tuple[str, ...],
) -> bool:
    if _pos(original, 0) != _pos(candidate, 0):
        return False

    if _pos(original, 1) and _pos(candidate, 1):
        return _pos(original, 1) == _pos(candidate, 1)

    return True


def _inflection_class(part_of_speech: tuple[str, ...]) -> tuple[str, str, str]:
    return (
        _pos(part_of_speech, 0),
        _pos(part_of_speech, 1),
        _pos(part_of_speech, 4),
    )


def _is_content_morpheme(morpheme: _AnalyzedMorpheme) -> bool:
    return (
        morpheme.surface not in _SKIPPED_SURFACES
        and _pos(morpheme.part_of_speech, 0) in _CONTENT_POS
        and _has_japanese_text(morpheme.surface)
    )


def _is_inflected_content_morpheme(morpheme: _AnalyzedMorpheme) -> bool:
    return (
        _pos(morpheme.part_of_speech, 0) in {"動詞", "形容詞"}
        and _pos(morpheme.part_of_speech, 5) not in {"", "*"}
    )


def _is_general_single_character_noun(morpheme: _AnalyzedMorpheme) -> bool:
    return (
        len(morpheme.surface) == 1
        and _pos(morpheme.part_of_speech, 0) == "名詞"
        and _pos(morpheme.part_of_speech, 1) == "普通名詞"
        and _pos(morpheme.part_of_speech, 2) == "一般"
    )


def _is_proper_noun(morpheme: _AnalyzedMorpheme) -> bool:
    return (
        _pos(morpheme.part_of_speech, 0) == "名詞"
        and _pos(morpheme.part_of_speech, 1) == "固有名詞"
    )


def _generation_audit(
    segment_position: int,
    sentence_index: int,
    morpheme: _AnalyzedMorpheme,
    reason: str,
    *,
    candidate_count: int = 0,
    candidate_examples: tuple[str, ...] = (),
) -> HomophoneCandidateGenerationAudit:
    return HomophoneCandidateGenerationAudit(
        segment_position=segment_position,
        sentence_index=sentence_index,
        surface=morpheme.surface,
        reading=morpheme.reading,
        part_of_speech=morpheme.part_of_speech,
        target_start=morpheme.start,
        target_end=morpheme.end,
        reason=reason,
        candidate_count=candidate_count,
        candidate_examples=candidate_examples,
    )


def _homophone_target(morpheme: _AnalyzedMorpheme) -> HomophoneTarget:
    return HomophoneTarget(
        text=morpheme.surface,
        reading=morpheme.reading,
        part_of_speech=morpheme.part_of_speech,
        start=morpheme.start,
        end=morpheme.end,
    )


def _shadow_candidate(
    analyzer: SudachiReadingAnalyzer,
    segment_position: int,
    sentence_index: int,
    strategy: str,
    target: HomophoneTarget,
    candidates: tuple[str, ...],
    morpheme_span: tuple[str, ...],
) -> HomophoneShadowCandidate:
    generated_candidates = tuple(
        dict.fromkeys(
            candidate for candidate in candidates if candidate != target.text
        )
    )
    unique_candidates = tuple(
        candidate
        for candidate in generated_candidates
        if _compatible_shadow_candidate(
            analyzer,
            target,
            candidate,
            strategy,
        )
    )
    retained = set(unique_candidates)
    return HomophoneShadowCandidate(
        segment_position=segment_position,
        sentence_index=sentence_index,
        strategy=strategy,
        surface=target.text,
        reading=target.reading,
        part_of_speech=target.part_of_speech,
        target_start=target.start,
        target_end=target.end,
        candidates=unique_candidates,
        morpheme_span=morpheme_span,
        generated_candidates=generated_candidates,
        filtered_out_candidates=tuple(
            candidate
            for candidate in generated_candidates
            if candidate not in retained
        ),
    )


def _compatible_shadow_candidate(
    analyzer: SudachiReadingAnalyzer,
    target: HomophoneTarget,
    candidate: str,
    strategy: str,
) -> bool:
    analyzed = analyzer.analyze_single_token(candidate)
    if analyzed is None:
        return False
    if analyzed.reading != target.reading and not (
        strategy == "inflected"
        and _plausible_asr_reading_variant(target.reading, analyzed.reading)
    ):
        return False
    if not _compatible_part_of_speech(target.part_of_speech, analyzed.part_of_speech):
        return False
    if not _compatible_script_change(target.text, analyzed.surface):
        return False
    if _pos(target.part_of_speech, 0) == "名詞":
        target_subtype = _pos(target.part_of_speech, 2)
        candidate_subtype = _pos(analyzed.part_of_speech, 2)
        if (
            target_subtype not in {"", "*"}
            and candidate_subtype not in {"", "*"}
            and target_subtype != candidate_subtype
        ):
            return False
    if strategy == "inflected":
        for index in (4, 5):
            expected = _pos(target.part_of_speech, index)
            actual = _pos(analyzed.part_of_speech, index)
            if expected not in {"", "*"} and expected != actual:
                return False
    return True


def _should_record_generation_audit(
    words: tuple[Word, ...],
    surface: str,
    *,
    max_confidence: float = _GENERATION_AUDIT_MAX_CONFIDENCE,
) -> bool:
    confidence = _surface_confidence(words, surface)
    return confidence is None or confidence <= max_confidence


def _compatible_script_change(original: str, candidate: str) -> bool:
    return _script_profile(original) == _script_profile(candidate)


def _script_profile(value: str) -> tuple[bool, bool, bool]:
    return (
        _has_kanji(value),
        any("ぁ" <= character <= "ゖ" for character in value),
        any("ァ" <= character <= "ヺ" for character in value),
    )


def _pos(part_of_speech: tuple[str, ...], index: int) -> str:
    if index >= len(part_of_speech):
        return ""
    return part_of_speech[index]


def _normalize_reading(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value.strip())
    if not normalized or normalized == "*":
        return ""

    return "".join(_katakana_to_hiragana(character) for character in normalized)


def _katakana_to_hiragana(character: str) -> str:
    codepoint = ord(character)
    if 0x30A1 <= codepoint <= 0x30F6:
        return chr(codepoint - 0x60)
    return character


def _has_japanese_text(value: str) -> bool:
    for character in value:
        codepoint = ord(character)
        if (
            0x3040 <= codepoint <= 0x309F
            or 0x30A0 <= codepoint <= 0x30FF
            or 0x3400 <= codepoint <= 0x9FFF
        ):
            return True

    return False


def _has_kanji(value: str) -> bool:
    for character in value:
        codepoint = ord(character)
        if 0x3400 <= codepoint <= 0x9FFF:
            return True

    return False


def _has_hiragana(value: str) -> bool:
    return any("ぁ" <= character <= "ゖ" for character in value)


def _trailing_hiragana(value: str) -> str:
    index = len(value)
    while index > 0 and "ぁ" <= value[index - 1] <= "ゖ":
        index -= 1
    return value[index:]


def _inflected_surface_from_lemma(
    source_surface: str,
    source_lemma: str,
    candidate_lemma: str,
) -> str | None:
    source_tail = _trailing_hiragana(source_surface)
    source_lemma_tail = _trailing_hiragana(source_lemma)
    candidate_lemma_tail = _trailing_hiragana(candidate_lemma)
    if not source_tail or not source_lemma_tail or not candidate_lemma_tail:
        return None

    source_stem = source_surface[: -len(source_tail)]
    source_lemma_stem = source_lemma[: -len(source_lemma_tail)]
    candidate_stem = candidate_lemma[: -len(candidate_lemma_tail)]
    if not source_stem or source_stem != source_lemma_stem or not candidate_stem:
        return None
    return f"{candidate_stem}{source_tail}"


def _plausible_asr_reading_variant(original: str, candidate: str) -> bool:
    original_units = _devoiced_reading_units(original)
    candidate_units = _devoiced_reading_units(candidate)
    if not original_units or not candidate_units:
        return False
    if abs(len(original_units) - len(candidate_units)) > 1:
        return False
    return _edit_distance(original_units, candidate_units) <= 1


def _devoiced_reading_units(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFD", _normalize_reading(value))
    return tuple(
        character
        for character in normalized
        if character not in {"\u3099", "\u309a"}
    )


def _edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _subsequence_start(sequence: list[int], subsequence: list[int]) -> int | None:
    if not subsequence:
        return None
    limit = len(sequence) - len(subsequence) + 1
    for index in range(max(limit, 0)):
        if sequence[index : index + len(subsequence)] == subsequence:
            return index
    return None


__all__ = [
    "BertHomophoneResolver",
    "BertMaskedLanguageHomophoneCandidateGenerator",
    "DEFAULT_HOMOPHONE_MIN_CANDIDATE_SCORE",
    "DEFAULT_HOMOPHONE_MIN_TOKEN_CHARS",
    "DEFAULT_HOMOPHONE_MAX_CANDIDATE_PIECES",
    "DEFAULT_HOMOPHONE_MAX_LEXICAL_CANDIDATES",
    "DEFAULT_HOMOPHONE_MAX_TARGETS_PER_SENTENCE",
    "DEFAULT_HOMOPHONE_MODEL_ID",
    "DEFAULT_HOMOPHONE_SCORE_MARGIN",
    "DEFAULT_HOMOPHONE_TOP_K",
    "HomophoneCandidateGenerator",
    "HomophoneLanguageModelCandidate",
    "HomophonePrefilterCandidateGenerator",
    "HomophoneResolverDependencyError",
    "HomophoneTarget",
    "SudachiReadingAnalyzer",
]
