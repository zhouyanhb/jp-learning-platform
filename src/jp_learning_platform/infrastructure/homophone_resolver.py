"""Constrained homophone semantic resolution adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import unicodedata

from jp_learning_platform.domain import Segment, Sentence, TimeRange, Word
from jp_learning_platform.infrastructure.pipeline_config import (
    DEFAULT_HOMOPHONE_PREFILTER_CONFIG,
)
from jp_learning_platform.workflow.homophone_stage import (
    HomophoneCandidateScore,
    HomophoneResolution,
    HomophoneResolutionDecision,
    HomophoneResolutionRequest,
)

DEFAULT_HOMOPHONE_MODEL_ID = "tohoku-nlp/bert-base-japanese-v3"
DEFAULT_HOMOPHONE_TOP_K = 80
DEFAULT_HOMOPHONE_SCORE_MARGIN = 0.0
DEFAULT_HOMOPHONE_MIN_CANDIDATE_SCORE = 0.001
DEFAULT_HOMOPHONE_MIN_TOKEN_CHARS = 2
DEFAULT_HOMOPHONE_MAX_CANDIDATE_PIECES = 3
DEFAULT_HOMOPHONE_MAX_LEXICAL_CANDIDATES = 64
DEFAULT_HOMOPHONE_MAX_TARGETS_PER_SENTENCE = (
    DEFAULT_HOMOPHONE_PREFILTER_CONFIG.max_targets_per_sentence
)
_DEFAULT_SUDACHI_SPLIT_MODE = "C"
_CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞"}
_SKIPPED_SURFACES = {"する", "した", "して", "ある", "いる", "ます", "です"}


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


@dataclass(frozen=True, slots=True)
class _AnalyzedMorpheme:
    surface: str
    reading: str
    part_of_speech: tuple[str, ...]
    start: int
    end: int


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

    def vocabulary_rank_for(self, text: str) -> float:
        """Use normalized token ids as a stable vocabulary-frequency proxy."""
        tokenizer, _, _ = self._load_model()
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        vocabulary_size = max(len(tokenizer.get_vocab()), 1)
        if not token_ids:
            return 1.0
        return min(sum(token_ids) / len(token_ids) / vocabulary_size, 1.0)

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
    min_token_chars: int = DEFAULT_HOMOPHONE_MIN_TOKEN_CHARS
    max_targets_per_sentence: int = DEFAULT_HOMOPHONE_MAX_TARGETS_PER_SENTENCE
    require_original_score: bool = True

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

    def resolve(self, request: HomophoneResolutionRequest) -> HomophoneResolution:
        if not isinstance(request, HomophoneResolutionRequest):
            raise TypeError("request must be a HomophoneResolutionRequest.")

        resolved_segments: list[Segment] = []
        decisions: list[HomophoneResolutionDecision] = []
        for segment in request.segments:
            resolved_segment, segment_decisions = self._resolve_segment(segment)
            resolved_segments.append(resolved_segment)
            decisions.extend(segment_decisions)

        return HomophoneResolution(
            source_path=request.source_path,
            segments=tuple(resolved_segments),
            decisions=tuple(decisions),
        )

    def _resolve_segment(
        self,
        segment: Segment,
    ) -> tuple[Segment, tuple[HomophoneResolutionDecision, ...]]:
        sentences = segment.sentences or (
            Sentence(
                text=segment.text,
                time_range=segment.time_range,
                words=(),
                speaker_id=segment.speaker_id,
            ),
        )

        resolved_sentences: list[Sentence] = []
        decisions: list[HomophoneResolutionDecision] = []
        for sentence_index, sentence in enumerate(sentences):
            resolved_sentence, sentence_decisions = self._resolve_sentence(
                segment.position,
                sentence_index,
                sentence,
            )
            resolved_sentences.append(resolved_sentence)
            decisions.extend(sentence_decisions)

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
                speaker_id=segment.speaker_id,
            ),
            tuple(decisions),
        )

    def _resolve_sentence(
        self,
        segment_position: int,
        sentence_index: int,
        sentence: Sentence,
    ) -> tuple[Sentence, tuple[HomophoneResolutionDecision, ...]]:
        assert self.analyzer is not None
        morphemes = self.analyzer.analyze(sentence.text)
        selected_targets, original_scores = self._prefilter_targets(
            sentence,
            morphemes,
        )
        decisions: list[HomophoneResolutionDecision] = []
        changes: list[_AcceptedChange] = []
        for morpheme in selected_targets:
            decision = self._decision_for_target(
                segment_position=segment_position,
                sentence_index=sentence_index,
                sentence_text=sentence.text,
                morpheme=morpheme,
                prefetched_original_score=original_scores.get(
                    (morpheme.start, morpheme.end)
                ),
                has_prefetched_original_score=(
                    (morpheme.start, morpheme.end) in original_scores
                ),
            )
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
            return sentence, tuple(decisions)

        text = _apply_text_changes(sentence.text, tuple(changes))
        words = _apply_word_changes(sentence.words, tuple(changes))
        return (
            Sentence(
                text=text,
                time_range=sentence.time_range,
                words=words,
                speaker_id=sentence.speaker_id,
            ),
            tuple(decisions),
        )

    def _prefilter_targets(
        self,
        sentence: Sentence,
        morphemes: tuple[_AnalyzedMorpheme, ...],
    ) -> tuple[tuple[_AnalyzedMorpheme, ...], dict[tuple[int, int], float | None]]:
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
            return eligible[: self.max_targets_per_sentence], {}

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
                continue
            targets.append(target)
            candidate_counts.append(len(candidates))
            filtered_morphemes.append(morpheme)

        if not targets:
            return (), {}

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
        return (
            tuple(item.morpheme for item in selected),
            {
                (item.morpheme.start, item.morpheme.end): item.original_score
                for item in selected
            },
        )

    def _decision_for_target(
        self,
        *,
        segment_position: int,
        sentence_index: int,
        sentence_text: str,
        morpheme: _AnalyzedMorpheme,
        prefetched_original_score: float | None = None,
        has_prefetched_original_score: bool = False,
    ) -> HomophoneResolutionDecision | None:
        if not self._should_consider(morpheme):
            return None

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
        for candidate in language_model_candidates:
            analyzed_candidate = self.analyzer.analyze_single_token(candidate.text)
            if analyzed_candidate is None:
                continue
            if analyzed_candidate.surface == morpheme.surface:
                continue
            if analyzed_candidate.reading != morpheme.reading:
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

        original_score = prefetched_original_score
        if not has_prefetched_original_score:
            original_score = self.candidate_generator.score_for(
                sentence_text,
                target,
                morpheme.surface,
            )
        if not scored_candidates:
            return HomophoneResolutionDecision(
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
            )

        selected = max(
            scored_candidates,
            key=lambda candidate: candidate.score or 0.0,
        )
        selected_score = selected.score
        accepted, reason = self._accept_candidate(
            original_score=original_score,
            selected_score=selected_score,
        )
        return HomophoneResolutionDecision(
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
    ) -> tuple[bool, str]:
        if selected_score is None:
            return False, "missing_candidate_score"

        if original_score is None:
            if self.require_original_score:
                return False, "missing_original_score"
            if selected_score < self.min_candidate_score:
                return False, "candidate_score_too_low"
            return True, "accepted_same_reading_context"

        if selected_score <= original_score + self.score_margin:
            return False, "candidate_not_better_than_original"

        return True, "accepted_same_reading_context"


def _apply_text_changes(
    text: str,
    changes: tuple[_AcceptedChange, ...],
) -> str:
    updated = text
    for change in sorted(changes, key=lambda item: item.start, reverse=True):
        updated = f"{updated[:change.start]}{change.selected_text}{updated[change.end:]}"
    return updated


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
                            speaker_id=word.speaker_id,
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
                speaker_id=word.speaker_id,
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


def _compatible_script_change(original: str, candidate: str) -> bool:
    if _has_kanji(original) and not _has_kanji(candidate):
        return False

    return True


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
