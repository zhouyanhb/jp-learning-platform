"""Word-aware subtitle builder adapter."""

from __future__ import annotations

from dataclasses import dataclass

from jp_learning_platform.domain import Sentence, Subtitle
from jp_learning_platform.workflow.subtitle_builder_stage import (
    SubtitleBuild,
    SubtitleBuildRequest,
)


@dataclass(frozen=True, slots=True)
class WordSubtitleBuilder:
    """Build subtitle cues from segment sentences while preserving word timing."""

    def build(self, request: SubtitleBuildRequest) -> SubtitleBuild:
        if not isinstance(request, SubtitleBuildRequest):
            raise TypeError("request must be a SubtitleBuildRequest.")

        subtitles: list[Subtitle] = []
        for segment in request.segments:
            sentences = segment.sentences or (
                Sentence(
                    text=segment.text,
                    time_range=segment.time_range,
                    words=(),
                    speaker_id=segment.speaker_id,
                ),
            )
            for sentence in sentences:
                subtitles.append(
                    Subtitle(
                        index=len(subtitles) + 1,
                        text=sentence.text,
                        time_range=sentence.time_range,
                        speaker_id=sentence.speaker_id or segment.speaker_id,
                    )
                )

        return SubtitleBuild(
            source_path=request.source_path,
            subtitles=tuple(subtitles),
        )


__all__ = ["WordSubtitleBuilder"]
