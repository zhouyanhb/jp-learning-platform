"""Infrastructure adapters for external tools."""

from jp_learning_platform.infrastructure.audio_loader import (
    AudioFileNotFoundError,
    AudioFormat,
    AudioLoader,
    AudioLoaderError,
    EmptyAudioFileError,
    LoadedAudio,
    UnsupportedAudioFormatError,
)
from jp_learning_platform.infrastructure.faster_whisper_transcriber import (
    FasterWhisperDependencyError,
    FasterWhisperTranscriber,
)
from jp_learning_platform.infrastructure.srt_subtitle_writer import (
    SrtSubtitleWriter,
    format_srt_subtitle,
    format_srt_timestamp,
)
from jp_learning_platform.infrastructure.tool_registry import (
    DuplicateToolError,
    RegisteredTool,
    ToolNotFoundError,
    ToolRegistry,
    ToolRegistryError,
)
from jp_learning_platform.infrastructure.word_subtitle_builder import (
    WordSubtitleBuilder,
)

__all__ = [
    "AudioFileNotFoundError",
    "AudioFormat",
    "AudioLoader",
    "AudioLoaderError",
    "DuplicateToolError",
    "EmptyAudioFileError",
    "FasterWhisperDependencyError",
    "FasterWhisperTranscriber",
    "LoadedAudio",
    "RegisteredTool",
    "SrtSubtitleWriter",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolRegistryError",
    "UnsupportedAudioFormatError",
    "WordSubtitleBuilder",
    "format_srt_subtitle",
    "format_srt_timestamp",
]
