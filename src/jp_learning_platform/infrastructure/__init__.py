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
    DEFAULT_WHISPER_COMPUTE_TYPE,
    DEFAULT_WHISPER_DEVICE,
    DEFAULT_WHISPER_MODEL_SIZE,
    FasterWhisperDependencyError,
    FasterWhisperTranscriber,
)
from jp_learning_platform.infrastructure.qwen_repairer import (
    DEFAULT_QWEN_CONTEXT,
    DEFAULT_QWEN_GPU_LAYERS,
    DEFAULT_QWEN_THREADS,
    LlamaCppQwenRepairer,
    PassthroughQwenRepairer,
    QwenDependencyError,
    QwenModelNotFoundError,
)
from jp_learning_platform.infrastructure.srt_subtitle_writer import (
    SrtSubtitleWriter,
    format_srt_subtitle,
    format_srt_timestamp,
)
from jp_learning_platform.infrastructure.subtitle_quality import (
    ConservativeSubtitleMerger,
    DomainSubtitleValidator,
    LocalReadabilityOptimizer,
)
from jp_learning_platform.infrastructure.tool_registry import (
    DuplicateToolError,
    RegisteredTool,
    ToolNotFoundError,
    ToolRegistry,
    ToolRegistryError,
)
from jp_learning_platform.infrastructure.whisperx_aligner import (
    DEFAULT_WHISPERX_LANGUAGE,
    PassthroughWhisperXAligner,
    WhisperXAlignerAdapter,
    WhisperXDependencyError,
)
from jp_learning_platform.infrastructure.word_subtitle_builder import (
    WordSubtitleBuilder,
)

__all__ = [
    "AudioFileNotFoundError",
    "AudioFormat",
    "AudioLoader",
    "AudioLoaderError",
    "ConservativeSubtitleMerger",
    "DEFAULT_QWEN_CONTEXT",
    "DEFAULT_QWEN_GPU_LAYERS",
    "DEFAULT_QWEN_THREADS",
    "DEFAULT_WHISPER_COMPUTE_TYPE",
    "DEFAULT_WHISPER_DEVICE",
    "DEFAULT_WHISPER_MODEL_SIZE",
    "DEFAULT_WHISPERX_LANGUAGE",
    "DomainSubtitleValidator",
    "DuplicateToolError",
    "EmptyAudioFileError",
    "FasterWhisperDependencyError",
    "FasterWhisperTranscriber",
    "LlamaCppQwenRepairer",
    "LocalReadabilityOptimizer",
    "LoadedAudio",
    "PassthroughQwenRepairer",
    "PassthroughWhisperXAligner",
    "QwenDependencyError",
    "QwenModelNotFoundError",
    "RegisteredTool",
    "SrtSubtitleWriter",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolRegistryError",
    "UnsupportedAudioFormatError",
    "WhisperXAlignerAdapter",
    "WhisperXDependencyError",
    "WordSubtitleBuilder",
    "format_srt_subtitle",
    "format_srt_timestamp",
]
