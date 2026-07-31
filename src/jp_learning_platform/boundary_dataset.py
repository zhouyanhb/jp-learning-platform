"""Build punctuation-free sentence-boundary datasets from reviewed transcripts."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata


_SPEAKER_LABEL = re.compile(r"(?P<speaker>(?:女|男)(?:[12])?)\s*[:：]")
_OPTION_LINE = re.compile(r"^[1-4](?:\.|\s)\s*\S")
_INSTRUCTION_LINE = re.compile(r"^(?:問題[0-9一二三四五六七八九十]+|[0-9]+番)$")
_TERMINAL_MARKS = frozenset(("。", "？", "！", "?", "!"))
_ROLES = frozenset(("instruction", "dialogue", "question", "option"))


@dataclass(frozen=True, slots=True)
class _AnnotatedUnit:
    source_text: str
    boundary_types: tuple[str, ...]
    source_line: int
    speaker: str | None
    line_kind: str


def build_boundary_dataset(source_path: Path) -> dict[str, object]:
    """Create deterministic silver sentence boundaries from a transcript."""
    source_text = source_path.read_text(encoding="utf-8")
    samples = []
    for index, (text, start_line, end_line) in enumerate(
        _paragraphs(source_text),
        start=1,
    ):
        units = _annotated_units(text, start_line)
        if not units:
            continue
        roles = _unit_roles(units)
        sentences = []
        boundaries = []
        offset = 0
        for unit_index, (unit, role) in enumerate(zip(units, roles, strict=True)):
            normalized = _semantic_text(unit.source_text)
            if not normalized:
                continue
            start_offset = offset
            offset += len(normalized)
            is_final = unit_index == len(units) - 1
            sentences.append(
                {
                    "text": normalized,
                    "source_text": unit.source_text.strip(),
                    "role": role,
                    "speaker": unit.speaker,
                    "source_line": unit.source_line,
                    "start_char": start_offset,
                    "end_char": offset,
                }
            )
            if not is_final:
                boundaries.append(
                    {
                        "after_char": offset,
                        "types": list(unit.boundary_types),
                    }
                )

        input_text = "".join(item["text"] for item in sentences)
        if not input_text:
            continue
        samples.append(
            {
                "id": f"2021-12-n2-{index:03d}",
                "source_lines": [start_line, end_line],
                "input_text": input_text,
                "sentences": sentences,
                "boundaries": boundaries,
                "needs_review": _needs_review(units, roles),
            }
        )

    return {
        "schema_version": 2,
        "dataset_id": "jlpt-2021-12-n2-sentence-boundaries",
        "language": "ja",
        "annotation_status": "silver",
        "source": _portable_source_path(source_path),
        "normalization": {
            "unicode": "NFKC",
            "speaker_labels_removed": True,
            "whitespace_removed": True,
            "punctuation_and_symbols_removed": True,
        },
        "boundary_sources": [
            "terminal_punctuation",
            "speaker_turn",
            "source_line_break",
        ],
        "roles": {
            "instruction": "Spoken or written section and item directions.",
            "dialogue": "Listening-test body, including dialogue and monologue.",
            "question": "Exam narrator question about the listening body.",
            "option": "Numbered answer or response option.",
        },
        "limitations": [
            "The source mixes spoken transcript text with printed answer choices.",
            "Source transcription and orthography have not been fully proofread.",
            "needs_review samples require human semantic-boundary verification.",
        ],
        "samples": samples,
    }


def write_boundary_dataset(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            build_boundary_dataset(source_path),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _paragraphs(source_text: str) -> tuple[tuple[str, int, int], ...]:
    paragraphs: list[tuple[str, int, int]] = []
    lines: list[str] = []
    start_line = 1
    for line_number, line in enumerate(source_text.splitlines(), start=1):
        if line.strip():
            if not lines:
                start_line = line_number
            lines.append(line.strip())
            continue
        if lines:
            paragraphs.append(("\n".join(lines), start_line, line_number - 1))
            lines = []
    if lines:
        paragraphs.append(("\n".join(lines), start_line, len(source_text.splitlines())))
    return tuple(paragraphs)


def _annotated_units(text: str, start_line: int) -> tuple[_AnnotatedUnit, ...]:
    units: list[_AnnotatedUnit] = []
    lines = text.splitlines()
    for line_offset, line in enumerate(lines):
        _annotate_line(units, line, start_line + line_offset)
        if line_offset < len(lines) - 1:
            _add_boundary_evidence(units, "source_line_break")
    return tuple(units)


def _annotate_line(
    units: list[_AnnotatedUnit],
    line: str,
    source_line: int,
) -> None:
    current: list[str] = []
    speaker_name: str | None = None
    line_kind = _line_kind(line)
    index = 0
    while index < len(line):
        speaker = _SPEAKER_LABEL.match(line, index)
        if speaker is not None:
            if current:
                _flush_unit(
                    units,
                    current,
                    "speaker_turn",
                    source_line,
                    speaker_name,
                    line_kind,
                )
            else:
                _add_boundary_evidence(units, "speaker_turn")
            speaker_name = speaker.group("speaker")
            index = speaker.end()
            continue

        character = line[index]
        if character in _TERMINAL_MARKS:
            current.append(character)
            _flush_unit(
                units,
                current,
                "terminal_punctuation",
                source_line,
                speaker_name,
                line_kind,
            )
        else:
            current.append(character)
        index += 1

    _flush_unit(
        units,
        current,
        "document_end",
        source_line,
        speaker_name,
        line_kind,
    )


def _flush_unit(
    units: list[_AnnotatedUnit],
    current: list[str],
    boundary_type: str,
    source_line: int,
    speaker: str | None,
    line_kind: str,
) -> None:
    source_text = "".join(current).strip()
    current.clear()
    if source_text:
        units.append(
            _AnnotatedUnit(
                source_text,
                (boundary_type,),
                source_line,
                speaker,
                line_kind,
            )
        )


def _add_boundary_evidence(
    units: list[_AnnotatedUnit],
    boundary_type: str,
) -> None:
    if not units:
        return
    previous = units[-1]
    boundary_types = tuple(
        item for item in previous.boundary_types if item != "document_end"
    )
    if boundary_type in boundary_types:
        return
    units[-1] = _AnnotatedUnit(
        previous.source_text,
        (*boundary_types, boundary_type),
        previous.source_line,
        previous.speaker,
        previous.line_kind,
    )


def _line_kind(line: str) -> str:
    normalized = unicodedata.normalize("NFKC", line).strip()
    if _OPTION_LINE.match(normalized):
        return "option"
    if _INSTRUCTION_LINE.match(normalized):
        return "instruction"
    return "content"


def _unit_roles(units: tuple[_AnnotatedUnit, ...]) -> tuple[str, ...]:
    option_lines = {unit.source_line for unit in units if unit.line_kind == "option"}
    first_option_line = min(option_lines) if option_lines else None
    question_line = None
    if len(option_lines) == 4 and first_option_line is not None:
        question_lines = {
            unit.source_line
            for unit in units
            if unit.line_kind == "content" and unit.source_line < first_option_line
        }
        question_line = max(question_lines) if question_lines else None

    roles: list[str] = []
    for unit in units:
        if unit.line_kind in {"instruction", "option"}:
            roles.append(unit.line_kind)
        elif len(option_lines) == 3:
            roles.append("dialogue")
        elif question_line is not None and unit.source_line == question_line:
            roles.append("question")
        elif unit.speaker is None and _looks_like_exam_question(unit.source_text):
            roles.append("question")
        else:
            roles.append("dialogue")

    question_texts = {
        _semantic_text(unit.source_text)
        for unit, role in zip(units, roles, strict=True)
        if role == "question"
    }
    for index, unit in enumerate(units):
        if (
            roles[index] == "dialogue"
            and _semantic_text(unit.source_text) in question_texts
        ):
            roles[index] = "question"
    return tuple(roles)


def _looks_like_exam_question(text: str) -> bool:
    stripped = text.rstrip()
    return stripped.endswith(("?", "？")) or bool(
        re.search(r"(?:か|でしょうか)[。]$", stripped)
    )


def _portable_source_path(source_path: Path) -> str:
    if source_path.parent.name == "input":
        return f"input/{source_path.name}"
    return source_path.name


def _semantic_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )


def _needs_review(
    units: tuple[_AnnotatedUnit, ...],
    roles: tuple[str, ...],
) -> bool:
    if any(role not in _ROLES for role in roles):
        return True
    option_count = len({unit.source_line for unit in units if unit.line_kind == "option"})
    if option_count not in {0, 3, 4}:
        return True
    return option_count == 4 and "question" not in roles


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_boundary_dataset(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
