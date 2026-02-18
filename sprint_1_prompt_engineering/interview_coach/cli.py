from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from services.interview_ops import (
    DATA_DIR,
    INTERVIEWEE_PROFILE_PATH,
    build_interview_profile,
    build_profile_filename,
    extract_jd_fields,
    parse_interviewee_profile,
    parse_interviewer_background,
    _as_interviewee_payload,
    _coerce_temperature_for_model,
    _load_interviewee_profile,
    _merge_interviewee_cover_letter,
    create_chat_completion_with_fallback,
)
from utils.interviewer_store import (
    InterviewerProfile,
    delete_interviewer,
    list_interviewers_with_paths,
    save_interviewer,
)


def _read_input(file_path: str | None, text: str | None, label: str) -> str:
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    if text:
        return text.strip()
    raise ValueError(f"Provide --{label}-text or --{label}-file")


def _load_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    fallback = os.getenv("OPENAI_API_KEY")
    if not fallback:
        raise ValueError("OPENAI_API_KEY is required. Provide --api-key or set the environment variable.")
    return fallback


def _extract_jd(args: argparse.Namespace) -> int:
    api_key = _load_api_key(args.api_key)
    jd_text = _read_input(args.jd_text_file, args.jd_text, "jd")
    extracted, model = extract_jd_fields(api_key, jd_text, args.company, args.position)
    profile = build_interview_profile(extracted)

    output_dir = Path(args.output_dir or DATA_DIR / "profiles")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = build_profile_filename(extracted.company, extracted.position)
    output_path = Path(args.output or output_dir / output_name)
    output_path.write_text(json.dumps(profile | {"_model": model}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path.as_posix())
    return 0


def _parse_interviewee(args: argparse.Namespace) -> int:
    api_key = _load_api_key(args.api_key)
    text = _read_input(args.input_file, args.input_text, "input")
    parsed = parse_interviewee_profile(api_key, text, args.model, source_label=args.source)

    output_path = Path(args.output or INTERVIEWEE_PROFILE_PATH)
    if args.source == "cover_letter":
        current_profile = _load_interviewee_profile()
        if not current_profile:
            raise ValueError(
                "No existing interviewee profile was found to merge cover-letter insights. "
                "Run parse-interviewee with --source resume first."
            )
        merged = _merge_interviewee_cover_letter(current_profile, parsed)
        output_payload = merged
    else:
        output_payload = _as_interviewee_payload(parsed)

    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path.as_posix())
    return 0


def _parse_interviewer(args: argparse.Namespace) -> int:
    api_key = _load_api_key(args.api_key)
    background = _read_input(args.input_file, args.input_text, "input")
    parsed = parse_interviewer_background(
        api_key=api_key,
        interviewer_name=args.name,
        background_text=background,
        model=args.model,
    )
    payload = parsed.model_dump()
    output = Path(args.output) if args.output else None

    if args.save:
        interviewer = InterviewerProfile(
            name=parsed.name.strip(),
            background=parsed.background,
            is_generic_ai=False,
            role=parsed.role,
            expertise=parsed.expertise,
            potential_questions=parsed.potential_questions,
        )
        saved_path = save_interviewer(interviewer)
        print(saved_path.as_posix())
    elif output is not None:
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output.as_posix())
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _list_interviewers(args: argparse.Namespace) -> int:
    records = list_interviewers_with_paths(args.jd_title)
    payload = [
        {"path": str(path), "name": profile.name, "role": profile.role} for path, profile in records
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _delete_interviewer(args: argparse.Namespace) -> int:
    delete_interviewer(args.path)
    print(f"deleted={args.path}")
    return 0


def _chat_turn(args: argparse.Namespace) -> int:
    api_key = _load_api_key(args.api_key)
    client = OpenAI(api_key=api_key)
    response, model = create_chat_completion_with_fallback(
        client=client,
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=_coerce_temperature_for_model(args.model, args.temperature),
    )
    content = response.choices[0].message.content or ""
    if model != args.model:
        content = f"[Model fallback: {model}]\\n{content}"
    print(content)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Interview Coach CLI")
    parser.add_argument("--api-key", help="OpenAI API key")

    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract-jd", help="Extract interview profile from a JD")
    extract.add_argument("--company", required=True)
    extract.add_argument("--position", required=True)
    group_jd = extract.add_mutually_exclusive_group(required=True)
    group_jd.add_argument("--jd-text", help="Raw job-description text")
    group_jd.add_argument("--jd-text-file", help="Path to job-description text file")
    extract.add_argument("--model", default="gpt-4o-mini")
    extract.add_argument("--output", help="Output profile path")
    extract.add_argument("--output-dir", default="data/profiles")
    extract.set_defaults(func=_extract_jd)

    parse_ee = subparsers.add_parser("parse-interviewee", help="Extract interviewee profile")
    parse_ee.add_argument("--source", choices=["resume", "cover_letter"], default="resume")
    parse_group = parse_ee.add_mutually_exclusive_group(required=True)
    parse_group.add_argument("--input-text", help="Input text")
    parse_group.add_argument("--input-file", help="Path to input text file")
    parse_ee.add_argument("--model", default="gpt-4o-mini")
    parse_ee.add_argument("--output", help="Interviewee profile output path")
    parse_ee.set_defaults(func=_parse_interviewee)

    parse_ivw = subparsers.add_parser(
        "parse-interviewer",
        help="Extract interviewer profile from background text",
    )
    parse_ivw.add_argument("--name", required=True)
    parse_group_i = parse_ivw.add_mutually_exclusive_group(required=True)
    parse_group_i.add_argument("--input-text", help="Raw background")
    parse_group_i.add_argument("--input-file", help="Path to background text file")
    parse_ivw.add_argument("--model", default="gpt-4o-mini")
    parse_ivw.add_argument("--output", help="Output interviewer profile path")
    parse_ivw.add_argument("--save", action="store_true", help="Persist parsed interviewer profile")
    parse_ivw.set_defaults(func=_parse_interviewer)

    list_ivw = subparsers.add_parser("list-interviewers", help="List persisted interviewer profiles")
    list_ivw.add_argument("--jd-title", default=None, help="Optional JD title filter")
    list_ivw.set_defaults(func=_list_interviewers)

    remove_ivw = subparsers.add_parser("delete-interviewer", help="Delete a persisted interviewer profile")
    remove_ivw.add_argument("path", help="Path to interviewer JSON file")
    remove_ivw.set_defaults(func=_delete_interviewer)

    chat_turn = subparsers.add_parser("chat-turn", help="Run one chat-turn completion")
    chat_turn.add_argument("--prompt", required=True)
    chat_turn.add_argument("--model", default="gpt-4o-mini")
    chat_turn.add_argument("--temperature", type=float, default=0.4)
    chat_turn.set_defaults(func=_chat_turn)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
