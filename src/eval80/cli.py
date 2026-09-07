"""Single command-line interface for the Eval80-v1 workflow."""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib import import_module


Command = Callable[[list[str] | None], int]

COMMANDS: dict[str, tuple[str, str]] = {
    "prepare": (".preparation", "prepare_command"),
    "validate-review": (".validation", "validate_review_command"),
    "freeze": (".validation", "freeze_command"),
    "run": (".inference", "run_command"),
    "validate-output": (".validation", "validate_output_command"),
    "score-recognition": (".scoring", "score_recognition_command"),
    "create-score-sheets": (".scoring", "create_score_sheets_command"),
    "summarize": (".scoring", "summarize_guidance_command"),
}


def _load_command(command_name: str) -> Command:
    module_name, function_name = COMMANDS[command_name]
    module = import_module(module_name, package=__package__)
    return getattr(module, function_name)


def _print_help() -> None:
    print(
        "Usage: python -m src.eval80.cli <command> [options]\n\n"
        "Commands:\n"
        "  prepare             Create the private Eval80 review package\n"
        "  validate-review     Validate the fixed set and human review\n"
        "  freeze              Create the approved label-free package\n"
        "  run                 Run one frozen A, B0, or B1 condition\n"
        "  validate-output     Validate one completed inference attempt\n"
        "  score-recognition   Calculate B0, B1, or C recognition results\n"
        "  create-score-sheets Create six blind reviewer score sheets\n"
        "  summarize           Validate and summarize completed guidance scores\n\n"
        "Use '<command> --help' for command-specific options."
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_help()
        return 0

    command_name = arguments.pop(0)
    if command_name not in COMMANDS:
        available = ", ".join(COMMANDS)
        print(f"Unknown Eval80 command {command_name!r}. Choose one of: {available}.")
        return 2
    return _load_command(command_name)(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
