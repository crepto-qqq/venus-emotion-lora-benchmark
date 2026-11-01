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
    "audit-format": (".output_contract", "audit_format_command"),
    "score-recognition": (".scoring", "score_recognition_command"),
    "compare-recognition": (".scoring", "compare_recognition_command"),
    "prepare-guidance-judge": (".quality_benchmark", "prepare_guidance_judge_command"),
    "judge-guidance": (".quality_judge_runner", "judge_guidance_command"),
    "lock-guidance-judge": (".quality_benchmark", "lock_guidance_judge_command"),
    "report-guidance": (".quality_benchmark", "report_guidance_command"),
    "report-benchmark": (".quality_benchmark", "report_benchmark_command"),
    "prepare-preference-judge": (".preference_benchmark", "prepare_preference_judge_command"),
    "judge-preference": (".preference_judge_runner", "judge_preference_command"),
    "report-preference": (".preference_benchmark", "report_preference_command"),
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
        "  run                 Run one frozen A, B0, B1, or C condition\n"
        "  validate-output     Validate one completed inference attempt\n"
        "  audit-format        Audit B1 or C structured output compliance\n"
        "  score-recognition   Calculate B0, B1, or C recognition results\n"
        "  compare-recognition Summarize paired B1/C recognition outcomes\n"
        "  prepare-guidance-judge Validate B1/C and make anonymous private judge items\n"
        "  judge-guidance      Run pinned image-aware judge and lock 156 scores\n"
        "  lock-guidance-judge Validate and lock 156 anonymous judge scores\n"
        "  report-guidance     Unblind locked scores into a paired aggregate\n"
        "  report-benchmark    Combine format, recognition and guidance proxy\n"
        "  prepare-preference-judge Prepare 156 private swapped-order preference items\n"
        "  judge-preference    Run and lock local image-aware A/B preference decisions\n"
        "  report-preference   Report B1/C wins, ties and swap instability\n"
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
