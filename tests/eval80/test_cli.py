import io
from contextlib import redirect_stdout
import unittest

from src.eval80.cli import main


class Eval80CliTests(unittest.TestCase):
    def test_top_level_help_lists_the_workflow_commands(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--help"])
        self.assertEqual(result, 0)
        self.assertIn("prepare", output.getvalue())
        self.assertIn("validate-review", output.getvalue())
        self.assertIn("create-score-sheets", output.getvalue())

    def test_unknown_command_returns_usage_error(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["not-a-command"])
        self.assertEqual(result, 2)
        self.assertIn("Unknown Eval80 command", output.getvalue())


if __name__ == "__main__":
    unittest.main()
