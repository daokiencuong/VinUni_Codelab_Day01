"""
Day 1 — LLM API Foundation
Test suite for student solutions and master keys.

Run from the 02-lab/ folder:
    pytest tests/ -v

All external API calls are mocked — no real API keys or internet connection required.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Locate directories relative to this test file
TESTS_DIR = Path(__file__).parent
LAB_DIR = TESTS_DIR.parent

# Resolve module paths
SOLUTION_PATH = LAB_DIR / "solution-code" / "solution.py"
TEMPLATE_PATH = LAB_DIR / "starter-code" / "template.py"

def _load_module(path: Path, unique_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Target file not found at: {path}")
    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Dynamically load the solution module if it exists (for grading), otherwise test the starter template
if SOLUTION_PATH.exists():
    _m = _load_module(SOLUTION_PATH, "solution")
else:
    _m = _load_module(TEMPLATE_PATH, "template")

# Import target functions
call_deepseek = getattr(_m, "call_deepseek")
call_gemini = getattr(_m, "call_gemini")
compare_models = getattr(_m, "compare_models")
streaming_chatbot = getattr(_m, "streaming_chatbot")
retry_with_backoff = getattr(_m, "retry_with_backoff")
batch_compare = getattr(_m, "batch_compare")
format_comparison_table = getattr(_m, "format_comparison_table")


# ---------------------------------------------------------------------------
# Mock Generators
# ---------------------------------------------------------------------------
def _make_chat_response(text: str):
    choice = MagicMock()
    choice.message.content = text
    resp = MagicMock()
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    resp.usage = usage
    return resp


def _make_deepseek_response(text: str = "Hello from DeepSeek"):
    return _make_chat_response(text)


def _make_gemini_response(text: str = "Hello from Gemini"):
    resp = MagicMock()
    resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = 12
    usage.candidates_token_count = 25
    resp.usage_metadata = usage
    return resp


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
class TestCallDeepSeek(unittest.TestCase):

    @patch.object(_m, "OpenAI")
    def test_deepseek_returns_tuple_structure_and_forwards_parameters(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_deepseek_response(
            "DeepSeek Mock Response"
        )

        text, latency, usage = call_deepseek(
            "Hello",
            model="deepseek-chat",
            temperature=0.2,
            top_p=0.8,
            max_tokens=128,
        )

        self.assertEqual(text, "DeepSeek Mock Response")
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["input_tokens"], 10)
        self.assertEqual(usage["output_tokens"], 20)
        MockOpenAI.assert_called_once_with(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        mock_client.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2,
            top_p=0.8,
            max_tokens=128,
            extra_body={"thinking": {"type": "disabled"}},
        )


class TestCallGemini(unittest.TestCase):

    @patch("google.genai.Client")
    def test_gemini_returns_tuple_structure_new_sdk(self, MockGenAIClient):
        mock_client = MagicMock()
        MockGenAIClient.return_value = mock_client
        mock_client.models.generate_content.return_value = _make_gemini_response(
            "Gemini Mock Response"
        )

        text, latency, usage = call_gemini("Hello")

        self.assertEqual(text, "Gemini Mock Response")
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 0.0)
        self.assertEqual(usage["input_tokens"], 12)
        self.assertEqual(usage["output_tokens"], 25)


class TestCompareModels(unittest.TestCase):

    def test_compare_models_evaluates_correct_costs(self):
        with patch.object(_m, "call_deepseek") as mock_deepseek, \
             patch.object(_m, "call_gemini") as mock_gemini:
            mock_deepseek.return_value = (
                "DeepSeek Response",
                0.5,
                {"input_tokens": 10, "output_tokens": 20},
            )
            mock_gemini.return_value = (
                "Gemini Response",
                0.4,
                {"input_tokens": 10, "output_tokens": 20},
            )

            result = compare_models("Prompt")

            self.assertIn("deepseek", result)
            self.assertIn("gemini_flash", result)

            self.assertAlmostEqual(
                result["deepseek"]["cost"],
                (10 * 0.3 + 20 * 1.2) / 1_000_000,
                places=8,
            )
            self.assertAlmostEqual(result["gemini_flash"]["cost"], 0.00000675, places=8)


class TestRetryWithBackoff(unittest.TestCase):

    def test_succeeds_immediately(self):
        val = retry_with_backoff(lambda: "success")
        self.assertEqual(val, "success")

    def test_succeeds_after_retries(self):
        state = {"attempts": 0}

        def flaky_fn():
            state["attempts"] += 1
            if state["attempts"] < 3:
                raise ValueError("Fail")
            return "ok"

        val = retry_with_backoff(flaky_fn, max_retries=3, base_delay=0.001)
        self.assertEqual(val, "ok")
        self.assertEqual(state["attempts"], 3)

    def test_raises_permanent_exception(self):
        def permanent_fail():
            raise RuntimeError("Permanent")

        with self.assertRaises(RuntimeError):
            retry_with_backoff(permanent_fail, max_retries=2, base_delay=0.001)


class TestBatchCompareAndFormat(unittest.TestCase):

    def test_batch_runs_and_formats_table(self):
        with patch.object(_m, "compare_models") as mock_comp:
            def _get_mock(prompt):
                return {
                    "deepseek": {"response": "DeepSeek response content here", "latency": 0.5, "cost": 0.000027, "input_tokens": 10, "output_tokens": 20},
                    "gemini_flash": {"response": "Gemini response content here", "latency": 0.4, "cost": 0.00000675, "input_tokens": 10, "output_tokens": 20}
                }
            mock_comp.side_effect = _get_mock

            results = batch_compare(["Q1", "Q2"])
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["prompt"], "Q1")

            table = format_comparison_table(results)
            self.assertIsInstance(table, str)
            self.assertIn("Prompt", table)
            self.assertIn("DeepSeek-Flash", table)
            self.assertIn("Gemini-Flash", table)
            self.assertIn("Q1", table)


class TestStreamingChatbot(unittest.TestCase):

    @patch("builtins.input", side_effect=["quit"])
    @patch("google.genai.Client")
    def test_exits_cleanly(self, MockClient, mock_input):
        # Ensure it quits cleanly without entering infinite loop
        try:
            streaming_chatbot()
        except SystemExit:
            pass


if __name__ == "__main__":
    unittest.main()
