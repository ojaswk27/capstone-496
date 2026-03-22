import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock

class TestConfig(unittest.TestCase):
    def test_default_config(self):
        from config import get_config
        cfg = get_config()
        assert cfg.llm.ollama_base_url == "http://localhost:11434"
        assert cfg.llm.temperature == 0.1
        assert cfg.llm.max_retries == 2
        assert cfg.llm.max_tool_calls == 5
        assert cfg.llm.max_validation_retries == 2

    def test_paths_exist(self):
        from config import get_config
        cfg = get_config()
        assert cfg.paths.base_dir.exists()

class TestOllamaClient(unittest.TestCase):
    def test_extract_json_clean(self):
        from llm.client import OllamaClient
        client = OllamaClient.__new__(OllamaClient)
        result = client.extract_json('{"vehicle": "drone", "payload_kg": 2.0}')
        assert result == {"vehicle": "drone", "payload_kg": 2.0}

    def test_extract_json_markdown_wrapped(self):
        from llm.client import OllamaClient
        client = OllamaClient.__new__(OllamaClient)
        text = '```json\n{"vehicle": "drone"}\n```'
        result = client.extract_json(text)
        assert result == {"vehicle": "drone"}

    def test_extract_json_with_surrounding_text(self):
        from llm.client import OllamaClient
        client = OllamaClient.__new__(OllamaClient)
        text = 'Here is the result:\n{"vehicle": "drone"}\nDone.'
        result = client.extract_json(text)
        assert result == {"vehicle": "drone"}

    def test_extract_json_invalid_returns_none(self):
        from llm.client import OllamaClient
        client = OllamaClient.__new__(OllamaClient)
        result = client.extract_json("not json at all")
        assert result is None

    @patch("llm.client.ollama")
    def test_chat_calls_ollama(self, mock_ollama):
        mock_client_instance = MagicMock()
        mock_client_instance.chat.return_value = {
            "message": {"role": "assistant", "content": "Hello"}
        }
        mock_ollama.Client.return_value = mock_client_instance
        from llm.client import OllamaClient
        client = OllamaClient()
        result = client.chat("Say hello", system_prompt="Be friendly")
        assert result == "Hello"
        mock_client_instance.chat.assert_called_once()

    @patch("llm.client.ollama")
    def test_chat_json_retries_on_bad_output(self, mock_ollama):
        mock_client_instance = MagicMock()
        mock_client_instance.chat.side_effect = [
            {"message": {"role": "assistant", "content": "not json at all"}},
            {"message": {"role": "assistant", "content": '{"vehicle": "drone"}'}},
        ]
        mock_ollama.Client.return_value = mock_client_instance
        from llm.client import OllamaClient
        client = OllamaClient()
        result = client.chat_json("classify this")
        assert result == {"vehicle": "drone"}
        assert mock_client_instance.chat.call_count == 2

    @patch("llm.client.ollama")
    def test_chat_with_tools_parses_tool_calls(self, mock_ollama):
        mock_client_instance = MagicMock()
        mock_client_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "size_drone",
                            "arguments": {"payload_kg": 2.0, "flight_time_minutes": 30},
                        }
                    }
                ],
            }
        }
        mock_ollama.Client.return_value = mock_client_instance
        from llm.client import OllamaClient, ToolResponse
        client = OllamaClient()
        resp = client.chat_with_tools("Design a drone", tools=[])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "size_drone"
        assert resp.tool_calls[0].arguments == {"payload_kg": 2.0, "flight_time_minutes": 30}


if __name__ == "__main__":
    unittest.main()
