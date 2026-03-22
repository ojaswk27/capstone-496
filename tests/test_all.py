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


class TestToolSchemas(unittest.TestCase):
    def test_generate_schema_for_size_drone(self):
        from llm.tools import generate_tool_schema
        from tools.drone_tools import size_drone
        schema = generate_tool_schema(
            size_drone,
            description="Complete drone sizing from requirements"
        )
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "size_drone"
        props = schema["function"]["parameters"]["properties"]
        assert "payload_kg" in props
        assert props["payload_kg"]["type"] == "number"
        assert "flight_time_minutes" in props
        required = schema["function"]["parameters"]["required"]
        assert "payload_kg" in required
        assert "flight_time_minutes" in required
        # Optional params should NOT be in required
        assert "num_motors" not in required

    def test_get_tools_for_vehicle(self):
        from llm.tools import get_tools_for_vehicle_type
        tools = get_tools_for_vehicle_type("drone")
        names = [t["function"]["name"] for t in tools]
        assert "size_drone" in names

    def test_all_vehicle_types_have_tools(self):
        from llm.tools import get_tools_for_vehicle_type
        for vtype in ["drone", "fixed_wing", "helicopter", "rocket", "satellite", "glider"]:
            tools = get_tools_for_vehicle_type(vtype)
            assert len(tools) > 0, f"No tools for {vtype}"

    def test_validate_tool_args_coerces_types(self):
        from llm.tools import validate_tool_args
        # 9B models often return strings instead of floats
        result = validate_tool_args("size_drone", {
            "payload_kg": "2.0",
            "flight_time_minutes": "30"
        })
        assert isinstance(result["payload_kg"], float)
        assert isinstance(result["flight_time_minutes"], float)
        assert result["payload_kg"] == 2.0

    def test_validate_tool_args_rejects_unknown_tool(self):
        from llm.tools import validate_tool_args
        with self.assertRaises(ValueError):
            validate_tool_args("nonexistent_tool", {})


class TestDesignState(unittest.TestCase):
    def test_create_initial_state(self):
        from graph.state import create_initial_state
        state = create_initial_state("design a drone with 2kg payload")
        assert state.raw_input == "design a drone with 2kg payload"
        assert state.phase == "understanding"
        assert state.vehicle_type == "unknown"
        assert state.retry_count == 0

    def test_state_has_no_rag_fields(self):
        from graph.state import DesignState
        fields = set(DesignState.model_fields.keys())
        assert "search_queries" not in fields
        assert "search_results" not in fields
        assert "extracted_formulas" not in fields
        assert "extracted_data" not in fields

    def test_state_has_new_fields(self):
        from graph.state import DesignState
        fields = set(DesignState.model_fields.keys())
        assert "agent_messages" in fields
        assert "tool_calls" in fields
        assert "retry_count" in fields
        assert "validation_feedback" in fields

    def test_vehicle_type_is_string(self):
        from graph.state import create_initial_state
        state = create_initial_state("drone")
        # With use_enum_values=True, vehicle_type should be a string
        assert isinstance(state.vehicle_type, str)


if __name__ == "__main__":
    unittest.main()
