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


class TestUnderstandAgent(unittest.TestCase):
    @patch("agents.understand.OllamaClient")
    def test_classifies_drone(self, MockClient):
        instance = MockClient.return_value
        instance.chat_json.return_value = {
            "vehicle_type": "drone",
            "payload_kg": 2.0,
            "endurance_hours": 0.5,
            "range_km": None,
            "speed_kmh": None,
            "altitude_m": None,
            "mission_type": "surveillance",
            "reasoning": "User wants a drone with 2kg payload"
        }
        from agents.understand import understand_agent
        from graph.state import create_initial_state
        state = create_initial_state("surveillance drone, 2kg payload, 30min flight")
        result = understand_agent(state)
        assert result.vehicle_type == "drone"
        assert result.requirements.payload_kg == 2.0
        assert result.requirements.endurance_hours == 0.5
        assert result.phase == "parameterizing"

    @patch("agents.understand.OllamaClient")
    def test_handles_llm_failure(self, MockClient):
        instance = MockClient.return_value
        instance.chat_json.return_value = None
        from agents.understand import understand_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone 2kg")
        result = understand_agent(state)
        assert result.phase == "error"
        assert len(result.errors) > 0


class TestParameterAgent(unittest.TestCase):
    @patch("agents.parameter.OllamaClient")
    def test_fills_missing_params(self, MockClient):
        instance = MockClient.return_value
        instance.chat_json.return_value = {
            "payload_kg": 2.0,
            "endurance_hours": 0.5,
            "range_km": 30.0,
            "speed_kmh": 60.0,
            "altitude_m": 500.0,
            "mission_type": "surveillance",
            "vehicle_specific": {"num_motors": 4, "application": "photography"},
            "reasoning": "Small surveillance drone"
        }
        from agents.parameter import parameter_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone 2kg payload 30min")
        state.vehicle_type = "drone"
        state.requirements.payload_kg = 2.0
        state.requirements.endurance_hours = 0.5
        result = parameter_agent(state)
        assert result.requirements.range_km == 30.0
        assert result.requirements.speed_kmh == 60.0
        assert result.phase == "designing"

    @patch("agents.parameter.OllamaClient")
    def test_preserves_existing_params(self, MockClient):
        instance = MockClient.return_value
        instance.chat_json.return_value = {
            "payload_kg": 2.0,
            "endurance_hours": 0.5,
            "range_km": 30.0,
            "speed_kmh": 60.0,
            "altitude_m": 500.0,
            "mission_type": "surveillance",
            "vehicle_specific": {},
            "reasoning": "test"
        }
        from agents.parameter import parameter_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone")
        state.vehicle_type = "drone"
        state.requirements.payload_kg = 5.0  # User specified
        result = parameter_agent(state)
        # LLM returned 2.0 but user specified 5.0 — user value should win
        assert result.requirements.payload_kg == 5.0


class TestDesignAgent(unittest.TestCase):
    @patch("agents.design.OllamaClient")
    def test_calls_tool_and_stores_result(self, MockClient):
        from llm.client import ToolCall, ToolResponse

        instance = MockClient.return_value
        instance.chat_with_tools.side_effect = [
            ToolResponse(
                message="",
                tool_calls=[ToolCall(
                    id="abc",
                    name="size_drone",
                    arguments={"payload_kg": 0.5, "flight_time_minutes": 20.0}
                )],
                raw_response={},
            ),
            ToolResponse(
                message="Design complete.",
                tool_calls=[],
                raw_response={},
            ),
        ]

        from agents.design import design_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone 0.5kg 20min")
        state.vehicle_type = "drone"
        state.requirements.payload_kg = 0.5
        state.requirements.endurance_hours = 20 / 60
        state.phase = "designing"
        result = design_agent(state)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "size_drone"
        assert result.tool_calls[0].success is True
        assert "design" in result.intermediate_results

    @patch("agents.design.OllamaClient")
    def test_handles_bad_tool_name(self, MockClient):
        from llm.client import ToolCall, ToolResponse

        instance = MockClient.return_value
        instance.chat_with_tools.side_effect = [
            ToolResponse(
                message="",
                tool_calls=[ToolCall(id="x", name="nonexistent_tool", arguments={})],
                raw_response={},
            ),
            ToolResponse(message="Could not complete.", tool_calls=[], raw_response={}),
        ]

        from agents.design import design_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone")
        state.vehicle_type = "drone"
        state.phase = "designing"
        result = design_agent(state)
        assert any(not tc.success for tc in result.tool_calls)


class TestValidateAgent(unittest.TestCase):
    @patch("agents.validate.OllamaClient")
    def test_passes_good_design(self, MockClient):
        instance = MockClient.return_value
        instance.chat_json.return_value = {
            "passed": True,
            "checks": {"payload": True, "endurance": True},
            "warnings": [],
            "errors": [],
            "feedback": ""
        }
        from agents.validate import validate_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone 0.5kg 20min")
        state.vehicle_type = "drone"
        state.requirements.payload_kg = 0.5
        state.intermediate_results["design"] = {"total_weight": 1.5, "hover_time": 22}
        result = validate_agent(state)
        assert result.validation_result.passed is True
        assert result.phase == "synthesizing"

    @patch("agents.validate.OllamaClient")
    def test_fails_bad_design(self, MockClient):
        instance = MockClient.return_value
        instance.chat_json.return_value = {
            "passed": False,
            "checks": {"payload": True, "endurance": False},
            "warnings": ["Low flight time"],
            "errors": ["Flight time 10min below target 20min"],
            "feedback": "Increase battery capacity or reduce weight"
        }
        from agents.validate import validate_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone 0.5kg 20min")
        state.vehicle_type = "drone"
        state.intermediate_results["design"] = {"total_weight": 2.0, "hover_time": 10}
        result = validate_agent(state)
        assert result.validation_result.passed is False
        assert result.validation_feedback is not None
        assert result.phase == "validating"


class TestWorkflow(unittest.TestCase):
    def test_graph_builds(self):
        from graph.workflow import build_design_graph
        graph = build_design_graph()
        assert graph is not None

    def test_synthesize_formats_output(self):
        from graph.workflow import synthesize_output
        from graph.state import create_initial_state, DesignPhase, ValidationResult
        state = create_initial_state("drone 0.5kg")
        state.vehicle_type = "drone"
        state.requirements.payload_kg = 0.5
        state.classification_confidence = 0.9
        state.intermediate_results["design"] = {
            "total_weight": 1.5,
            "hover_time": 22,
            "frame_size": 350,
        }
        state.validation_result = ValidationResult(passed=True)
        result = synthesize_output(state)
        assert result.design_output is not None
        assert result.design_output.vehicle_type == "drone"
        assert result.phase == "complete"


if __name__ == "__main__":
    unittest.main()
