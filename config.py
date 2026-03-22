"""
Configuration for Aerospace Design Assistant.
Simplified for Ollama-based LLM inference.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """Ollama LLM configuration."""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:latest"
    temperature: float = 0.1
    max_retries: int = 2
    max_tool_calls: int = 5
    max_validation_retries: int = 2

    def __post_init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", self.ollama_base_url)
        self.ollama_model = os.getenv("OLLAMA_MODEL", self.ollama_model)


@dataclass
class WorkflowConfig:
    """LangGraph workflow configuration."""
    verbose: bool = True
    timeout_seconds: int = 300


@dataclass
class PathConfig:
    """File path configuration."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output")
    examples_dir: Path = field(default_factory=lambda: Path(__file__).parent / "examples")

    def __post_init__(self):
        for dir_path in [self.output_dir, self.examples_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def print_status(self):
        print("\n" + "=" * 50)
        print("Aerospace Design Assistant - Configuration")
        print("=" * 50)
        print(f"  Ollama URL: {self.llm.ollama_base_url}")
        print(f"  Model: {self.llm.ollama_model}")
        print(f"  Temperature: {self.llm.temperature}")
        print("=" * 50 + "\n")


_config = Config()

def get_config() -> Config:
    return _config

def reload_config() -> Config:
    global _config
    _config = Config()
    return _config
