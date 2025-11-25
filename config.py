"""
Configuration Management for Aerospace Design Assistant

Handles API keys, environment variables, and application settings.
Supports fallback mode for operation without API keys.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str = "openai"  # "openai" or "anthropic"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    anthropic_model: str = "claude-3-sonnet-20240229"
    temperature: float = 0.1
    max_tokens: int = 4096
    
    def __post_init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", self.anthropic_api_key)
    
    @property
    def is_available(self) -> bool:
        """Check if any LLM API is available."""
        return bool(self.openai_api_key or self.anthropic_api_key)
    
    @property
    def active_provider(self) -> Optional[str]:
        """Get the active provider based on available keys."""
        if self.provider == "openai" and self.openai_api_key:
            return "openai"
        elif self.provider == "anthropic" and self.anthropic_api_key:
            return "anthropic"
        elif self.openai_api_key:
            return "openai"
        elif self.anthropic_api_key:
            return "anthropic"
        return None


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"  # "cpu" or "cuda"
    batch_size: int = 32


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    persist_directory: str = "./chroma_db"
    collection_name: str = "aerospace_papers"
    distance_metric: str = "cosine"  # "cosine", "l2", or "ip"


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    min_relevance_score: float = 0.5


@dataclass
class LangSmithConfig:
    """Configuration for LangSmith tracing."""
    enabled: bool = True
    api_key: Optional[str] = None
    project_name: str = "aerospace-design-assistant"
    
    def __post_init__(self):
        self.api_key = os.getenv("LANGSMITH_API_KEY", self.api_key)
        if self.enabled and self.api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            os.environ["LANGSMITH_API_KEY"] = self.api_key
    
    @property
    def is_available(self) -> bool:
        """Check if LangSmith is configured."""
        return bool(self.api_key)


@dataclass
class WorkflowConfig:
    """Configuration for LangGraph workflow."""
    max_iterations: int = 3
    enable_human_in_loop: bool = False
    verbose: bool = True
    timeout_seconds: int = 300


@dataclass
class PathConfig:
    """Configuration for file paths."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")
    papers_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "papers")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output")
    examples_dir: Path = field(default_factory=lambda: Path(__file__).parent / "examples")
    
    def __post_init__(self):
        # Ensure directories exist
        for dir_path in [self.data_dir, self.papers_dir, self.output_dir, self.examples_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration class aggregating all config sections."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    langsmith: LangSmithConfig = field(default_factory=LangSmithConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Fallback mode settings
    fallback_mode: bool = False
    
    def __post_init__(self):
        # Enable fallback mode if no LLM API is available
        if not self.llm.is_available:
            self.fallback_mode = True
            print("⚠️  No LLM API keys found. Running in fallback mode with limited functionality.")
    
    @property
    def is_fully_configured(self) -> bool:
        """Check if all required configurations are set."""
        return self.llm.is_available
    
    def print_status(self):
        """Print configuration status."""
        print("\n" + "=" * 50)
        print("Aerospace Design Assistant - Configuration Status")
        print("=" * 50)
        
        # LLM Status
        if self.llm.is_available:
            print(f"✅ LLM Provider: {self.llm.active_provider}")
        else:
            print("❌ LLM: No API keys configured")
        
        # LangSmith Status
        if self.langsmith.is_available:
            print(f"✅ LangSmith: Enabled ({self.langsmith.project_name})")
        else:
            print("⚠️  LangSmith: Not configured (tracing disabled)")
        
        # Embedding Status
        print(f"✅ Embeddings: {self.embedding.model_name}")
        
        # Vector DB Status
        print(f"✅ Vector DB: ChromaDB ({self.vector_db.persist_directory})")
        
        # Fallback Mode
        if self.fallback_mode:
            print("\n⚠️  FALLBACK MODE ACTIVE")
            print("   The system will use pre-computed results and rule-based logic.")
        
        print("=" * 50 + "\n")


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment."""
    global config
    config = Config()
    return config


# Environment template for .env file
ENV_TEMPLATE = """# Aerospace Design Assistant Environment Configuration
# Copy this file to .env and fill in your API keys

# LLM Provider API Keys (at least one required for full functionality)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# LangSmith for tracing and debugging (optional but recommended)
LANGSMITH_API_KEY=your_langsmith_api_key_here

# Preferred LLM provider: "openai" or "anthropic"
LLM_PROVIDER=openai
"""


def create_env_template():
    """Create a template .env file if it doesn't exist."""
    env_path = Path(__file__).parent / ".env.template"
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(ENV_TEMPLATE)
        print(f"Created environment template at {env_path}")


if __name__ == "__main__":
    # When run directly, print configuration status and create template
    create_env_template()
    config.print_status()
