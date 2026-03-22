import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

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

if __name__ == "__main__":
    unittest.main()
