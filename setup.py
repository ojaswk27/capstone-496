"""Setup configuration for Aerospace Design Assistant"""

from setuptools import setup, find_packages

setup(
    name="aerospace-design-assistant",
    version="1.0.0",
    description="AI-Powered Aerospace Design Assistant using LangGraph",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(include=["graph", "graph.*", "nodes", "nodes.*", "tools", "tools.*", "rag", "rag.*", "utils", "utils.*"]),
    install_requires=[
        "langchain>=0.2.16",
        "langgraph>=0.2.20",
        "langchain-anthropic>=0.1.23",
        "anthropic>=0.34.2",
        "chromadb>=0.5.5",
        "sentence-transformers>=3.0.1",
        "PyPDF2>=3.0.1",
        "pdfplumber>=0.11.2",
        "numpy>=1.26.4",
        "scipy>=1.13.1",
        "pydantic>=2.8.2",
        "python-dotenv>=1.0.1",
        "rich>=13.7.1",
    ],
    extras_require={
        "dev": [
            "langgraph-cli>=0.1.48",
            "ipython>=8.26.0",
            "jupyter>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "aerospace-design=main:main",
        ],
    },
)
