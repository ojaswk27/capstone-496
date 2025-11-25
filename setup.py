"""
MAT496 Capstone Project
AI-Powered Aerospace Design Assistant

Setup configuration for package installation.
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read() if f else ""

setup(
    name="aerospace_design_assistant",
    version="1.0.0",
    author="MAT496 Student",
    description="AI-Powered Aerospace Design Assistant using LangGraph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "aerospace-design=main:app",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
