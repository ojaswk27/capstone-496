#!/usr/bin/env python3
"""
Verification script to check if you've updated all the necessary files.
Run this to see what still needs to be done.
"""

import os
import sys


def check_file_updated(filepath, search_string):
    """Check if a file contains a specific string (indicating it's been updated)."""
    if not os.path.exists(filepath):
        return False, "File not found"

    try:
        with open(filepath, 'r') as f:
            content = f.read()
            if search_string in content:
                return True, "✅ Updated"
            else:
                return False, "❌ OLD VERSION (needs update)"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    print("=" * 60)
    print("FILE UPDATE VERIFICATION")
    print("=" * 60)

    checks = [
        {
            "file": "rag/extractor.py",
            "search": "desired_output: str = None",
            "description": "Has desired_output parameter"
        },
        {
            "file": "rag/extractor.py",
            "search": "PREFER THE SIMPLEST, MOST DIRECT formula",
            "description": "Has emphatic simplicity instruction"
        },
        {
            "file": "tools/aerospace_tools.py",
            "search": "desired_output = None",
            "description": "Detects output variable from query"
        },
        {
            "file": "nodes/workflow_nodes.py",
            "search": "_clean_json_text",
            "description": "Has JSON cleaning function"
        },
        {
            "file": "nodes/workflow_nodes.py",
            "search": "mass",
            "description": "Uses mass-based inputs for drone calc"
        },
        {
            "file": "ingest_data.py",
            "search": "--clear",
            "description": "Has command-line flags"
        },
        {
            "file": "data/papers/drones/basic_hover_thrust.txt",
            "search": "T = W = mass * g",
            "description": "Fundamental hover equation present"
        }
    ]

    all_good = True

    for check in checks:
        filepath = check["file"]
        search_str = check["search"]
        desc = check["description"]

        found, status = check_file_updated(filepath, search_str)

        symbol = "✅" if found else "❌"
        print(f"\n{symbol} {filepath}")
        print(f"   Check: {desc}")
        print(f"   Status: {status}")

        if not found:
            all_good = False

    print("\n" + "=" * 60)
    if all_good:
        print("✅ ALL FILES UPDATED! You're good to go!")
    else:
        print("❌ SOME FILES NEED UPDATING")
        print("\nTo fix:")
        print("1. Copy files from /mnt/user-data/outputs/ to your project")
        print("2. Make sure file paths match your project structure")
        print("3. Re-run this script to verify")
    print("=" * 60)


if __name__ == "__main__":
    main()