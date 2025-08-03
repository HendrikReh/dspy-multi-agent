#!/usr/bin/env python
"""Test runner script for DSPy multi-agent system."""
import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run the test suite."""
    print("Running DSPy Multi-Agent Tests...")
    print("=" * 60)
    
    # Run unit tests
    cmd = [sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short"]
    
    # Add coverage if available
    try:
        import coverage
        cmd = [sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short", "--cov=src", "--cov-report=term-missing"]
    except ImportError:
        print("Note: Install pytest-cov for coverage reports")
    
    result = subprocess.run(cmd)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())