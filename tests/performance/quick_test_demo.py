#!/usr/bin/env python
"""Quick demo to show where results are saved."""
import json
from pathlib import Path
from datetime import datetime

# Show existing results
results_dir = Path("tests/integration/results")
print(f"Results directory: {results_dir.absolute()}")
print("\nExisting result files:")

if results_dir.exists():
    json_files = list(results_dir.glob("*.json"))
    if json_files:
        for f in sorted(json_files):
            print(f"  - {f.name}")
            # Show file size
            size = f.stat().st_size
            print(f"    Size: {size} bytes")
            # Show if it has actual results
            with open(f, 'r') as file:
                data = json.load(file)
                models_tested = data.get('models_tested', [])
                if models_tested:
                    print(f"    Models tested: {', '.join(models_tested)}")
                else:
                    print(f"    No successful model tests (empty results)")
    else:
        print("  No result files found")
else:
    print("  Results directory doesn't exist")

print("\n" + "="*60)
print("To generate results with actual data:")
print("1. For quick test: uv run test_current_models.py")
print("2. For visualization: uv run visualize_results.py <json_file>")
print("3. Results are saved in: tests/integration/results/")
print("="*60)