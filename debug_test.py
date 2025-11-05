#!/usr/bin/env python
"""Debug script to find where test hangs"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests/stage1_interpretation')

print("Step 1: Basic imports...")
from pathlib import Path
print("  ✓ pathlib imported")

print("Step 2: Import test module...")
import test_nl_to_ltlf_generation as test_mod
print("  ✓ test module imported")

print("Step 3: Load CSV...")
csv_path = Path('tests/stage1_interpretation/nl_to_ltlf_test_cases.csv')
test_cases = test_mod.load_test_cases(csv_path)
print(f"  ✓ Loaded {len(test_cases)} test cases")

print("Step 4: Import generator...")
from stage1_interpretation.ltlf_generator import NLToLTLfGenerator
print("  ✓ NLToLTLfGenerator imported")

print("Step 5: Import config...")
from config import get_config
print("  ✓ config imported")

print("Step 6: Get config...")
config = get_config()
print(f"  ✓ Config loaded (model: {config.openai_model})")

print("Step 7: Set up domain file...")
domain_file = str(Path('src/legacy/fond/domains/blocksworld/domain.pddl'))
print(f"  ✓ Domain file: {domain_file}")

print("Step 8: Create generator (THIS IS WHERE IT MIGHT HANG)...")
sys.stdout.flush()
generator = NLToLTLfGenerator(
    api_key=config.openai_api_key,
    model=config.openai_model,
    domain_file=domain_file
)
print("  ✓ Generator created!")

print("\n✓ All steps completed successfully - no hang detected!")
