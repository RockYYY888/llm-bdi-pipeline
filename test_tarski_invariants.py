"""
Test Tarski library for computing invariants from PDDL domain
"""
from tarski.io import PDDLReader
import tarski

print(f"Tarski version: {tarski.__version__}")
print()

# Check available modules
print("Checking available Tarski analysis modules:")
try:
    import tarski.analysis
    print("  ✓ tarski.analysis available")
    print(f"    Available: {dir(tarski.analysis)}")
except ImportError as e:
    print(f"  ✗ tarski.analysis not available: {e}")

try:
    from tarski.grounding import LPGroundingStrategy
    print("  ✓ tarski.grounding available")
except ImportError:
    print("  ✗ tarski.grounding not available")

print()

# Read domain and problem (Tarski requires both)
print("Reading PDDL domain and problem...")
reader = PDDLReader(raise_on_error=True)
problem = reader.read_problem(
    'src/domains/blocksworld/domain.pddl',
    'src/domains/blocksworld/minimal_problem.pddl'
)

print(f"Domain loaded: {problem.domain_name}")
print(f"Language: {problem.language}")
print()

# List predicates
print("Predicates in domain:")
for pred in problem.language.predicates:
    if pred.name not in ['=', '!=']:  # Skip built-in predicates
        print(f"  {pred.name}/{pred.arity}")
print()

# List actions
print("Actions in domain:")
for action_name in problem.actions:
    action = problem.actions[action_name]
    print(f"  {action_name}")
    print(f"    Parameters: {action.parameters}")
    print(f"    Preconditions: {action.precondition}")
    effects = action.effects if isinstance(action.effects, list) else []
    print(f"    Effects: {len(effects)}")
print()

# Try different invariant computation approaches
print("Attempting to compute invariants...")
print()

# Approach 1: Try static analysis module
try:
    from tarski.grounding.lp_grounding import ground_problem_schemas_into_plain_operators
    print("✓ LP grounding available - this can help with invariant analysis")
except ImportError:
    print("✗ LP grounding not available")

# Approach 2: Check for TIM (Type Inference Module)
try:
    from tarski.fstrips import TIM
    print("✓ TIM (Type Inference Module) available!")
except ImportError:
    print("✗ TIM not available in this version")

# Approach 3: Manual mutex analysis from actions
print("\n" + "="*80)
print("Manual Mutex Analysis from Actions:")
print("="*80)

mutex_pairs = set()
singleton_preds = {}

for action_name, action in problem.actions.items():
    effects = action.effects if isinstance(action.effects, list) else []

    # Track what gets added/deleted
    adds = []
    deletes = []

    for eff in effects:
        # Tarski uses AddEffect and DelEffect classes
        from tarski.fstrips.fstrips import AddEffect, DelEffect
        if isinstance(eff, DelEffect):
            deletes.append(str(eff.atom.predicate))
        elif isinstance(eff, AddEffect):
            adds.append(str(eff.atom.predicate))

    # Find add/delete pairs
    for add in adds:
        for delete in deletes:
            if add != delete:
                pair = tuple(sorted([add, delete]))
                mutex_pairs.add(pair)

    # Count predicate occurrences
    for add in adds:
        singleton_preds[add] = singleton_preds.get(add, 0) + 1

print(f"\nMutex pairs (predicates that are add/deleted together):")
for p1, p2 in sorted(mutex_pairs):
    print(f"  Exclusive({p1}, {p2})")

print(f"\nSingleton predicates (appear at most once per action):")
for pred, max_count in sorted(singleton_preds.items()):
    if max_count == 1:
        print(f"  At-most-one {pred}(...)")

print("\n" + "="*80)
