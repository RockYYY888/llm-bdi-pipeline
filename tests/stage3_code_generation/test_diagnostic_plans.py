"""
Diagnostic test to understand why goal plans are not generated
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.stage3_code_generation.forward_planner import ForwardStatePlanner
from src.stage3_code_generation.agentspeak_codegen import AgentSpeakCodeGenerator
from src.stage3_code_generation.state_space import PredicateAtom
from src.utils.pddl_parser import PDDLParser

print("="*80)
print("DIAGNOSTIC: Goal Plan Generation")
print("="*80)

# Load domain
domain_path = project_root / 'src' / 'legacy' / 'fond' / 'domains' / 'blocksworld' / 'domain.pddl'
domain = PDDLParser.parse_domain(str(domain_path))
objects = ['a', 'b']

# Create goal
goal_preds = [PredicateAtom('on', ('a', 'b'))]
goal_name = "on(a, b)"

print(f"\nGoal: {goal_name}")
print(f"Goal predicates: {[str(p) for p in goal_preds]}")

# Run forward planner
planner = ForwardStatePlanner(domain, objects)
graph = planner.explore_from_goal(goal_preds, max_depth=2)

print(f"\n=== State Graph ===")
print(f"States: {len(graph.states)}")
print(f"Transitions: {len(graph.transitions)}")
print(f"Goal state: {sorted([str(p) for p in graph.goal_state.predicates])}")

# Find paths
paths = graph.find_shortest_paths_to_goal()
print(f"\n=== Path Finding ===")
print(f"Total paths: {len(paths)}")

non_trivial = [(s, p) for s, p in paths.items() if s != graph.goal_state and p]
print(f"Non-trivial paths: {len(non_trivial)}")

# Show some example paths
if non_trivial:
    print(f"\nExample non-trivial paths:")
    for i, (state, path) in enumerate(non_trivial[:3]):
        preds = ', '.join(str(p) for p in sorted(state.predicates, key=str)[:3])
        print(f"  {i+1}. State: {preds}...")
        print(f"     Path: {len(path)} steps → goal")

# Generate AgentSpeak code
print(f"\n=== AgentSpeak Code Generation ===")
codegen = AgentSpeakCodeGenerator(
    state_graph=graph,
    goal_name=goal_name,
    domain=domain,
    objects=objects
)

asl_code = codegen.generate()

# Analyze generated code
print(f"Code length: {len(asl_code)} characters")
print(f"Lines: {len(asl_code.split(chr(10)))}")

# Count plans
goal_plan_count = asl_code.count(f"+!{goal_name}")
action_plan_count = asl_code.count("+!") - goal_plan_count
print(f"Goal plans generated: {goal_plan_count}")
print(f"Action plans generated: {action_plan_count}")

# Extract goal plans section
if "Goal Achievement Plans" in asl_code:
    start = asl_code.index("Goal Achievement Plans")
    end = asl_code.find("/*", start + 30)
    if end == -1:
        end = len(asl_code)
    goal_section = asl_code[start:end]

    print(f"\n=== Goal Plans Section ===")
    print(goal_section[:500])
else:
    print("\n⚠️ No 'Goal Achievement Plans' section found")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
