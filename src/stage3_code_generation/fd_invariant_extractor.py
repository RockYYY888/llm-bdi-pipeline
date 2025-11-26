"""
Fast Downward Invariant Extractor

Uses Fast Downward's invariant synthesis to extract static mutex groups
from PDDL domains. These are true domain invariants (not transient effect-based mutex).

References:
- Helmert 2009 - The Fast Downward Planning System
- Fox & Long 1998 - Automatic Invariant Synthesis (TIM)
"""

import subprocess
import re
import tempfile
import os
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class LiftedMutexPattern:
    """
    Represents a lifted mutex constraint between two predicates.

    Two ground predicates are mutex if they match this pattern:
    1. Predicate names match pred1_name and pred2_name (or vice versa)
    2. Arguments at shared_positions are EQUAL
    3. Arguments at different_positions are DIFFERENT

    Example: holding(X) ∧ on(X, Y) is impossible
        pred1_name="holding", pred1_arity=1
        pred2_name="on", pred2_arity=2
        shared_positions=((0, 0),)  # first arg of both must be same
        different_positions=()

    Example: on(X, Y) ∧ on(X, Z) where Y≠Z is impossible
        pred1_name="on", pred1_arity=2
        pred2_name="on", pred2_arity=2
        shared_positions=((0, 0),)       # first arg same
        different_positions=((1, 1),)    # second arg different
    """
    pred1_name: str
    pred1_arity: int
    pred2_name: str
    pred2_arity: int
    shared_positions: Tuple[Tuple[int, int], ...]
    different_positions: Tuple[Tuple[int, int], ...]

    def matches(self, atom1_name: str, atom1_args: Tuple[str, ...],
                atom2_name: str, atom2_args: Tuple[str, ...]) -> bool:
        """
        Check if two ground atoms match this mutex pattern.

        Args:
            atom1_name: First predicate name
            atom1_args: First predicate arguments
            atom2_name: Second predicate name
            atom2_args: Second predicate arguments

        Returns:
            True if the atoms are mutex according to this pattern
        """
        # Check predicate names match (in either order)
        if (atom1_name == self.pred1_name and atom2_name == self.pred2_name):
            a1_args, a2_args = atom1_args, atom2_args
        elif (atom1_name == self.pred2_name and atom2_name == self.pred1_name):
            a1_args, a2_args = atom2_args, atom1_args
        else:
            return False

        # Check arities
        if len(a1_args) != self.pred1_arity or len(a2_args) != self.pred2_arity:
            return False

        # Check shared positions have EQUAL values
        for pos1, pos2 in self.shared_positions:
            if a1_args[pos1] != a2_args[pos2]:
                return False

        # Check different positions have DIFFERENT values
        for pos1, pos2 in self.different_positions:
            if a1_args[pos1] == a2_args[pos2]:
                return False  # Should be different but aren't

        return True

    def __hash__(self):
        # Normalize order for hashing (smaller pred name first)
        if self.pred1_name <= self.pred2_name:
            return hash((self.pred1_name, self.pred1_arity,
                        self.pred2_name, self.pred2_arity,
                        self.shared_positions, self.different_positions))
        else:
            # Swap and also swap position indices
            swapped_shared = tuple((p2, p1) for p1, p2 in self.shared_positions)
            swapped_diff = tuple((p2, p1) for p1, p2 in self.different_positions)
            return hash((self.pred2_name, self.pred2_arity,
                        self.pred1_name, self.pred1_arity,
                        swapped_shared, swapped_diff))

    def __eq__(self, other):
        if not isinstance(other, LiftedMutexPattern):
            return False
        # Check equality in both orders
        direct = (self.pred1_name == other.pred1_name and
                 self.pred1_arity == other.pred1_arity and
                 self.pred2_name == other.pred2_name and
                 self.pred2_arity == other.pred2_arity and
                 set(self.shared_positions) == set(other.shared_positions) and
                 set(self.different_positions) == set(other.different_positions))
        if direct:
            return True
        # Check swapped order
        swapped_shared = set((p2, p1) for p1, p2 in self.shared_positions)
        swapped_diff = set((p2, p1) for p1, p2 in self.different_positions)
        return (self.pred1_name == other.pred2_name and
               self.pred1_arity == other.pred2_arity and
               self.pred2_name == other.pred1_name and
               self.pred2_arity == other.pred1_arity and
               swapped_shared == set(other.shared_positions) and
               swapped_diff == set(other.different_positions))


class FDInvariantExtractor:
    """
    Extracts static mutex groups from PDDL domain using Fast Downward
    """

    def __init__(self, domain_path: str, objects: List[str]):
        """
        Initialize extractor

        Args:
            domain_path: Path to PDDL domain file
            objects: List of object names for grounding (e.g., ['b1', 'b2', 'b3'])
        """
        # Convert to absolute path to ensure FD can find the file
        self.domain_path = str(Path(domain_path).resolve())
        self.objects = objects

    def extract_invariants(self) -> Tuple[Set[str], Set['LiftedMutexPattern']]:
        """
        Extract static mutex groups using Fast Downward

        Returns:
            (singleton_predicates, lifted_mutex_patterns)
            - singleton_predicates: {pred_name1, pred_name2, ...}
            - lifted_mutex_patterns: Set[LiftedMutexPattern] - lifted constraints for precise mutex checking

        Raises:
            SystemExit: If Fast Downward is not available or extraction fails
        """
        # Check if Fast Downward is available
        if not self._check_fd_available():
            print("\n" + "="*80)
            print("[FATAL ERROR] Fast Downward not available")
            print("="*80)
            print("\nFast Downward is required for static mutex analysis.")
            print("\nPlease install Fast Downward:")
            print("  1. cd /Users/lyw/Desktop/FYP/llm-bdi-pipeline-dev")
            print("  2. git clone https://github.com/aibasel/downward.git fast-downward")
            print("  3. cd fast-downward")
            print("  4. ./build.py")
            print("\nOr ensure fast-downward.py is in your PATH")
            print("="*80)
            import sys
            sys.exit(1)

        # Create mock problem file
        problem_path = self._create_mock_problem()

        try:
            # Run Fast Downward translator
            sas_output = self._run_fd_translator(problem_path)

            # Parse SAS output to extract lifted patterns
            singleton_predicates, lifted_patterns = self._parse_sas_mutex_groups(sas_output)

            return singleton_predicates, lifted_patterns

        except Exception as e:
            print("\n" + "="*80)
            print("[FATAL ERROR] Fast Downward invariant extraction failed")
            print("="*80)
            print(f"Error: {e}")
            print(f"\nDomain path: {self.domain_path}")
            print(f"Objects: {self.objects}")
            print(f"Problem path: {problem_path}")
            print("\nPlease check:")
            print("  1. Domain file is valid PDDL")
            print("  2. Fast Downward is properly installed")
            print("  3. Fast Downward can process this domain")
            print("="*80)
            import traceback
            traceback.print_exc()
            import sys
            sys.exit(1)

        finally:
            # Cleanup mock problem file
            if os.path.exists(problem_path):
                os.remove(problem_path)

    def _check_fd_available(self) -> bool:
        """Check if Fast Downward is available"""
        # Check if fast-downward directory exists in project root
        fd_dir = Path(__file__).parent.parent.parent / 'fast-downward'
        if fd_dir.exists():
            fd_script = fd_dir / 'fast-downward.py'
            if fd_script.exists():
                return True

        # Check if fast-downward.py is in PATH
        try:
            result = subprocess.run(
                ['fast-downward.py', '--help'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _create_mock_problem(self) -> str:
        """
        Create mock problem file for invariant extraction

        Extracts domain name from domain file. Uses blocksworld-like structure
        for init state as this is the current target domain.

        Returns:
            Path to created problem file
        """
        # Extract domain name from domain file
        domain_name = "blocksworld"  # Default for current use case
        try:
            with open(self.domain_path, 'r') as f:
                content = f.read()
                # Look for (define (domain <name>)
                import re
                match = re.search(r'\(define\s+\(domain\s+(\w+)\)', content)
                if match:
                    domain_name = match.group(1)
        except Exception:
            # Use default if parsing fails
            pass

        # Generate problem with all objects
        # Note: For blocksworld we assume block type, for generic domains this may need adjustment
        objects_str = ' '.join(f"{obj} - block" for obj in self.objects)

        # Create simple initial state (all blocks on table, all clear, hand empty)
        # This works for blocksworld and provides enough state for FD to extract invariants
        init_facts = []
        for obj in self.objects:
            init_facts.append(f"    (ontable {obj})")
            init_facts.append(f"    (clear {obj})")
        init_facts.append("    (handempty)")

        # Use a simple goal to force FD to generate invariants
        # Goal: stack first block on second block
        goal = f"(on {self.objects[0]} {self.objects[1]})" if len(self.objects) >= 2 else "(handempty)"

        problem_content = f"""(define (problem mock-invariant-extraction)
  (:domain {domain_name})
  (:objects {objects_str})
  (:init
{chr(10).join(init_facts)}
  )
  (:goal (and {goal}))
)
"""

        # Write to temp file
        temp_dir = tempfile.gettempdir()
        problem_path = os.path.join(temp_dir, 'fd_mock_problem.pddl')

        with open(problem_path, 'w') as f:
            f.write(problem_content)

        return problem_path

    def _run_fd_translator(self, problem_path: str) -> str:
        """
        Run Fast Downward translator to generate SAS output

        Args:
            problem_path: Path to problem file

        Returns:
            Content of output.sas file
        """
        # Find Fast Downward script
        fd_dir = Path(__file__).parent.parent.parent / 'fast-downward'
        if fd_dir.exists():
            fd_script = str(fd_dir / 'fast-downward.py')
        else:
            fd_script = 'fast-downward.py'  # Assume in PATH

        # Create temp directory for FD output
        temp_dir = tempfile.mkdtemp()

        try:
            # Run FD translator with --translate option
            cmd = [
                fd_script,
                '--translate',
                self.domain_path,
                problem_path,
                '--sas-file', os.path.join(temp_dir, 'output.sas')
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=60,
                cwd=temp_dir
            )

            if result.returncode != 0:
                stderr = result.stderr.decode() if result.stderr else ''
                stdout = result.stdout.decode() if result.stdout else ''
                raise RuntimeError(f"Fast Downward translator failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}")

            # Read SAS output
            sas_path = os.path.join(temp_dir, 'output.sas')
            if not os.path.exists(sas_path):
                raise RuntimeError("SAS output file not generated")

            with open(sas_path, 'r') as f:
                return f.read()

        finally:
            # Cleanup temp directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _parse_sas_mutex_groups(self, sas_content: str) -> Tuple[Set[str], Set[LiftedMutexPattern]]:
        """
        Parse SAS output to extract lifted mutex patterns.

        Extracts TWO types of constraints:
        1. INTRA-VAR MUTEX: From SAS variable definitions (atoms within same var are mutex)
        2. CROSS-VAR MUTEX: From mutex_group sections (atoms across vars are mutex)

        Args:
            sas_content: Content of output.sas file

        Returns:
            (singleton_predicates, lifted_mutex_patterns)
        """
        singleton_predicates = set()
        lifted_patterns = set()

        # Step 1: Parse variable definitions to extract FULL atom info (name + args)
        # var_atoms[var_id] = [(pred_name, (arg1, arg2, ...)), ...]
        var_atoms: Dict[int, List[Tuple[str, Tuple[str, ...]]]] = {}

        var_pattern = re.compile(
            r'begin_variable\s+var(\d+)\s+(-?\d+)\s+(\d+)\s+(.*?)\s+end_variable',
            re.DOTALL
        )

        for match in var_pattern.finditer(sas_content):
            var_id = int(match.group(1))
            atoms_str = match.group(4)

            var_atoms[var_id] = []

            for line in atoms_str.strip().split('\n'):
                line = line.strip()
                if line.startswith('Atom '):
                    # Parse: "Atom predname(arg1, arg2, ...)" or "Atom predname()"
                    atom_match = re.match(r'Atom\s+(\w+)\(([^)]*)\)', line)
                    if atom_match:
                        pred_name = atom_match.group(1)
                        args_str = atom_match.group(2).strip()
                        if args_str:
                            # Split by comma and strip whitespace
                            args = tuple(a.strip() for a in args_str.split(','))
                        else:
                            args = ()
                        var_atoms[var_id].append((pred_name, args))

        # Step 2: Extract INTRA-VAR lifted mutex patterns
        # For each pair of atoms within a var, they are mutex
        for var_id, atoms in var_atoms.items():
            for (name1, args1), (name2, args2) in combinations(atoms, 2):
                pattern = self._create_lifted_pattern(name1, args1, name2, args2)
                if pattern:
                    lifted_patterns.add(pattern)

        # Step 3: Parse mutex groups for CROSS-VAR mutex
        mutex_pattern = re.compile(
            r'begin_mutex_group\s+(\d+)\s+((?:\d+\s+\d+\s*\n?)+)end_mutex_group',
            re.DOTALL
        )

        for match in mutex_pattern.finditer(sas_content):
            entries_str = match.group(2)

            # Collect atoms in this mutex group
            group_atoms: List[Tuple[str, Tuple[str, ...]]] = []

            for line in entries_str.strip().split('\n'):
                parts = line.strip().split()
                if len(parts) == 2:
                    var_id = int(parts[0])
                    value_id = int(parts[1])

                    if var_id in var_atoms and value_id < len(var_atoms[var_id]):
                        group_atoms.append(var_atoms[var_id][value_id])

            # Extract lifted patterns for all pairs in mutex group
            for (name1, args1), (name2, args2) in combinations(group_atoms, 2):
                pattern = self._create_lifted_pattern(name1, args1, name2, args2)
                if pattern:
                    lifted_patterns.add(pattern)

            # Identify singleton predicates (appearing multiple times in same group)
            pred_counts = {}
            for name, _ in group_atoms:
                pred_counts[name] = pred_counts.get(name, 0) + 1

            for pred_name, count in pred_counts.items():
                if count > 1:
                    singleton_predicates.add(pred_name)

        print(f"[Mutex Analysis] Extracted {len(lifted_patterns)} unique lifted mutex patterns")

        return singleton_predicates, lifted_patterns

    def _create_lifted_pattern(self, name1: str, args1: Tuple[str, ...],
                              name2: str, args2: Tuple[str, ...]) -> Optional[LiftedMutexPattern]:
        """
        Create a lifted mutex pattern from two ground atoms.

        Identifies shared positions (same arg value) and different positions.

        Args:
            name1, args1: First atom
            name2, args2: Second atom

        Returns:
            LiftedMutexPattern or None if pattern would be trivial
        """
        shared_positions = []
        different_positions = []

        # For same-arity predicates: compare corresponding positions
        if len(args1) == len(args2):
            for pos in range(len(args1)):
                if args1[pos] == args2[pos]:
                    shared_positions.append((pos, pos))
                else:
                    different_positions.append((pos, pos))
        else:
            # Different arities: check cross-position matches
            # Find positions where args match across predicates
            for pos1, arg1 in enumerate(args1):
                for pos2, arg2 in enumerate(args2):
                    if arg1 == arg2:
                        shared_positions.append((pos1, pos2))
            # For different arities, we don't track "different" positions
            # since position correspondence is unclear

        if name1 == name2:
            # Same predicate mutex (e.g., holding(X) ⊕ holding(Y) or on(X,Y) ⊕ on(X,Z))
            #
            # Case 1: No shared positions, has different positions
            #   Example: holding(b1) vs holding(b2) → holding(X) ⊕ holding(Y) where X≠Y
            #   This is the SINGLETON pattern - at most one instance allowed
            #
            # Case 2: Has shared positions and different positions
            #   Example: on(b2,b1) vs on(b2,b3) → on(X,Y) ⊕ on(X,Z) where Y≠Z
            #   This means: same first arg, different second arg
            #
            # Case 3: Only shared positions, no different
            #   Example: on(b1,b2) vs on(b1,b2) → same atom, skip
            #
            if not different_positions:
                # No different positions = would only match identical atoms
                return None
            # Has different positions: valid pattern (whether or not shared exists)
            # - No shared: singleton constraint (holding(X) ⊕ holding(Y))
            # - Has shared: same-prefix constraint (on(X,Y) ⊕ on(X,Z))
        else:
            # Different predicates mutex (e.g., handempty ⊕ holding(X))
            # Always valid as long as we have two ground atoms that are mutex
            pass

        return LiftedMutexPattern(
            pred1_name=name1,
            pred1_arity=len(args1),
            pred2_name=name2,
            pred2_arity=len(args2),
            shared_positions=tuple(shared_positions),
            different_positions=tuple(different_positions)
        )

# Test function
def test_fd_extractor():
    """Test Fast Downward invariant extractor"""
    domain_path = 'src/domains/blocksworld/domain.pddl'
    objects = ['b1', 'b2', 'b3', 'b4', 'b5']

    extractor = FDInvariantExtractor(domain_path, objects)
    static_mutex, singletons, lifted_patterns = extractor.extract_invariants()

    print("Static Mutex Groups (from FD invariants):")
    for pred, mutex_preds in sorted(static_mutex.items()):
        print(f"  {pred} ↔ {mutex_preds}")

    print(f"\nSingleton Predicates:")
    for pred in sorted(singletons):
        print(f"  - {pred}")

    print(f"\nLifted Mutex Patterns:")
    for pattern in sorted(lifted_patterns, key=lambda p: (p.pred1_name, p.pred2_name)):
        print(f"  {pattern.pred1_name}(arity={pattern.pred1_arity}) ⊕ {pattern.pred2_name}(arity={pattern.pred2_arity})")
        print(f"    shared_positions={pattern.shared_positions}, different_positions={pattern.different_positions}")


if __name__ == "__main__":
    test_fd_extractor()
