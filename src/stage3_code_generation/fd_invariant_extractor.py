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
from typing import Dict, Set, List, Tuple


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

    def extract_invariants(self) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        Extract static mutex groups using Fast Downward

        Returns:
            (static_mutex_map, singleton_predicates)
            - static_mutex_map: {pred_name: {mutex_pred1, mutex_pred2, ...}}
            - singleton_predicates: {pred_name1, pred_name2, ...}
        """
        # Check if Fast Downward is available
        if not self._check_fd_available():
            print("[WARNING] Fast Downward not available - using fallback static mutex")
            return self._fallback_static_mutex()

        # Create mock problem file
        problem_path = self._create_mock_problem()

        try:
            # Run Fast Downward translator
            sas_output = self._run_fd_translator(problem_path)

            # Parse SAS output to extract mutex groups
            static_mutex_map, singleton_predicates = self._parse_sas_mutex_groups(sas_output)

            return static_mutex_map, singleton_predicates

        except Exception as e:
            print(f"[WARNING] Fast Downward invariant extraction failed: {e}")
            print("[WARNING] Falling back to hardcoded static mutex")
            return self._fallback_static_mutex()

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

        Returns:
            Path to created problem file
        """
        # Generate problem with all objects
        objects_str = ' '.join(f"{obj} - block" for obj in self.objects)

        # Create simple initial state (all blocks on table, all clear, hand empty)
        init_facts = []
        for obj in self.objects:
            init_facts.append(f"    (ontable {obj})")
            init_facts.append(f"    (clear {obj})")
        init_facts.append("    (handempty)")

        # Use a simple goal to force FD to generate invariants
        # Goal: stack first block on second block
        goal = f"(on {self.objects[0]} {self.objects[1]})" if len(self.objects) >= 2 else "(handempty)"

        problem_content = f"""(define (problem mock-invariant-extraction)
  (:domain blocksworld)
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

    def _parse_sas_mutex_groups(self, sas_content: str) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        Parse SAS output to extract mutex groups

        SAS format:
        1. Variables define possible values for state variables
        2. Mutex groups define invariants across different variables

        Args:
            sas_content: Content of output.sas file

        Returns:
            (static_mutex_map, singleton_predicates)
        """
        static_mutex_map = {}
        singleton_predicates = set()

        # Step 1: Parse variable definitions to build var_id -> {value_id -> predicate_name}
        var_map = {}  # var_id -> {value_id -> pred_name}

        var_pattern = re.compile(
            r'begin_variable\s+var(\d+)\s+(-?\d+)\s+(\d+)\s+(.*?)\s+end_variable',
            re.DOTALL
        )

        for match in var_pattern.finditer(sas_content):
            var_id = int(match.group(1))
            # group(2) is the axiom layer, group(3) is range
            atoms_str = match.group(4)

            var_map[var_id] = {}

            # Parse each line as "Atom predicate(args)" or "NegatedAtom ..."
            for value_id, line in enumerate(atoms_str.strip().split('\n')):
                # Only consider positive atoms (skip NegatedAtom)
                if line.strip().startswith('Atom '):
                    # Extract predicate name (without arguments)
                    atom_match = re.search(r'Atom\s+(\w+)\(', line)
                    if atom_match:
                        pred_name = atom_match.group(1)
                        var_map[var_id][value_id] = pred_name

        # Step 2: Parse mutex groups
        # Format: begin_mutex_group \n count \n var_id value_id \n ... \n end_mutex_group
        mutex_pattern = re.compile(
            r'begin_mutex_group\s+(\d+)\s+((?:\d+\s+\d+\s*\n?)+)end_mutex_group',
            re.DOTALL
        )

        for match in mutex_pattern.finditer(sas_content):
            count = int(match.group(1))
            entries_str = match.group(2)

            # Parse var_id value_id pairs
            predicates_in_group = []
            for line in entries_str.strip().split('\n'):
                parts = line.strip().split()
                if len(parts) == 2:
                    var_id = int(parts[0])
                    value_id = int(parts[1])

                    # Look up the predicate name
                    if var_id in var_map and value_id in var_map[var_id]:
                        pred_name = var_map[var_id][value_id]
                        predicates_in_group.append(pred_name)

            # Analyze the mutex group
            # Example: ['handempty', 'holding', 'holding', 'holding']
            # Unique predicates: {'handempty', 'holding'}

            unique_preds = set(predicates_in_group)

            # Count occurrences of each predicate
            from collections import Counter
            pred_counts = Counter(predicates_in_group)

            # Identify singletons: predicates that appear multiple times in the group
            # Example: 'holding' appears 3 times → only one instance allowed
            for pred_name, count in pred_counts.items():
                if count > 1:
                    singleton_predicates.add(pred_name)

            # Add pairwise mutex for DIFFERENT predicates
            # Example: 'handempty' and 'holding' are mutex
            # Do NOT add 'holding' ↔ 'holding' (that's handled by singleton)
            unique_preds_list = list(unique_preds)
            for i in range(len(unique_preds_list)):
                for j in range(i + 1, len(unique_preds_list)):
                    pred1 = unique_preds_list[i]
                    pred2 = unique_preds_list[j]

                    if pred1 not in static_mutex_map:
                        static_mutex_map[pred1] = set()
                    static_mutex_map[pred1].add(pred2)

                    if pred2 not in static_mutex_map:
                        static_mutex_map[pred2] = set()
                    static_mutex_map[pred2].add(pred1)

        return static_mutex_map, singleton_predicates

    def _fallback_static_mutex(self) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        Fallback static mutex when Fast Downward is not available

        Returns:
            (static_mutex_map, singleton_predicates)
        """
        # For blocksworld domain:
        # - holding ↔ handempty (mutex: can't hold a block and have empty hand)
        # - holding is singleton (only one block can be held at a time)
        # - handempty is NOT singleton (it's a boolean predicate, not multi-instance)
        static_mutex_map = {
            'holding': {'handempty'},
            'handempty': {'holding'}
        }

        singleton_predicates = {'holding'}

        return static_mutex_map, singleton_predicates


# Test function
def test_fd_extractor():
    """Test Fast Downward invariant extractor"""
    domain_path = 'src/domains/blocksworld/domain.pddl'
    objects = ['b1', 'b2', 'b3', 'b4', 'b5']

    extractor = FDInvariantExtractor(domain_path, objects)
    static_mutex, singletons = extractor.extract_invariants()

    print("Static Mutex Groups (from FD invariants):")
    for pred, mutex_preds in sorted(static_mutex.items()):
        print(f"  {pred} ↔ {mutex_preds}")

    print(f"\nSingleton Predicates:")
    for pred in sorted(singletons):
        print(f"  - {pred}")


if __name__ == "__main__":
    test_fd_extractor()
