"""
PR2 FOND Planner Wrapper

This module provides a Python interface to the PR2 Docker-based FOND planner.
PR2 uses PRP (Planner for Relevant Policies) to generate strong cyclic policies
for fully observable non-deterministic (FOND) planning problems.
"""

import subprocess
import re
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


class PR2Planner:
    """
    Wrapper for PR2 FOND planner using Docker.

    PR2 generates policies (not plans) that handle non-deterministic action outcomes.
    """

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize PR2 planner wrapper.

        Args:
            project_root: Path to project root (defaults to repository root)
        """
        if project_root is None:
            # Default to repository root
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.pr2_dir = self.project_root / "external" / "pr2"

        # Verify Docker image exists
        self._verify_docker_image()

    def _verify_docker_image(self) -> None:
        """Verify PR2 Docker image is available."""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", "pr2:latest"],
                capture_output=True,
                text=True,
                check=True
            )
            if not result.stdout.strip():
                raise RuntimeError(
                    "PR2 Docker image not found. Please build it first:\n"
                    "cd external/pr2 && docker build -t pr2 ."
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to check Docker images: {e}")

    def solve(self, domain_file: str, problem_file: str) -> Optional[List[Tuple[str, List[str]]]]:
        """
        Solve a FOND planning problem using PR2/PRP.

        Args:
            domain_file: Path to PDDL domain file (absolute or relative to project root)
            problem_file: Path to PDDL problem file (absolute or relative to problem root)

        Returns:
            List of (action_name, [parameters]) tuples, or None if no plan found
            Format matches pyperplan interface for drop-in replacement
        """
        # Convert to absolute paths
        domain_path = Path(domain_file)
        problem_path = Path(problem_file)

        if not domain_path.is_absolute():
            domain_path = self.project_root / domain_path
        if not problem_path.is_absolute():
            problem_path = self.project_root / problem_path

        # Verify files exist
        if not domain_path.exists():
            raise FileNotFoundError(f"Domain file not found: {domain_path}")
        if not problem_path.exists():
            raise FileNotFoundError(f"Problem file not found: {problem_path}")

        # Convert paths to be relative to PR2 directory for Docker mount
        # We mount the pr2 directory as /PROJECT and use relative paths from there
        try:
            domain_rel = f"/PROJECT/{domain_path.relative_to(self.pr2_dir)}"
            problem_rel = f"/PROJECT/{problem_path.relative_to(self.pr2_dir)}"
        except ValueError:
            # Files are outside pr2 directory - need to copy them
            raise NotImplementedError(
                "Currently only supports files within external/pr2/ directory. "
                f"Domain: {domain_path}, Problem: {problem_path}"
            )

        # Run PR2/PRP using Docker
        # Mount pr2 directory as /PROJECT (matching PR2 README recommendation)
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{str(self.pr2_dir)}:/PROJECT",
            "pr2",
            "/PLANNERS/bin/prp",
            domain_rel,
            problem_rel
        ]

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            output = result.stdout + result.stderr

            # Parse output for plan
            policy_actions = self._parse_policy(output)
            success = "Strong cyclic plan found" in output

            if success and policy_actions:
                return policy_actions
            else:
                return None

        except subprocess.TimeoutExpired:
            print("PR2 planner timed out after 5 minutes")
            return None
        except subprocess.CalledProcessError as e:
            print(f"PR2 planner failed: {e}")
            return None

    def _parse_policy(self, output: str) -> List[Tuple[str, List[str]]]:
        """
        Parse policy actions from PR2/PRP output.

        PR2 outputs a policy with state-action pairs. We extract the unique
        actions that appear in the policy.

        Args:
            output: Full output from PR2/PRP

        Returns:
            List of (action_name, [parameters]) tuples
            Example: [("pick-up", ["b5", "b4"]), ("put-on-block", ["b2", "b5"])]
        """
        actions = []

        # Look for lines like: "pick-up_DETDUP_1 b5 b4 (1)"
        # These appear in the initial plan section
        plan_section = False
        for line in output.split('\n'):
            # Start of plan output
            if "Plan length:" in line or "Generated" in line:
                plan_section = True
                continue

            # End of plan output
            if plan_section and ("Creating the simulator" in line or "Regressing" in line):
                break

            # Extract action lines
            if plan_section:
                # Match lines like: "pick-up_DETDUP_1 b5 b4 (1)"
                match = re.match(r'^([a-zA-Z0-9_-]+(?:_DETDUP_\d+)?)\s+(.+?)\s+\(\d+\)$', line.strip())
                if match:
                    action_name = match.group(1).replace('_DETDUP_0', '').replace('_DETDUP_1', '')
                    action_params = match.group(2).split()
                    # Remove underscores from action name for readability (convert to hyphens)
                    action_name = action_name.replace('_', '-')
                    actions.append((action_name, action_params))

        return actions

    def solve_from_strings(self, domain_str: str, problem_str: str, verbose: bool = False) -> Tuple[Optional[List[Tuple[str, List[str]]]], dict]:
        """
        Solve a FOND planning problem from string representations.

        This method creates temporary PDDL files in the PR2 directory to work around
        Docker volume mounting requirements.

        Args:
            domain_str: PDDL domain as string
            problem_str: PDDL problem as string
            verbose: If True, print detailed PR2 output

        Returns:
            Tuple of (plan, info_dict) where:
            - plan: List of (action_name, [parameters]) tuples, or None if no plan found
            - info_dict: Dictionary containing PR2 execution info:
                - success: bool - Whether planning succeeded
                - plan_length: int - Number of actions in plan
                - pr2_output: str - Full PR2/PRP output
                - error: str - Error message if planning failed
        """
        # Create temporary files in PR2 directory (so they can be mounted)
        temp_dir = self.pr2_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        domain_file = temp_dir / "temp_domain.pddl"
        problem_file = temp_dir / "temp_problem.pddl"

        info_dict = {
            "success": False,
            "plan_length": 0,
            "pr2_output": "",
            "error": None
        }

        try:
            domain_file.write_text(domain_str)
            problem_file.write_text(problem_str)

            # Run PR2 planner
            plan, pr2_output = self._solve_with_output(str(domain_file), str(problem_file))

            # Update info dict
            info_dict["pr2_output"] = pr2_output

            if plan:
                info_dict["success"] = True
                info_dict["plan_length"] = len(plan)
            else:
                info_dict["error"] = "No plan found by PR2 planner"

            # Print detailed output if requested
            if verbose:
                print("\n" + "="*80)
                print("PR2 PLANNER OUTPUT:")
                print("="*80)
                print(pr2_output)
                print("="*80 + "\n")

        except Exception as e:
            info_dict["error"] = str(e)
            plan = None
        finally:
            # Clean up temporary files
            if domain_file.exists():
                domain_file.unlink()
            if problem_file.exists():
                problem_file.unlink()
            # Remove temp directory if empty
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()

        return plan, info_dict

    def _solve_with_output(self, domain_file: str, problem_file: str) -> Tuple[Optional[List[Tuple[str, List[str]]]], str]:
        """
        Internal method to solve planning problem and return both plan and full output.

        Returns:
            Tuple of (plan, output_string)
        """
        # Convert to absolute paths
        domain_path = Path(domain_file)
        problem_path = Path(problem_file)

        if not domain_path.is_absolute():
            domain_path = self.project_root / domain_path
        if not problem_path.is_absolute():
            problem_path = self.project_root / problem_path

        # Verify files exist
        if not domain_path.exists():
            raise FileNotFoundError(f"Domain file not found: {domain_path}")
        if not problem_path.exists():
            raise FileNotFoundError(f"Problem file not found: {problem_path}")

        # Convert paths to be relative to PR2 directory for Docker mount
        try:
            domain_rel = f"/PROJECT/{domain_path.relative_to(self.pr2_dir)}"
            problem_rel = f"/PROJECT/{problem_path.relative_to(self.pr2_dir)}"
        except ValueError:
            raise NotImplementedError(
                "Currently only supports files within external/pr2/ directory. "
                f"Domain: {domain_path}, Problem: {problem_path}"
            )

        # Run PR2/PRP using Docker
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{str(self.pr2_dir)}:/PROJECT",
            "pr2",
            "/PLANNERS/bin/prp",
            domain_rel,
            problem_rel
        ]

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            output = result.stdout + result.stderr

            # Parse output for plan
            policy_actions = self._parse_policy(output)
            success = "Strong cyclic plan found" in output

            if success and policy_actions:
                return policy_actions, output
            else:
                return None, output

        except subprocess.TimeoutExpired:
            return None, "PR2 planner timed out after 5 minutes"
        except subprocess.CalledProcessError as e:
            return None, f"PR2 planner failed: {e}"


def main():
    """Test PR2 planner with blocksworld example."""
    planner = PR2Planner()

    # Test with FOND blocksworld problem
    domain = "external/pr2/fond-benchmarks/blocksworld/domain.pddl"
    problem = "external/pr2/fond-benchmarks/blocksworld/p1.pddl"

    print(f"Testing PR2 planner with:")
    print(f"  Domain: {domain}")
    print(f"  Problem: {problem}")
    print()

    policy = planner.solve(domain, problem)

    print("=" * 80)
    print("PARSED POLICY:")
    print("=" * 80)
    if policy:
        print(f"✓ Strong cyclic plan found with {len(policy)} actions:")
        for i, (action, params) in enumerate(policy, 1):
            print(f"  {i}. {action}({', '.join(params)})")
    else:
        print("✗ No plan found")
    print()


if __name__ == "__main__":
    main()
