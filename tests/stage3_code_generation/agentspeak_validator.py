"""
AgentSpeak Code Validator

Validates generated AgentSpeak code for:
1. Action Plans (PDDL action → AgentSpeak)
2. Goal Achievement Plans (state graph → plans)
3. Basic structure (beliefs, syntax)
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class ValidationError:
    """Represents a validation error"""
    category: str  # "action_plan", "goal_plan", "structure"
    severity: str  # "error", "warning"
    message: str
    context: Optional[str] = None  # Code snippet showing the issue


class AgentSpeakValidator:
    """
    Validator for generated AgentSpeak code

    Validates two types of plans:
    1. Action Plans: PDDL action → AgentSpeak goal plans
       - First line MUST be physical action call
       - Must have belief updates matching PDDL effects

    2. Goal Achievement Plans: state → goal progression
       - First line MAY be subgoal (not necessarily physical action)
       - Must have action goal invocations
       - May have recursive goal checks
    """

    def __init__(self, asl_code: str, domain=None, goals: List[str] = None):
        """
        Initialize validator

        Args:
            asl_code: Generated AgentSpeak code
            domain: PDDL domain (optional, for detailed validation)
            goals: List of goal names to check for
        """
        self.code = asl_code
        self.domain = domain
        self.goals = goals or []
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []

    def validate(self) -> Tuple[bool, List[ValidationError]]:
        """
        Run all validations

        Returns:
            (passed, errors) where passed is True if no errors
        """
        self.errors = []
        self.warnings = []

        # 1. Basic structure validation
        self._validate_structure()

        # 2. Action plans validation
        self._validate_action_plans()

        # 3. Goal plans validation
        self._validate_goal_plans()

        # 4. Check statistics match actual counts
        self._validate_statistics()

        all_issues = self.errors + self.warnings
        return (len(self.errors) == 0, all_issues)

    # ========================================================================
    # Structure Validation
    # ========================================================================

    def _validate_structure(self):
        """Validate basic AgentSpeak structure"""

        # Check initial beliefs section exists
        if "/* Initial Beliefs */" not in self.code:
            self.errors.append(ValidationError(
                category="structure",
                severity="error",
                message="Missing initial beliefs section"
            ))

        # Check has some initial beliefs
        beliefs = re.findall(r'^(\w+)\([^)]*\)\.$', self.code, re.MULTILINE)
        if len(beliefs) == 0:
            self.warnings.append(ValidationError(
                category="structure",
                severity="warning",
                message="No initial beliefs found"
            ))

        # Check code is not empty
        if len(self.code.strip()) < 50:
            self.errors.append(ValidationError(
                category="structure",
                severity="error",
                message="Generated code is too short (< 50 chars)"
            ))

    # ========================================================================
    # Action Plans Validation
    # ========================================================================

    def _validate_action_plans(self):
        """Validate Action Plans (PDDL action → AgentSpeak)"""

        # Find action plans section
        action_section = self._extract_section("PDDL Action Plans")
        if not action_section:
            self.errors.append(ValidationError(
                category="action_plan",
                severity="error",
                message="No PDDL Action Plans section found"
            ))
            return

        # Extract all action plans
        action_plans = self._extract_plans(action_section)

        if len(action_plans) == 0:
            self.errors.append(ValidationError(
                category="action_plan",
                severity="error",
                message="No action plans found"
            ))
            return

        # Validate each action plan
        for plan in action_plans:
            self._validate_single_action_plan(plan)

    def _validate_single_action_plan(self, plan: str):
        """
        Validate a single action plan

        Expected format:
        +!action_name(Args) : context <-
            action_name_physical(Args);
            belief_updates.
        """
        # Extract plan components
        match = re.match(
            r'\+!(\w+)(\([^)]*\))?\s*:\s*(.+?)\s*<-\s*(.+)',
            plan,
            re.DOTALL
        )

        if not match:
            self.errors.append(ValidationError(
                category="action_plan",
                severity="error",
                message="Action plan format invalid",
                context=plan[:100]
            ))
            return

        action_name = match.group(1)
        params = match.group(2) or ""
        context = match.group(3)
        body = match.group(4)

        # Check context is not empty
        if not context.strip() or context.strip() == "true":
            self.warnings.append(ValidationError(
                category="action_plan",
                severity="warning",
                message=f"Action plan {action_name} has empty/trivial context",
                context=f"Context: {context}"
            ))

        # Extract body lines
        body_lines = [line.strip() for line in re.split(r'[;\n]', body) if line.strip()]

        if len(body_lines) == 0:
            self.errors.append(ValidationError(
                category="action_plan",
                severity="error",
                message=f"Action plan {action_name} has empty body"
            ))
            return

        # CRITICAL: First line must be physical action call
        first_line = body_lines[0].rstrip('.')
        expected_physical = f"{action_name}_physical"

        if not first_line.startswith(expected_physical):
            self.errors.append(ValidationError(
                category="action_plan",
                severity="error",
                message=f"Action plan {action_name}: first line must be physical action call",
                context=f"Expected: {expected_physical}(...)\nGot: {first_line}"
            ))

        # Check has belief updates (lines starting with + or -)
        has_updates = any(
            line.startswith('+') or line.startswith('-')
            for line in body_lines[1:]
        )

        if not has_updates:
            self.warnings.append(ValidationError(
                category="action_plan",
                severity="warning",
                message=f"Action plan {action_name} has no belief updates",
                context=f"Body: {body[:100]}"
            ))

    # ========================================================================
    # Goal Plans Validation
    # ========================================================================

    def _validate_goal_plans(self):
        """Validate Goal Achievement Plans"""

        # Find goal plans section
        goal_section = self._extract_section("Goal Achievement Plans")
        if not goal_section:
            # May not have goal plans section if goal is directly achievable
            return

        # Extract all goal plans
        goal_plans = self._extract_plans(goal_section)

        if len(goal_plans) == 0:
            self.warnings.append(ValidationError(
                category="goal_plan",
                severity="warning",
                message="No goal achievement plans found"
            ))
            return

        # Validate each goal plan
        for plan in goal_plans:
            self._validate_single_goal_plan(plan)

    def _validate_single_goal_plan(self, plan: str):
        """
        Validate a single goal achievement plan

        Expected format:
        +!goal_name : context <-
            !subgoal1;        // optional precondition subgoals
            !action_goal(...);  // action invocation
            !goal_name.       // optional recursive check
        """
        # Extract plan components
        match = re.match(
            r'\+!(\w+)(\([^)]*\))?\s*:\s*(.+?)\s*<-\s*(.+)',
            plan,
            re.DOTALL
        )

        if not match:
            self.errors.append(ValidationError(
                category="goal_plan",
                severity="error",
                message="Goal plan format invalid",
                context=plan[:100]
            ))
            return

        goal_name = match.group(1)
        params = match.group(2) or ""
        context = match.group(3)
        body = match.group(4)

        # Extract body lines
        body_lines = [line.strip() for line in re.split(r'[;\n]', body) if line.strip()]

        if len(body_lines) == 0:
            self.errors.append(ValidationError(
                category="goal_plan",
                severity="error",
                message=f"Goal plan {goal_name} has empty body"
            ))
            return

        # Check: body should contain at least one goal invocation (!)
        has_goal_invocation = any('!' in line for line in body_lines)

        if not has_goal_invocation:
            self.errors.append(ValidationError(
                category="goal_plan",
                severity="error",
                message=f"Goal plan {goal_name} has no goal invocations",
                context=f"Body: {body[:100]}"
            ))

        # Check: should have at least one action goal invocation
        # Action goals typically start with !pick_, !put_, !move_, etc.
        action_goal_pattern = r'!(pick|put|move|stack|unstack)'
        has_action_goal = any(re.search(action_goal_pattern, line) for line in body_lines)

        # Check if this is a success plan (goal already achieved)
        is_success_plan = any('already achieved' in line.lower() for line in body_lines)

        if not has_action_goal and not is_success_plan:
            # This might be okay if it's a subgoal plan, so just warn
            self.warnings.append(ValidationError(
                category="goal_plan",
                severity="warning",
                message=f"Goal plan {goal_name} has no obvious action goal invocation",
                context=f"Body lines: {body_lines[:3]}"
            ))

    # ========================================================================
    # Statistics Validation
    # ========================================================================

    def _validate_statistics(self):
        """
        Validate that statistics in header match actual counts

        Note: Statistics count only "goal achievement plans" (from non-goal states),
        not success/failure plans
        """

        # Extract statistics from header
        stats_match = re.search(r'Goal Plans:\s*(\d+)', self.code)
        if not stats_match:
            return  # No statistics to validate

        claimed_count = int(stats_match.group(1))

        # Count actual goal plans (excluding success plan)
        goal_section = self._extract_section("Goal Achievement Plans")
        if goal_section:
            all_plans = self._extract_plans(goal_section)

            # Filter out success plan (contains "already achieved")
            non_success_plans = [
                plan for plan in all_plans
                if 'already achieved' not in plan.lower()
            ]

            actual_count = len(non_success_plans)

            if actual_count != claimed_count:
                self.errors.append(ValidationError(
                    category="structure",
                    severity="error",
                    message=f"Goal plan count mismatch: header says {claimed_count}, found {actual_count} (excluding success plan)"
                ))

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _extract_section(self, section_name: str) -> Optional[str]:
        """
        Extract a section from the code

        Sections are marked by comments like /* PDDL Action Plans ... */
        Section continues until next major section or end of file
        """
        # Find the section marker
        marker_pattern = rf'/\*\s*{re.escape(section_name)}[^*]*\*/'
        marker_match = re.search(marker_pattern, self.code, re.DOTALL)

        if not marker_match:
            return None

        start_pos = marker_match.end()

        # Find the next major section marker (/* WORD WORD ... */)
        # Major sections have capitalized words or specific keywords
        next_section_pattern = r'/\*\s*(?:Initial Beliefs|PDDL Action|Goal Achievement|Success|Failure)'
        next_match = re.search(next_section_pattern, self.code[start_pos:])

        if next_match:
            end_pos = start_pos + next_match.start()
            return self.code[start_pos:end_pos]
        else:
            # No next section, take rest of code
            return self.code[start_pos:]

    def _extract_plans(self, section: str) -> List[str]:
        """
        Extract individual plans from a section

        Plans start with +! (achievement goal) or -! (failure handler)
        Only return achievement goals (+!), not failure handlers
        """
        plans = []

        # Split by +! to get achievement goal plans
        # (Failure plans with -! are handled separately)
        parts = section.split('+!')

        for part in parts[1:]:  # Skip first empty part
            # Find the end of this plan (period followed by newline or end)
            end_match = re.search(r'\.\s*(?=\n|$)', part)
            if end_match:
                plan_text = '+!' + part[:end_match.end()].strip()
                plans.append(plan_text)

        return plans

    def format_report(self) -> str:
        """Generate a formatted validation report"""
        lines = []
        lines.append("=" * 80)
        lines.append("AGENTSPEAK VALIDATION REPORT")
        lines.append("=" * 80)

        if len(self.errors) == 0 and len(self.warnings) == 0:
            lines.append("\n✅ ALL VALIDATIONS PASSED\n")
            return "\n".join(lines)

        if self.errors:
            lines.append(f"\n❌ ERRORS: {len(self.errors)}")
            for i, err in enumerate(self.errors, 1):
                lines.append(f"\n{i}. [{err.category}] {err.message}")
                if err.context:
                    lines.append(f"   Context: {err.context}")

        if self.warnings:
            lines.append(f"\n⚠️  WARNINGS: {len(self.warnings)}")
            for i, warn in enumerate(self.warnings, 1):
                lines.append(f"\n{i}. [{warn.category}] {warn.message}")
                if warn.context:
                    lines.append(f"   Context: {warn.context}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


# Quick test function
def validate_agentspeak_code(asl_code: str, domain=None, goals: List[str] = None) -> Tuple[bool, str]:
    """
    Convenience function to validate AgentSpeak code

    Returns:
        (passed, report) tuple
    """
    validator = AgentSpeakValidator(asl_code, domain, goals)
    passed, issues = validator.validate()
    report = validator.format_report()
    return passed, report
