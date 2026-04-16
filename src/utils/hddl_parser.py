"""
HDDL domain parser.

Extracts domain information from HDDL domain files:
- domain name
- requirements
- types
- predicates
- tasks
- methods
- actions
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class HDDLPredicate:
    """Represents an HDDL predicate."""

    name: str
    parameters: List[str]

    def to_signature(self) -> str:
        """Convert to signature format: on(X, Y)."""
        if not self.parameters:
            return self.name

        var_names: List[str] = []
        for param in self.parameters:
            parts = param.split("-")
            vars_part = parts[0].strip()
            vars_in_this_part = [
                token.strip().lstrip("?").upper()
                for token in vars_part.split()
                if token.strip()
            ]
            var_names.extend(vars_in_this_part)

        return f"{self.name}({', '.join(var_names)})"


@dataclass
class HDDLAction:
    """Represents an HDDL primitive action schema."""

    name: str
    parameters: List[str]
    preconditions: str
    effects: str

    def to_description(self) -> str:
        """Convert to a human-readable action description."""
        params_str = ", ".join(self.parameters) if self.parameters else "none"
        return (
            f"{self.name}({params_str})\n"
            f"    Pre: {self.preconditions}\n"
            f"    Eff: {self.effects}"
        )


@dataclass
class HDDLTask:
    """Represents an HDDL compound task schema."""

    name: str
    parameters: List[str]

    def to_signature(self) -> str:
        """Convert to signature format: place_on(X, Y)."""
        if not self.parameters:
            return self.name

        var_names: List[str] = []
        for param in self.parameters:
            parts = param.split("-")
            vars_part = parts[0].strip()
            vars_in_this_part = [
                token.strip().lstrip("?").upper()
                for token in vars_part.split()
                if token.strip()
            ]
            var_names.extend(vars_in_this_part)

        return f"{self.name}({', '.join(var_names)})"


@dataclass
class HDDLSubtask:
    """Represents one labelled subtask inside an HDDL method body."""

    label: str
    task_name: str
    args: List[str]


@dataclass
class HDDLFact:
    """Represents one grounded fact in a problem file."""

    predicate: str
    args: List[str]
    is_positive: bool = True

    def to_signature(self) -> str:
        if not self.args:
            atom = self.predicate
        else:
            atom = f"{self.predicate}({', '.join(self.args)})"
        return atom if self.is_positive else f"not {atom}"


@dataclass
class HDDLTaskInvocation:
    """Represents one task call in a problem HTN network."""

    task_name: str
    args: List[str]
    label: Optional[str] = None

    def to_signature(self) -> str:
        if not self.args:
            return self.task_name
        return f"{self.task_name}({', '.join(self.args)})"


@dataclass
class HDDLMethod:
    """Represents an HDDL method."""

    name: str
    task_name: str
    task_args: List[str]
    parameters: List[str]
    precondition: str
    subtasks: List[HDDLSubtask]
    ordering: List[Tuple[str, str]]


@dataclass
class HDDLDomain:
    """Parsed HDDL domain information."""

    name: str
    requirements: List[str]
    types: List[str]
    predicates: List[HDDLPredicate]
    tasks: List[HDDLTask]
    methods: List[HDDLMethod]
    actions: List[HDDLAction]

    def get_predicate_signatures(self) -> List[str]:
        """Get all predicate signatures in format: on(X, Y)."""
        return [predicate.to_signature() for predicate in self.predicates]

    def get_action_names(self) -> List[str]:
        """Get all primitive action names."""
        return [action.name for action in self.actions]

    def get_task_signatures(self) -> List[str]:
        """Get all task signatures in format: place_on(X, Y)."""
        return [task.to_signature() for task in self.tasks]


@dataclass
class HDDLProblem:
    """Parsed HDDL problem information."""

    name: str
    domain_name: str
    objects: List[str]
    object_types: Dict[str, str]
    htn_parameter_types: Dict[str, str]
    init_facts: List[HDDLFact]
    htn_tasks: List[HDDLTaskInvocation]
    htn_ordered: bool
    htn_ordering: List[Tuple[str, str]]
    goal_facts: List[HDDLFact]


class HDDLParser:
    """Parser for the subset of HDDL used by this project."""

    @staticmethod
    def parse_domain(file_path: str) -> HDDLDomain:
        """
        Parse an HDDL domain file.

        Args:
            file_path: Path to `domain.hddl`.

        Returns:
            HDDLDomain with parsed information.
        """
        content = Path(file_path).read_text()
        content = re.sub(r";.*$", "", content, flags=re.MULTILINE)

        domain_name_match = re.search(r"\(define\s+\(domain\s+([^\s)]+)\)", content)
        domain_name = domain_name_match.group(1) if domain_name_match else "unknown"

        requirements = HDDLParser._extract_simple_list_block(content, "requirements")
        types = HDDLParser._extract_simple_list_block(content, "types")
        predicates = HDDLParser._extract_predicates(content)
        tasks = HDDLParser._extract_tasks(content)
        methods = HDDLParser._extract_methods(content)
        actions = HDDLParser._extract_actions(content)

        return HDDLDomain(
            name=domain_name,
            requirements=requirements,
            types=types,
            predicates=predicates,
            tasks=tasks,
            methods=methods,
            actions=actions,
        )

    @staticmethod
    def parse_problem(file_path: str) -> HDDLProblem:
        """
        Parse an HDDL problem file.

        Args:
            file_path: Path to `problem.hddl`.

        Returns:
            HDDLProblem with parsed objects, init, HTN tasks, and optional goal facts.
        """
        content = Path(file_path).read_text()
        content = re.sub(r";.*$", "", content, flags=re.MULTILINE)

        problem_name_match = re.search(r"\(define\s+\(problem\s+([^\s)]+)\)", content)
        problem_name = problem_name_match.group(1) if problem_name_match else "unknown_problem"
        domain_name_match = re.search(r"\(:domain\s+([^\s)]+)\)", content)
        domain_name = domain_name_match.group(1) if domain_name_match else "unknown_domain"

        objects, object_types = HDDLParser._extract_problem_objects(content)
        htn_parameter_types = HDDLParser._extract_problem_htn_parameters(content)
        init_facts = HDDLParser._extract_problem_init_facts(content)
        htn_tasks = HDDLParser._extract_problem_htn_tasks(content)
        htn_ordered = HDDLParser._problem_htn_tasks_are_ordered(content)
        htn_ordering = HDDLParser._extract_problem_htn_ordering(content)
        goal_facts = HDDLParser._extract_problem_goal_facts(content)

        return HDDLProblem(
            name=problem_name,
            domain_name=domain_name,
            objects=objects,
            object_types=object_types,
            htn_parameter_types=htn_parameter_types,
            init_facts=init_facts,
            htn_tasks=htn_tasks,
            htn_ordered=htn_ordered,
            htn_ordering=htn_ordering,
            goal_facts=goal_facts,
        )

    @staticmethod
    def _extract_simple_list_block(content: str, keyword: str) -> List[str]:
        block = HDDLParser._extract_single_block(content, keyword)
        if block is None:
            return []

        match = re.search(rf"\(:{keyword}\s+(.*?)\)$", block, flags=re.DOTALL)
        if not match:
            return []

        return [token.strip() for token in match.group(1).split() if token.strip()]

    @staticmethod
    def _extract_problem_objects(content: str) -> Tuple[List[str], Dict[str, str]]:
        block = HDDLParser._extract_single_block(content, "objects")
        if block is None:
            return [], {}

        inner = block[len("(:objects"): -1].strip()
        raw_params = HDDLParser._parse_parameters(inner)
        objects: List[str] = []
        object_types: Dict[str, str] = {}
        for item in raw_params:
            if " - " in item:
                object_name, type_name = item.split(" - ", 1)
                object_name = object_name.strip()
                type_name = type_name.strip() or "object"
            else:
                object_name = item.strip()
                type_name = "object"
            if not object_name:
                continue
            objects.append(object_name)
            object_types[object_name] = type_name
        return objects, object_types

    @staticmethod
    def _extract_problem_htn_parameters(content: str) -> Dict[str, str]:
        block = HDDLParser._extract_single_block(content, "htn")
        if block is None:
            return {}
        params_expr = HDDLParser._extract_expression_after_keyword(block, ":parameters")
        if params_expr is None:
            return {}
        params_str = params_expr[1:-1].strip() if params_expr.startswith("(") and params_expr.endswith(")") else params_expr
        raw_params = HDDLParser._parse_parameters(params_str)
        htn_parameter_types: Dict[str, str] = {}
        for item in raw_params:
            if " - " in item:
                parameter_name, type_name = item.split(" - ", 1)
                parameter_name = parameter_name.strip()
                type_name = type_name.strip() or "object"
            else:
                parameter_name = item.strip()
                type_name = "object"
            if not parameter_name:
                continue
            htn_parameter_types[parameter_name] = type_name
        return htn_parameter_types

    @staticmethod
    def _extract_problem_init_facts(content: str) -> List[HDDLFact]:
        block = HDDLParser._extract_single_block(content, "init")
        if block is None:
            return []
        tree = _SimpleSExpressionParser.parse_expression(block)
        if not isinstance(tree, list) or not tree or tree[0] != ":init":
            return []
        facts: List[HDDLFact] = []
        for item in tree[1:]:
            fact = HDDLParser._sexpr_to_fact(item)
            if fact is not None:
                facts.append(fact)
        return facts

    @staticmethod
    def _extract_problem_goal_facts(content: str) -> List[HDDLFact]:
        block = HDDLParser._extract_single_block(content, "goal")
        if block is None:
            return []
        goal_expr = HDDLParser._extract_expression_after_keyword(block, ":goal")
        if goal_expr is None:
            return []
        tree = _SimpleSExpressionParser.parse_expression(goal_expr)
        if isinstance(tree, list) and tree and tree[0] == "and":
            items = tree[1:]
        else:
            items = [tree]
        facts: List[HDDLFact] = []
        for item in items:
            fact = HDDLParser._sexpr_to_fact(item)
            if fact is not None:
                facts.append(fact)
        return facts

    @staticmethod
    def _extract_problem_htn_tasks(content: str) -> List[HDDLTaskInvocation]:
        block = HDDLParser._extract_single_block(content, "htn")
        if block is None:
            return []
        tasks_expr = HDDLParser._extract_expression_after_keyword(block, ":tasks")
        if tasks_expr is None:
            tasks_expr = HDDLParser._extract_expression_after_keyword(block, ":ordered-subtasks")
        if tasks_expr is None:
            tasks_expr = HDDLParser._extract_expression_after_keyword(block, ":subtasks")
        if tasks_expr is None:
            return []

        tree = _SimpleSExpressionParser.parse_expression(tasks_expr)
        items = tree[1:] if isinstance(tree, list) and tree and tree[0] == "and" else [tree]
        task_invocations: List[HDDLTaskInvocation] = []
        for item in items:
            if not isinstance(item, list) or not item:
                continue
            task_label: Optional[str] = None
            if (
                len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], list)
                and item[1]
            ):
                task_label = str(item[0])
                task_name = str(item[1][0])
                task_args = [str(value) for value in item[1][1:]]
            else:
                task_name = str(item[0])
                task_args = [str(value) for value in item[1:]]
            task_invocations.append(
                HDDLTaskInvocation(
                    task_name=task_name,
                    args=task_args,
                    label=task_label,
                )
            )
        return task_invocations

    @staticmethod
    def _problem_htn_tasks_are_ordered(content: str) -> bool:
        block = HDDLParser._extract_single_block(content, "htn")
        if block is None:
            return False
        return ":ordered-subtasks" in block

    @staticmethod
    def _extract_problem_htn_ordering(content: str) -> List[Tuple[str, str]]:
        block = HDDLParser._extract_single_block(content, "htn")
        if block is None:
            return []
        ordering_expr = HDDLParser._extract_expression_after_keyword(block, ":ordering")
        return HDDLParser._parse_ordering(ordering_expr)

    @staticmethod
    def _sexpr_to_fact(item: object) -> Optional[HDDLFact]:
        if not isinstance(item, list) or not item:
            return None
        if item[0] == "not" and len(item) == 2 and isinstance(item[1], list):
            atom = item[1]
            if not atom:
                return None
            return HDDLFact(
                predicate=str(atom[0]),
                args=[str(value) for value in atom[1:]],
                is_positive=False,
            )
        return HDDLFact(
            predicate=str(item[0]),
            args=[str(value) for value in item[1:]],
            is_positive=True,
        )

    @staticmethod
    def _extract_predicates(content: str) -> List[HDDLPredicate]:
        block = HDDLParser._extract_single_block(content, "predicates")
        if block is None:
            return []

        inner = block[len("(:predicates"): -1].strip()
        pattern = r"\(([a-zA-Z][a-zA-Z0-9_-]*)\s*([^)]*)\)"
        matches = re.findall(pattern, inner)

        predicates: List[HDDLPredicate] = []
        for name, params_str in matches:
            predicates.append(
                HDDLPredicate(
                    name=name,
                    parameters=HDDLParser._parse_parameters(params_str),
                )
            )
        return predicates

    @staticmethod
    def _extract_tasks(content: str) -> List[HDDLTask]:
        tasks: List[HDDLTask] = []
        for block in HDDLParser._extract_blocks(content, "task"):
            name_match = re.match(r"\(:task\s+([^\s()]+)", block)
            if not name_match:
                continue

            params_expr = HDDLParser._extract_expression_after_keyword(block, ":parameters")
            params_str = params_expr[1:-1].strip() if params_expr else ""
            tasks.append(
                HDDLTask(
                    name=name_match.group(1),
                    parameters=HDDLParser._parse_parameters(params_str),
                )
            )
        return tasks

    @staticmethod
    def _extract_methods(content: str) -> List[HDDLMethod]:
        methods: List[HDDLMethod] = []
        for block in HDDLParser._extract_blocks(content, "method"):
            name_match = re.match(r"\(:method\s+([^\s()]+)", block)
            if not name_match:
                continue

            params_expr = HDDLParser._extract_expression_after_keyword(block, ":parameters")
            params_str = params_expr[1:-1].strip() if params_expr else ""

            task_expr = HDDLParser._extract_expression_after_keyword(block, ":task")
            task_name, task_args = HDDLParser._parse_task_invocation(task_expr)

            precondition_expr = HDDLParser._extract_expression_after_keyword(
                block,
                ":precondition",
            )
            subtasks_expr = HDDLParser._extract_expression_after_keyword(
                block,
                ":ordered-subtasks",
            )
            ordering: List[Tuple[str, str]] = []
            if subtasks_expr is None:
                subtasks_expr = HDDLParser._extract_expression_after_keyword(block, ":subtasks")
                ordering_expr = HDDLParser._extract_expression_after_keyword(block, ":ordering")
                ordering = HDDLParser._parse_ordering(ordering_expr)

            subtasks = HDDLParser._parse_subtasks(subtasks_expr)
            if not ordering and subtasks:
                ordering = [
                    (subtasks[index].label, subtasks[index + 1].label)
                    for index in range(len(subtasks) - 1)
                ]

            methods.append(
                HDDLMethod(
                    name=name_match.group(1),
                    task_name=task_name,
                    task_args=task_args,
                    parameters=HDDLParser._parse_parameters(params_str),
                    precondition=HDDLParser._clean_formula(precondition_expr),
                    subtasks=subtasks,
                    ordering=ordering,
                )
            )
        return methods

    @staticmethod
    def _extract_actions(content: str) -> List[HDDLAction]:
        actions: List[HDDLAction] = []
        for block in HDDLParser._extract_blocks(content, "action"):
            name_match = re.match(r"\(:action\s+([^\s()]+)", block)
            if not name_match:
                continue

            params_expr = HDDLParser._extract_expression_after_keyword(block, ":parameters")
            params_str = params_expr[1:-1].strip() if params_expr else ""
            precondition_expr = HDDLParser._extract_expression_after_keyword(
                block,
                ":precondition",
            )
            effect_expr = HDDLParser._extract_expression_after_keyword(block, ":effect")

            actions.append(
                HDDLAction(
                    name=name_match.group(1),
                    parameters=HDDLParser._parse_parameters(params_str),
                    preconditions=HDDLParser._clean_formula(precondition_expr),
                    effects=HDDLParser._clean_formula(effect_expr),
                )
            )
        return actions

    @staticmethod
    def _extract_single_block(content: str, keyword: str) -> Optional[str]:
        blocks = HDDLParser._extract_blocks(content, keyword)
        if not blocks:
            return None
        return blocks[0]

    @staticmethod
    def _extract_blocks(content: str, keyword: str) -> List[str]:
        token = f"(:{keyword}"
        blocks: List[str] = []
        cursor = 0

        while True:
            start = content.find(token, cursor)
            if start == -1:
                break
            end = HDDLParser._find_matching_paren(content, start)
            blocks.append(content[start:end + 1])
            cursor = end + 1

        return blocks

    @staticmethod
    def _find_matching_paren(content: str, start_index: int) -> int:
        depth = 0
        for index in range(start_index, len(content)):
            char = content[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index
        raise ValueError(f"Unclosed HDDL block starting at index {start_index}")

    @staticmethod
    def _extract_expression_after_keyword(block: str, keyword: str) -> Optional[str]:
        start = block.find(keyword)
        if start == -1:
            return None

        cursor = start + len(keyword)
        while cursor < len(block) and block[cursor].isspace():
            cursor += 1

        if cursor >= len(block):
            return None

        if block[cursor] == "(":
            end = HDDLParser._find_matching_paren(block, cursor)
            return block[cursor:end + 1]

        end = cursor
        while end < len(block) and not block[end].isspace() and block[end] != ")":
            end += 1
        return block[cursor:end]

    @staticmethod
    def _parse_parameters(params_str: str) -> List[str]:
        names: List[str] = []
        if not params_str:
            return names

        tokens = [token for token in params_str.split() if token]
        current_vars: List[str] = []
        index = 0

        while index < len(tokens):
            token = tokens[index]
            if token == "-":
                type_name = tokens[index + 1] if index + 1 < len(tokens) else "object"
                for variable in current_vars:
                    names.append(f"{variable} - {type_name}")
                current_vars = []
                index += 2
                continue

            current_vars.append(token)
            index += 1

        for variable in current_vars:
            names.append(variable)

        return names

    @staticmethod
    def _clean_formula(formula: Optional[str]) -> str:
        if not formula:
            return "none"

        compact = " ".join(formula.split()).strip()
        if compact == "(and)":
            return "none"
        return compact

    @staticmethod
    def _parse_task_invocation(expression: Optional[str]) -> Tuple[str, List[str]]:
        if not expression:
            return "unknown_task", []

        text = expression.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1].strip()
        tokens = [token for token in text.split() if token]
        if not tokens:
            return "unknown_task", []
        return tokens[0], tokens[1:]

    @staticmethod
    def _parse_subtasks(expression: Optional[str]) -> List[HDDLSubtask]:
        if not expression:
            return []

        text = expression.strip()
        if text == "(and)":
            return []

        tree = _SimpleSExpressionParser.parse_expression(text)
        items = tree[1:] if isinstance(tree, list) and tree and tree[0] == "and" else [tree]

        subtasks: List[HDDLSubtask] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, list) or not item:
                continue

            label = f"s{index}"
            task_expr = item

            if len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], list):
                label = item[0]
                task_expr = item[1]

            if not isinstance(task_expr, list) or not task_expr:
                continue

            task_name = str(task_expr[0])
            args = [str(value) for value in task_expr[1:]]
            subtasks.append(HDDLSubtask(label=label, task_name=task_name, args=args))

        return subtasks

    @staticmethod
    def _parse_ordering(expression: Optional[str]) -> List[Tuple[str, str]]:
        if not expression:
            return []

        tree = _SimpleSExpressionParser.parse_expression(expression)
        items = tree[1:] if isinstance(tree, list) and tree and tree[0] == "and" else [tree]

        ordering: List[Tuple[str, str]] = []
        for item in items:
            if not isinstance(item, list) or len(item) != 3:
                continue
            if item[0] == "<":
                ordering.append((str(item[1]), str(item[2])))
            elif item[1] == "<":
                ordering.append((str(item[0]), str(item[2])))
        return ordering


class _SimpleSExpressionParser:
    """Tiny S-expression parser for the HDDL subset used here."""

    @staticmethod
    def tokenize(expression: str) -> List[str]:
        spaced = expression.replace("(", " ( ").replace(")", " ) ")
        return [token for token in spaced.split() if token]

    @classmethod
    def parse(cls, tokens: List[str], index: int = 0) -> Tuple[object, int]:
        if index >= len(tokens):
            raise ValueError("Unexpected end of HDDL expression")

        token = tokens[index]
        if token == "(":
            result: List[object] = []
            cursor = index + 1
            while cursor < len(tokens) and tokens[cursor] != ")":
                item, cursor = cls.parse(tokens, cursor)
                result.append(item)
            if cursor >= len(tokens):
                raise ValueError("Unclosed HDDL expression")
            return result, cursor + 1

        if token == ")":
            raise ValueError("Unexpected ')' in HDDL expression")

        return token, index + 1

    @classmethod
    def parse_expression(cls, expression: str) -> object:
        tokens = cls.tokenize(expression)
        if not tokens:
            return []
        tree, _ = cls.parse(tokens, 0)
        return tree


def test_hddl_parser() -> None:
    """Ad hoc parser smoke test."""
    domain_file = Path(__file__).parent.parent / "domains" / "blocksworld" / "domain.hddl"
    domain = HDDLParser.parse_domain(str(domain_file))
    print(f"Domain: {domain.name}")
    print(f"Tasks: {[task.name for task in domain.tasks]}")
    print(f"Methods: {[method.name for method in domain.methods]}")
    print(f"Actions: {[action.name for action in domain.actions]}")


if __name__ == "__main__":
    test_hddl_parser()
