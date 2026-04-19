"""
Stage 5 AgentSpeak lowering passes.

These passes rewrite rendered HTN method plans into a Jason-runnable subset by
compiling witness-dependent choices into explicit finite plan variants before
Stage 6 execution.
"""

from __future__ import annotations

import json
import re
from collections import Counter, deque
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple


class ASLMethodLowering:
    """Compile rendered HTN method plans into Jason-runnable specialisations."""

    def compile_method_plans(
        self,
        agentspeak_code: str,
        *,
        seed_facts: Sequence[str],
        runtime_objects: Sequence[str],
        object_types: Dict[str, str],
        type_parent_map: Dict[str, Optional[str]],
        method_library: Any | None = None,
        max_candidates_per_clause: int = 64,
        max_total_specialised_chunks: int = 4096,
    ) -> str:
        start_marker = "/* HTN Method Plans */"
        end_marker = "/* DFA Transition Wrappers */"
        start_index = agentspeak_code.find(start_marker)
        end_index = agentspeak_code.find(end_marker)
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            return agentspeak_code

        prefix = agentspeak_code[:start_index]
        section = agentspeak_code[start_index:end_index]
        suffix = agentspeak_code[end_index:]
        section_lines = section.splitlines()
        if not section_lines:
            return agentspeak_code

        header = section_lines[0]
        content_lines = section_lines[1:]
        chunks: List[List[str]] = []
        current: List[str] = []
        for line in content_lines:
            if not line.strip():
                if current:
                    chunks.append(current)
                    current = []
                continue
            current.append(line)
        if current:
            chunks.append(current)

        fact_index, type_domains = self._runtime_fact_index_for_local_witness_grounding(
            seed_facts=seed_facts,
            runtime_objects=runtime_objects,
            object_types=object_types,
            type_parent_map=type_parent_map,
        )
        callee_context_specs = self._callee_context_specs_from_agentspeak_prefix(prefix)
        for chunk_lines in chunks:
            self._record_callee_context_spec(chunk_lines, callee_context_specs)
        current_fact_arg_pair_scores = self._runtime_fact_arg_pair_scores(seed_facts)
        task_effect_predicates = self._task_effect_predicates_by_task(method_library)
        specialised_chunks: List[str] = []
        changed = False
        total_specialised_chunks = 0
        for chunk in chunks:
            specialised = self._specialise_method_chunk_local_witnesses(
                chunk,
                fact_index=fact_index,
                type_domains=type_domains,
                max_candidates_per_clause=max_candidates_per_clause,
            )
            next_specialised_count = total_specialised_chunks + max(0, len(specialised) - 1)
            if next_specialised_count > max_total_specialised_chunks:
                specialised = ["\n".join(chunk)]
            else:
                total_specialised_chunks = next_specialised_count
            if len(specialised) != 1 or specialised[0] != "\n".join(chunk):
                changed = True
            specialised_chunks.extend(specialised)

        pre_noop_specialisation_chunks = list(specialised_chunks)
        specialised_chunks = self._specialise_chunks_from_body_goal_contexts(
            specialised_chunks,
            callee_context_specs=callee_context_specs,
            max_candidates_per_chunk=max_candidates_per_clause,
        )
        specialised_chunks = self._preserve_chunk_growth_budget(
            pre_noop_specialisation_chunks,
            specialised_chunks,
            max_total_specialised_chunks=max_total_specialised_chunks,
        )
        if specialised_chunks != pre_noop_specialisation_chunks:
            changed = True
        pre_noop_specialisation_chunks = list(specialised_chunks)
        specialised_chunks = self._specialise_chunks_from_prefix_goal_support_contexts(
            specialised_chunks,
            callee_context_specs=callee_context_specs,
            max_candidates_per_chunk=max_candidates_per_clause,
        )
        specialised_chunks = self._preserve_chunk_growth_budget(
            pre_noop_specialisation_chunks,
            specialised_chunks,
            max_total_specialised_chunks=max_total_specialised_chunks,
        )
        if specialised_chunks != pre_noop_specialisation_chunks:
            changed = True
        pre_noop_specialisation_chunks = list(specialised_chunks)
        specialised_chunks = self._specialise_chunks_from_noop_body_support(
            specialised_chunks,
            fact_index=fact_index,
            type_domains=type_domains,
            max_candidates_per_chunk=max_candidates_per_clause,
        )
        specialised_chunks = self._preserve_chunk_growth_budget(
            pre_noop_specialisation_chunks,
            specialised_chunks,
            max_total_specialised_chunks=max_total_specialised_chunks,
        )
        if specialised_chunks != pre_noop_specialisation_chunks:
            changed = True
        pre_body_noop_context_chunks = list(specialised_chunks)
        specialised_chunks = self._specialise_chunks_from_body_noop_contexts(
            specialised_chunks,
            max_candidates_per_chunk=max_candidates_per_clause,
        )
        specialised_chunks = self._preserve_chunk_growth_budget(
            pre_body_noop_context_chunks,
            specialised_chunks,
            max_total_specialised_chunks=max_total_specialised_chunks,
        )
        if specialised_chunks != pre_body_noop_context_chunks:
            changed = True
        pre_recursive_source_guard_chunks = list(specialised_chunks)
        specialised_chunks = self._specialise_self_recursive_source_guards(
            specialised_chunks,
            fact_index=fact_index,
            type_domains=type_domains,
            max_candidates_per_chunk=max_candidates_per_clause,
        )
        specialised_chunks = self._preserve_chunk_growth_budget(
            pre_recursive_source_guard_chunks,
            specialised_chunks,
            max_total_specialised_chunks=max_total_specialised_chunks,
        )
        if specialised_chunks != pre_recursive_source_guard_chunks:
            changed = True
        pre_recursive_state_cleanup_chunks = list(specialised_chunks)
        specialised_chunks = self._strip_child_state_atoms_from_self_recursive_chunks(
            specialised_chunks,
        )
        if specialised_chunks != pre_recursive_state_cleanup_chunks:
            changed = True
        pre_noop_prefix_context_chunks = list(specialised_chunks)
        specialised_chunks = self._specialise_chunks_from_noop_prefix_contexts(
            specialised_chunks,
            max_candidates_per_chunk=max_candidates_per_clause,
        )
        specialised_chunks = self._preserve_chunk_growth_budget(
            pre_noop_prefix_context_chunks,
            specialised_chunks,
            max_total_specialised_chunks=max_total_specialised_chunks,
        )
        if specialised_chunks != pre_noop_prefix_context_chunks:
            changed = True
        pre_guard_promotion_chunks = list(specialised_chunks)
        specialised_chunks = self._promote_body_no_ancestor_guards_to_context(
            specialised_chunks,
        )
        if specialised_chunks != pre_guard_promotion_chunks:
            changed = True
        pre_blocked_goal_guard_chunks = list(specialised_chunks)
        specialised_chunks = self._guard_body_goals_against_blocked_runtime_goals(
            specialised_chunks,
        )
        if specialised_chunks != pre_blocked_goal_guard_chunks:
            changed = True
        pre_conflict_filter_chunks = list(specialised_chunks)
        specialised_chunks = self._discard_conflicting_head_context_chunks(
            specialised_chunks,
        )
        if specialised_chunks != pre_conflict_filter_chunks:
            changed = True
        pre_ordered_chunks = list(specialised_chunks)
        specialised_chunks = self._order_runtime_method_plan_chunks(
            specialised_chunks,
            fact_index=fact_index,
            current_fact_arg_pair_scores=current_fact_arg_pair_scores,
            type_domains=type_domains,
            task_effect_predicates=task_effect_predicates,
        )
        if specialised_chunks != pre_ordered_chunks:
            changed = True

        if not changed:
            return agentspeak_code

        rewritten_section = "\n\n".join([header, *specialised_chunks]).rstrip() + "\n\n"
        return f"{prefix}{rewritten_section}{suffix}"

    @staticmethod
    def _preserve_chunk_growth_budget(
        base_chunks: Sequence[str],
        expanded_chunks: Sequence[str],
        *,
        max_total_specialised_chunks: int,
    ) -> List[str]:
        base_list = list(base_chunks)
        expanded_list = list(expanded_chunks)
        if (
            max_total_specialised_chunks < 0
            or len(expanded_list) <= len(base_list) + max_total_specialised_chunks
        ):
            return expanded_list

        remaining_base = Counter(base_list)
        grouped_extras: List[List[str]] = []
        grouped_bases: List[str] = []
        pending_extras: List[str] = []
        for chunk in expanded_list:
            if remaining_base.get(chunk, 0) > 0:
                grouped_extras.append(pending_extras)
                grouped_bases.append(chunk)
                pending_extras = []
                remaining_base[chunk] -= 1
                continue
            pending_extras.append(chunk)

        for chunk in base_list:
            if remaining_base.get(chunk, 0) <= 0:
                continue
            grouped_extras.append([])
            grouped_bases.append(chunk)
            remaining_base[chunk] -= 1

        selected_extra_signatures: Set[Tuple[int, int]] = set()
        extra_offsets = [0 for _ in grouped_extras]
        selected_extra_count = 0
        while selected_extra_count < max_total_specialised_chunks:
            progressed = False
            for group_index, extras in enumerate(grouped_extras):
                offset = extra_offsets[group_index]
                if offset >= len(extras):
                    continue
                selected_extra_signatures.add((group_index, offset))
                extra_offsets[group_index] = offset + 1
                selected_extra_count += 1
                progressed = True
                if selected_extra_count >= max_total_specialised_chunks:
                    break
            if not progressed:
                break

        preserved_chunks: List[str] = []
        for group_index, base_chunk in enumerate(grouped_bases):
            for extra_index, extra_chunk in enumerate(grouped_extras[group_index]):
                if (group_index, extra_index) in selected_extra_signatures:
                    preserved_chunks.append(extra_chunk)
            preserved_chunks.append(base_chunk)

        return preserved_chunks

    @staticmethod
    @lru_cache(maxsize=131072)
    def _chunk_lines(chunk: str) -> Tuple[str, ...]:
        return tuple(str(chunk or "").splitlines())

    def _chunk_is_fully_grounded(self, chunk: str) -> bool:
        return not self._extract_asl_variables(chunk)

    def _specialise_chunks_from_noop_body_support(
        self,
        chunks: Sequence[str],
        *,
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        max_candidates_per_chunk: int,
    ) -> List[str]:
        if not chunks:
            return []

        noop_specs_by_task: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            if self._chunk_is_fully_grounded(chunk):
                continue
            lines = self._chunk_lines(chunk)
            if not lines:
                continue
            parsed_head = self._parse_asl_method_head(lines[0])
            if parsed_head is None or not self._chunk_is_noop_method_plan(lines[1:]):
                continue
            task_name, head_args, context_parts = parsed_head
            context_atoms: List[Dict[str, Any]] = []
            inequalities: List[Tuple[str, str]] = []
            for part in context_parts:
                parsed = self._parse_asl_context_conjunct(part)
                if parsed is None:
                    continue
                if parsed.get("kind") == "atom":
                    context_atoms.append(parsed)
                elif parsed.get("kind") == "inequality":
                    inequalities.append((str(parsed["lhs"]), str(parsed["rhs"])))
            noop_specs_by_task.setdefault(task_name, []).append(
                {
                    "head_args": head_args,
                    "context_atoms": tuple(context_atoms),
                    "inequalities": tuple(inequalities),
                },
            )

        expanded_chunks: List[str] = []
        noop_support_binding_cache: Dict[
            Tuple[
                str,
                Tuple[str, ...],
                Tuple[str, ...],
                Tuple[Tuple[str, Tuple[str, ...]], ...],
                Tuple[Tuple[str, str], ...],
                Tuple[Tuple[str, str], ...],
                Tuple[Tuple[str, str], ...],
                int,
            ],
            Tuple[Tuple[Tuple[str, str], ...], ...],
        ] = {}
        for chunk in chunks:
            if self._chunk_is_fully_grounded(chunk):
                expanded_chunks.append(chunk)
                continue
            lines = self._chunk_lines(chunk)
            if not lines:
                expanded_chunks.append(chunk)
                continue
            parsed_head = self._parse_asl_method_head(lines[0])
            if parsed_head is None:
                expanded_chunks.append(chunk)
                continue
            task_name, head_args, context_parts = parsed_head
            body_goals = [
                goal
                for goal in (
                    self._parse_asl_goal_call(line)
                    for line in lines[1:]
                )
                if goal is not None
            ]
            trigger_vars = {
                arg
                for arg in head_args
                if self._looks_like_asl_variable(arg)
            }
            allow_trigger_binding_specialisations = not self._is_transition_task_name(task_name)
            allow_chunk_binding_specialisations = len(body_goals) <= 1
            caller_type_constraints: List[Tuple[str, str]] = []
            caller_inequalities: List[Tuple[str, str]] = []
            for part in context_parts:
                parsed = self._parse_asl_context_conjunct(part)
                if parsed is None:
                    continue
                if parsed.get("kind") == "atom" and str(parsed.get("predicate")) == "object_type":
                    args = tuple(parsed.get("args") or ())
                    if len(args) == 2:
                        caller_type_constraints.append((str(args[0]), str(args[1])))
                elif parsed.get("kind") == "inequality":
                    caller_inequalities.append((str(parsed["lhs"]), str(parsed["rhs"])))

            candidate_chunks = [chunk]
            seen_chunks = {chunk}
            chunk_vars = self._extract_asl_variables(chunk)
            combined_bindings: List[Dict[str, str]] = [{}]
            seen_combined_bindings: set[Tuple[Tuple[str, str], ...]] = {()}
            for line in lines[1:]:
                goal = self._parse_asl_goal_call(line)
                if goal is None:
                    continue
                goal_support_bindings: List[Dict[str, str]] = []
                for spec in noop_specs_by_task.get(str(goal[0]), ()):
                    context_atoms = tuple(spec.get("context_atoms") or ())
                    support_cache_key = (
                        str(goal[0]),
                        tuple(goal[1]),
                        tuple(spec.get("head_args") or ()),
                        tuple(
                            (
                                str(atom.get("predicate") or ""),
                                tuple(str(arg) for arg in tuple(atom.get("args") or ())),
                            )
                            for atom in context_atoms
                        ),
                        tuple(spec.get("inequalities") or ()),
                        tuple(caller_type_constraints),
                        tuple(caller_inequalities),
                        int(max_candidates_per_chunk),
                    )
                    cached_signatures = noop_support_binding_cache.get(support_cache_key)
                    if cached_signatures is None:
                        cached_signatures = tuple(
                            tuple(sorted(binding.items()))
                            for binding in self._noop_support_bindings(
                                head_args=tuple(spec.get("head_args") or ()),
                                goal_args=tuple(goal[1]),
                                context_atoms=context_atoms,
                                inequalities=tuple(spec.get("inequalities") or ()),
                                fact_index=fact_index,
                                type_domains=type_domains,
                                caller_type_constraints=tuple(caller_type_constraints),
                                caller_inequalities=tuple(caller_inequalities),
                                max_bindings=max_candidates_per_chunk,
                            )
                        )
                        noop_support_binding_cache[support_cache_key] = cached_signatures
                    for binding_signature in cached_signatures:
                        binding = dict(binding_signature)
                        goal_support_bindings.append(binding)
                        if not allow_trigger_binding_specialisations:
                            continue
                        head_binding = {
                            variable: value
                            for variable, value in binding.items()
                            if variable in trigger_vars
                        }
                        if not head_binding:
                            continue
                        specialised_chunk = "\n".join(
                            self._substitute_asl_bindings(chunk_line, head_binding)
                            for chunk_line in lines
                        )
                        if specialised_chunk in seen_chunks:
                            continue
                        if len(candidate_chunks) >= max_candidates_per_chunk:
                            continue
                        seen_chunks.add(specialised_chunk)
                        candidate_chunks.append(specialised_chunk)
                if not goal_support_bindings:
                    continue
                next_combined_bindings = list(combined_bindings)
                if not allow_chunk_binding_specialisations:
                    continue
                for existing_binding in combined_bindings:
                    for support_binding in goal_support_bindings:
                        merged_binding = self._merge_asl_bindings(
                            existing_binding,
                            support_binding,
                        )
                        if merged_binding is None:
                            continue
                        signature = tuple(sorted(merged_binding.items()))
                        if signature in seen_combined_bindings:
                            continue
                        if len(next_combined_bindings) >= max_candidates_per_chunk:
                            continue
                        seen_combined_bindings.add(signature)
                        next_combined_bindings.append(merged_binding)
                combined_bindings = next_combined_bindings
            for binding in combined_bindings:
                chunk_binding = {
                    variable: value
                    for variable, value in binding.items()
                    if variable in chunk_vars
                    and (
                        allow_trigger_binding_specialisations
                        or variable not in trigger_vars
                    )
                }
                if not chunk_binding:
                    continue
                specialised_chunk = "\n".join(
                    self._substitute_asl_bindings(chunk_line, chunk_binding)
                    for chunk_line in lines
                )
                if specialised_chunk in seen_chunks:
                    continue
                if len(candidate_chunks) >= max_candidates_per_chunk:
                    continue
                seen_chunks.add(specialised_chunk)
                candidate_chunks.append(specialised_chunk)
            expanded_chunks.extend(candidate_chunks)

        return expanded_chunks

    def _specialise_chunks_from_body_noop_contexts(
        self,
        chunks: Sequence[str],
        *,
        max_candidates_per_chunk: int,
    ) -> List[str]:
        if not chunks:
            return []

        noop_specs_by_task: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            lines = self._chunk_lines(chunk)
            if not lines:
                continue
            parsed_head = self._parse_asl_method_head(lines[0])
            if parsed_head is None or not self._chunk_is_noop_method_plan(lines[1:]):
                continue
            task_name, head_args, context_parts = parsed_head
            noop_specs_by_task.setdefault(task_name, []).append(
                {
                    "head_args": head_args,
                    "context_parts": tuple(context_parts),
                },
            )
        if not noop_specs_by_task:
            return list(chunks)

        expanded_chunks: List[str] = []
        for chunk in chunks:
            lines = self._chunk_lines(chunk)
            if not lines:
                expanded_chunks.append(chunk)
                continue
            parsed_head = self._parse_asl_method_head(lines[0])
            if parsed_head is None or self._chunk_is_noop_method_plan(lines[1:]):
                expanded_chunks.append(chunk)
                continue
            chunk_vars = self._extract_asl_variables(chunk)
            candidate_chunks: List[str] = []
            seen_chunks = {chunk}
            for line in lines[1:]:
                goal = self._parse_asl_goal_call(line)
                if goal is None:
                    continue
                goal_name, goal_args = goal
                for spec in noop_specs_by_task.get(str(goal_name), ()):
                    extra_context = self._instantiate_noop_prefix_context(
                        head_args=tuple(spec.get("head_args") or ()),
                        goal_args=tuple(goal_args),
                        context_parts=tuple(spec.get("context_parts") or ()),
                        chunk_vars=chunk_vars,
                    )
                    extra_context = tuple(
                        part
                        for part in extra_context
                        if not str(part).startswith("not progress_target")
                        and not str(part).startswith("not blocked_runtime_goal")
                    )
                    if not extra_context:
                        continue
                    rewritten_head = self._append_asl_method_context_parts(
                        lines[0],
                        extra_context,
                    )
                    if rewritten_head == lines[0]:
                        continue
                    specialised_chunk = "\n".join([rewritten_head, *lines[1:]])
                    if specialised_chunk in seen_chunks:
                        continue
                    seen_chunks.add(specialised_chunk)
                    candidate_chunks.append(specialised_chunk)
                    if len(candidate_chunks) >= max_candidates_per_chunk:
                        break
                if len(candidate_chunks) >= max_candidates_per_chunk:
                    break
            expanded_chunks.extend(candidate_chunks)
            expanded_chunks.append(chunk)

        return expanded_chunks

    def _specialise_chunks_from_body_goal_contexts(
        self,
        chunks: Sequence[str],
        *,
        callee_context_specs: Dict[str, List[Dict[str, Any]]],
        max_candidates_per_chunk: int,
    ) -> List[str]:
        if not chunks or not callee_context_specs:
            return list(chunks)

        expanded_chunks: List[str] = []
        for chunk in chunks:
            if self._chunk_is_fully_grounded(chunk):
                expanded_chunks.append(chunk)
                continue
            lines = self._chunk_lines(chunk)
            if not lines:
                expanded_chunks.append(chunk)
                continue
            parsed_head = self._parse_asl_method_head(lines[0])
            if parsed_head is None:
                expanded_chunks.append(chunk)
                continue

            _, head_args, context_parts = parsed_head
            parsed_context_parts = tuple(
                parsed
                for parsed in (
                    self._parse_asl_context_conjunct(part)
                    for part in context_parts
                )
                if parsed is not None
            )
            body_goals = [
                goal
                for goal in (
                    self._parse_asl_goal_call(line)
                    for line in lines[1:]
                )
                if goal is not None
            ]
            if not body_goals:
                expanded_chunks.append(chunk)
                continue

            trigger_vars = {
                str(term)
                for term in tuple(head_args or ())
                if self._looks_like_asl_variable(str(term))
            }
            chunk_vars = frozenset(self._extract_asl_variables(chunk) - trigger_vars)
            if not chunk_vars:
                expanded_chunks.append(chunk)
                continue
            candidate_chunks: List[str] = []
            seen_chunks = {chunk}
            for goal_index, (goal_name, goal_args) in enumerate(body_goals):
                if goal_index != 0:
                    continue
                for spec in callee_context_specs.get(str(goal_name), ()):
                    candidate_bindings = self._body_goal_context_bindings(
                        goal_args=tuple(goal_args),
                        callee_head_args=tuple(spec.get("head_args") or ()),
                        callee_context_parts=tuple(spec.get("context_parts") or ()),
                        caller_context_parts=parsed_context_parts,
                        chunk_vars=chunk_vars,
                    )
                    for binding in candidate_bindings:
                        specialised_chunk = "\n".join(
                            self._substitute_asl_bindings(line, binding)
                            for line in lines
                        )
                        if specialised_chunk in seen_chunks:
                            continue
                        seen_chunks.add(specialised_chunk)
                        candidate_chunks.append(specialised_chunk)
                        if len(candidate_chunks) >= max_candidates_per_chunk:
                            break
                    if len(candidate_chunks) >= max_candidates_per_chunk:
                        break
                if len(candidate_chunks) >= max_candidates_per_chunk:
                    break

            expanded_chunks.extend(candidate_chunks)
            expanded_chunks.append(chunk)

        return expanded_chunks

    def _specialise_chunks_from_prefix_goal_support_contexts(
        self,
        chunks: Sequence[str],
        *,
        callee_context_specs: Dict[str, List[Dict[str, Any]]],
        max_candidates_per_chunk: int,
    ) -> List[str]:
        if not chunks or not callee_context_specs:
            return list(chunks)

        expanded_chunks: List[str] = []
        for chunk in chunks:
            if self._chunk_is_fully_grounded(chunk):
                expanded_chunks.append(chunk)
                continue
            lines = self._chunk_lines(chunk)
            if not lines:
                expanded_chunks.append(chunk)
                continue
            parsed_head = self._parse_asl_method_head(lines[0])
            if parsed_head is None:
                expanded_chunks.append(chunk)
                continue

            _, head_args, context_parts = parsed_head
            trigger_vars = {
                str(term)
                for term in tuple(head_args or ())
                if self._looks_like_asl_variable(str(term))
            }
            local_chunk_vars = frozenset(self._extract_asl_variables(chunk) - trigger_vars)
            if not local_chunk_vars:
                expanded_chunks.append(chunk)
                continue

            parsed_context_parts = tuple(
                parsed
                for parsed in (
                    self._parse_asl_context_conjunct(part)
                    for part in context_parts
                )
                if parsed is not None
            )
            body_goals = [
                goal
                for goal in (
                    self._parse_asl_goal_call(line)
                    for line in lines[1:]
                )
                if goal is not None
            ]
            if len(body_goals) < 2:
                expanded_chunks.append(chunk)
                continue

            candidate_chunks: List[str] = []
            seen_chunks = {chunk}
            for goal_index in range(1, len(body_goals)):
                previous_goal_args = tuple(str(arg) for arg in tuple(body_goals[goal_index - 1][1] or ()))
                goal_name, goal_args = body_goals[goal_index]
                current_goal_args = tuple(str(arg) for arg in tuple(goal_args or ()))
                unresolved_vars = [
                    arg
                    for arg in current_goal_args
                    if str(arg) in local_chunk_vars
                    and str(arg) not in previous_goal_args
                ]
                if len(unresolved_vars) != 1:
                    continue
                allowed_predicates = self._callee_context_predicates(
                    str(goal_name),
                    callee_context_specs,
                )
                candidate_bindings = self._prefix_goal_support_bindings(
                    unresolved_var=str(unresolved_vars[0]),
                    previous_goal_args=previous_goal_args,
                    caller_context_parts=parsed_context_parts,
                    allowed_predicates=allowed_predicates,
                )
                for binding in candidate_bindings:
                    specialised_chunk = "\n".join(
                        self._substitute_asl_bindings(line, binding)
                        for line in lines
                    )
                    if specialised_chunk in seen_chunks:
                        continue
                    seen_chunks.add(specialised_chunk)
                    candidate_chunks.append(specialised_chunk)
                    if len(candidate_chunks) >= max_candidates_per_chunk:
                        break
                if len(candidate_chunks) >= max_candidates_per_chunk:
                    break

            expanded_chunks.extend(candidate_chunks)
            expanded_chunks.append(chunk)

        return expanded_chunks

    def _specialise_self_recursive_source_guards(
        self,
        chunks: Sequence[str],
        *,
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        max_candidates_per_chunk: int,
    ) -> List[str]:
        if not chunks:
            return []

        noop_state_specs = self._noop_state_specs_by_task(chunks)
        if not noop_state_specs:
            return list(chunks)

        expanded_chunks: List[str] = []
        seen_global_chunks: set[str] = set()
        recursive_candidate_cap = max(max_candidates_per_chunk, 256)
        for chunk in chunks:
            candidate_chunks = self._self_recursive_source_guard_chunks(
                chunk,
                noop_state_specs=noop_state_specs,
                fact_index=fact_index,
                type_domains=type_domains,
                max_candidates_per_chunk=recursive_candidate_cap,
            )
            for candidate in candidate_chunks:
                if candidate in seen_global_chunks:
                    continue
                seen_global_chunks.add(candidate)
                expanded_chunks.append(candidate)
            if chunk not in seen_global_chunks:
                seen_global_chunks.add(chunk)
                expanded_chunks.append(chunk)
        return expanded_chunks

    def _noop_state_specs_by_task(
        self,
        chunks: Sequence[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        specs_by_task: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            if self._chunk_is_fully_grounded(chunk):
                continue
            lines = self._chunk_lines(chunk)
            if not lines:
                continue
            parsed_head = self._parse_asl_method_head(lines[0])
            if parsed_head is None or not self._chunk_is_noop_method_plan(lines[1:]):
                continue
            task_name, head_args, context_parts = parsed_head
            head_arg_set = set(head_args)
            for part in context_parts:
                parsed = self._parse_asl_context_conjunct(part)
                if parsed is None or parsed.get("kind") != "atom":
                    continue
                predicate = str(parsed.get("predicate") or "")
                args = tuple(str(arg) for arg in tuple(parsed.get("args") or ()))
                if not predicate or predicate in {"object", "object_type"}:
                    continue
                if not any(arg in head_arg_set for arg in args):
                    continue
                if any(
                    self._looks_like_asl_variable(arg) and arg not in head_arg_set
                    for arg in args
                ):
                    continue
                specs_by_task.setdefault(task_name, []).append(
                    {
                        "head_args": head_args,
                        "state_predicate": predicate,
                        "state_args": args,
                    },
                )
                break
        return specs_by_task

    def _self_recursive_source_guard_chunks(
        self,
        chunk: str,
        *,
        noop_state_specs: Dict[str, List[Dict[str, Any]]],
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        max_candidates_per_chunk: int,
    ) -> List[str]:
        if self._chunk_is_fully_grounded(chunk):
            return []
        lines = self._chunk_lines(chunk)
        if not lines:
            return []
        parsed_head = self._parse_asl_method_head(lines[0])
        if parsed_head is None or self._chunk_is_noop_method_plan(lines[1:]):
            return []

        task_name, head_args, context_parts = parsed_head
        state_specs = noop_state_specs.get(task_name)
        if not state_specs:
            return []

        body_goals = [
            goal
            for goal in (
                self._parse_asl_goal_call(line)
                for line in lines[1:]
            )
            if goal is not None
        ]
        self_goals = [
            goal
            for goal in body_goals
            if str(goal[0]) == task_name
        ]
        if len(self_goals) != 1:
            return []
        self_goal_args = tuple(self_goals[0][1])
        if len(self_goal_args) != len(head_args):
            return []

        progress_positions = [
            index
            for index, (head_arg, child_arg) in enumerate(zip(head_args, self_goal_args))
            if str(head_arg) != str(child_arg)
        ]
        if len(progress_positions) != 1:
            return []
        progress_index = progress_positions[0]
        for index, (head_arg, child_arg) in enumerate(zip(head_args, self_goal_args)):
            if index == progress_index:
                continue
            if str(head_arg) != str(child_arg):
                return []

        parent_progress_term = str(head_args[progress_index])
        child_progress_term = str(self_goal_args[progress_index])
        if (
            not self._looks_like_asl_variable(parent_progress_term)
            or not self._looks_like_asl_variable(child_progress_term)
        ):
            return []

        type_constraints: List[Tuple[str, str]] = []
        inequalities: List[Tuple[str, str]] = []
        binding_atoms: List[Dict[str, Any]] = []
        state_seed_context_parts: List[str] = []
        for part in context_parts:
            parsed = self._parse_asl_context_conjunct(part)
            if parsed is None:
                continue
            if parsed.get("kind") == "inequality":
                inequalities.append((str(parsed["lhs"]), str(parsed["rhs"])))
                continue
            if parsed.get("kind") != "atom":
                continue
            predicate = str(parsed.get("predicate") or "")
            args = tuple(str(arg) for arg in tuple(parsed.get("args") or ()))
            if predicate == "object_type" and len(args) == 2:
                type_constraints.append((args[0], args[1]))
                continue
            if predicate in {"object", "object_type"}:
                continue
            if self._self_recursive_binding_atom_is_state_seed(
                predicate=predicate,
                args=args,
                state_specs=state_specs,
                head_args=head_args,
                progress_index=progress_index,
                child_progress_term=child_progress_term,
            ):
                state_seed_context_parts.append(str(part))
                continue
            binding_atoms.append({"predicate": predicate, "args": args})

        if not binding_atoms:
            return []

        edge_vars = sorted(
            self._extract_asl_variables(
                "\n".join([lines[0], *[str(part) for part in context_parts], *lines[1:]]),
            ),
        )
        max_binding_candidates = max(max_candidates_per_chunk, 256)
        candidate_bindings = self._candidate_bindings_for_local_witnesses(
            binding_atoms=binding_atoms,
            type_constraints=type_constraints,
            inequalities=inequalities,
            local_vars=edge_vars,
            fact_index=fact_index,
            type_domains=type_domains,
            max_candidates_per_clause=max_binding_candidates,
        )
        if not candidate_bindings:
            return []

        edge_records_by_key: Dict[
            Tuple[str, ...],
            List[Dict[str, Any]],
        ] = {}
        adjacency_by_key: Dict[Tuple[str, ...], Dict[str, set[str]]] = {}
        for binding in candidate_bindings:
            parent_value = self._resolve_local_witness_term(parent_progress_term, binding)
            child_value = self._resolve_local_witness_term(child_progress_term, binding)
            if not parent_value or not child_value:
                continue
            if self._canonical_runtime_token(parent_value) == self._canonical_runtime_token(child_value):
                continue
            key_values: List[str] = []
            missing_key_value = False
            for index, head_arg in enumerate(head_args):
                if index == progress_index:
                    continue
                resolved = self._resolve_local_witness_term(str(head_arg), binding)
                if not resolved:
                    missing_key_value = True
                    break
                key_values.append(resolved)
            if missing_key_value:
                continue
            key = tuple(key_values)
            edge_records_by_key.setdefault(key, []).append(
                {
                    "binding": dict(binding),
                    "child": child_value,
                    "parent": parent_value,
                },
            )
            adjacency_by_key.setdefault(key, {}).setdefault(child_value, set()).add(parent_value)

        candidate_chunks_by_key: Dict[Tuple[str, ...], List[str]] = {}
        for key, edge_records in edge_records_by_key.items():
            adjacency = adjacency_by_key.get(key, {})
            if not adjacency:
                continue
            reachable_runtime_sources = set(
                self._forward_reachable_states(
                    adjacency,
                    self._self_recursive_seed_sources_for_key(
                        state_specs=state_specs,
                        noop_state_key=key,
                        progress_index=progress_index,
                        fact_index=fact_index,
                    ),
                ),
            )
            if not reachable_runtime_sources:
                continue
            distance_cache: Dict[str, Dict[str, int]] = {}
            key_seen_chunks: set[str] = set()
            key_candidate_chunks_by_target: Dict[str, List[str]] = {}
            key_candidate_count = 0
            sorted_edge_records = sorted(
                edge_records,
                key=lambda record: (
                    str(record["parent"]),
                    str(record["child"]),
                    tuple(sorted(record["binding"].items())),
                ),
            )
            for edge_record in sorted_edge_records:
                predecessor = str(edge_record["child"])
                target = str(edge_record["parent"])
                distances_to_target = distance_cache.setdefault(
                    target,
                    self._reverse_shortest_distances(adjacency, target),
                )
                distances_to_predecessor = distance_cache.setdefault(
                    predecessor,
                    self._reverse_shortest_distances(adjacency, predecessor),
                )
                predecessor_distance_to_target = distances_to_target.get(predecessor)
                if predecessor_distance_to_target is None:
                    continue
                source_items = sorted(
                    distances_to_predecessor.items(),
                    key=lambda item: (item[1], item[0]),
                )
                for source, distance_to_predecessor in source_items:
                    if source == target:
                        continue
                    if source not in reachable_runtime_sources:
                        continue
                    distance_to_target = distances_to_target.get(source)
                    if distance_to_target is None:
                        continue
                    if distance_to_target != distance_to_predecessor + 1:
                        continue
                    guard = self._source_state_guard_for_self_recursive_chunk(
                        state_specs=state_specs,
                        noop_state_key=key,
                        progress_index=progress_index,
                        source=source,
                        edge_binding=edge_record["binding"],
                    )
                    if guard is None:
                        continue
                    grounded_chunk = "\n".join(
                        self._substitute_asl_bindings(line, edge_record["binding"])
                        for line in lines
                    )
                    grounded_lines = self._chunk_lines(grounded_chunk)
                    grounded_state_seed_parts = tuple(
                        self._substitute_asl_bindings(part, edge_record["binding"])
                        for part in state_seed_context_parts
                    )
                    guarded_head = self._replace_asl_method_context_parts(
                        grounded_lines[0],
                        replace_context_parts=grounded_state_seed_parts,
                        replacement_context_parts=(guard,),
                    )
                    guarded_head = self._replace_self_recursive_state_contexts_with_guard(
                        guarded_head,
                        state_specs=state_specs,
                        noop_state_key=key,
                        progress_index=progress_index,
                        guard=guard,
                    )
                    guarded_chunk = "\n".join([guarded_head, *grounded_lines[1:]])
                    if guarded_chunk == chunk or guarded_chunk in key_seen_chunks:
                        continue
                    key_seen_chunks.add(guarded_chunk)
                    key_candidate_chunks_by_target.setdefault(target, []).append(guarded_chunk)
                    key_candidate_count += 1
                    if key_candidate_count >= max_candidates_per_chunk:
                        break
                if key_candidate_count >= max_candidates_per_chunk:
                    break
            if key_candidate_chunks_by_target:
                interleaved_key_chunks: List[str] = []
                target_offsets = {
                    target_name: 0
                    for target_name in key_candidate_chunks_by_target
                }
                ordered_targets = sorted(key_candidate_chunks_by_target)
                while len(interleaved_key_chunks) < max_candidates_per_chunk:
                    progressed = False
                    for target_name in ordered_targets:
                        candidates = key_candidate_chunks_by_target.get(target_name, [])
                        offset = target_offsets.get(target_name, 0)
                        if offset >= len(candidates):
                            continue
                        interleaved_key_chunks.append(candidates[offset])
                        target_offsets[target_name] = offset + 1
                        progressed = True
                        if len(interleaved_key_chunks) >= max_candidates_per_chunk:
                            break
                    if not progressed:
                        break
                if interleaved_key_chunks:
                    candidate_chunks_by_key[key] = interleaved_key_chunks

        if not candidate_chunks_by_key:
            return []

        interleaved_chunks: List[str] = []
        seen_global_chunks: set[str] = set()
        key_offsets = {
            key: 0
            for key in candidate_chunks_by_key
        }
        ordered_keys = sorted(candidate_chunks_by_key)
        while len(interleaved_chunks) < max_candidates_per_chunk:
            progressed = False
            for key in ordered_keys:
                candidates = candidate_chunks_by_key.get(key, [])
                offset = key_offsets.get(key, 0)
                while offset < len(candidates):
                    candidate = candidates[offset]
                    offset += 1
                    if candidate in seen_global_chunks:
                        continue
                    seen_global_chunks.add(candidate)
                    interleaved_chunks.append(candidate)
                    key_offsets[key] = offset
                    progressed = True
                    break
                else:
                    key_offsets[key] = offset
                if len(interleaved_chunks) >= max_candidates_per_chunk:
                    break
            if not progressed:
                break

        return interleaved_chunks

    def _strip_child_state_atoms_from_self_recursive_chunks(
        self,
        chunks: Sequence[str],
    ) -> List[str]:
        if not chunks:
            return []

        noop_state_specs = self._noop_state_specs_by_task(chunks)
        if not noop_state_specs:
            return list(chunks)

        rewritten_chunks: List[str] = []
        for chunk in chunks:
            lines = self._chunk_lines(chunk)
            if not lines:
                rewritten_chunks.append(chunk)
                continue
            parsed_head = self._parse_asl_method_head(lines[0])
            if parsed_head is None:
                rewritten_chunks.append(chunk)
                continue
            task_name, head_args, context_parts = parsed_head
            state_specs = noop_state_specs.get(str(task_name))
            if not state_specs:
                rewritten_chunks.append(chunk)
                continue
            body_goals = [
                goal
                for goal in (
                    self._parse_asl_goal_call(line)
                    for line in lines[1:]
                )
                if goal is not None
            ]
            self_goals = [
                goal
                for goal in body_goals
                if str(goal[0]) == str(task_name)
            ]
            if len(self_goals) != 1:
                rewritten_chunks.append(chunk)
                continue

            self_goal_args = tuple(str(arg) for arg in tuple(self_goals[0][1] or ()))
            if len(self_goal_args) != len(tuple(head_args or ())):
                rewritten_chunks.append(chunk)
                continue
            progress_positions = [
                index
                for index, (head_arg, child_arg) in enumerate(zip(head_args, self_goal_args))
                if str(head_arg) != str(child_arg)
            ]
            if len(progress_positions) != 1:
                rewritten_chunks.append(chunk)
                continue
            progress_index = progress_positions[0]
            child_progress_term = str(self_goal_args[progress_index])

            matching_state_parts: List[Tuple[str, Optional[str]]] = []
            for part in context_parts:
                parsed = self._parse_asl_context_conjunct(part)
                if parsed is None or parsed.get("kind") != "atom":
                    continue
                predicate = str(parsed.get("predicate") or "")
                args = tuple(str(arg) for arg in tuple(parsed.get("args") or ()))
                progress_term = self._self_recursive_state_progress_term(
                    predicate=predicate,
                    args=args,
                    state_specs=state_specs,
                    progress_index=progress_index,
                )
                if progress_term is None:
                    continue
                matching_state_parts.append((str(part), progress_term))

            if len(matching_state_parts) <= 1:
                rewritten_chunks.append(chunk)
                continue

            replacement_parts = tuple(
                part
                for part, progress_term in matching_state_parts
                if self._canonical_runtime_token(progress_term)
                != self._canonical_runtime_token(child_progress_term)
            )
            if not replacement_parts:
                replacement_parts = (matching_state_parts[0][0],)

            rewritten_head = self._replace_asl_method_context_parts(
                lines[0],
                replace_context_parts=tuple(part for part, _ in matching_state_parts),
                replacement_context_parts=replacement_parts,
            )
            rewritten_chunks.append("\n".join([rewritten_head, *lines[1:]]))

        return rewritten_chunks

    def _replace_self_recursive_state_contexts_with_guard(
        self,
        head_line: str,
        *,
        state_specs: Sequence[Dict[str, Any]],
        noop_state_key: Tuple[str, ...],
        progress_index: int,
        guard: str,
    ) -> str:
        parsed_head = self._parse_asl_method_head(head_line)
        if parsed_head is None:
            return head_line

        _, _, context_parts = parsed_head
        replace_context_parts: List[str] = []
        for part in context_parts:
            parsed = self._parse_asl_context_conjunct(part)
            if parsed is None or parsed.get("kind") != "atom":
                continue
            predicate = str(parsed.get("predicate") or "")
            args = tuple(str(arg) for arg in tuple(parsed.get("args") or ()))
            if not self._self_recursive_state_context_matches_key(
                predicate=predicate,
                args=args,
                state_specs=state_specs,
                noop_state_key=noop_state_key,
                progress_index=progress_index,
            ):
                continue
            replace_context_parts.append(str(part))

        if not replace_context_parts:
            return head_line

        return self._replace_asl_method_context_parts(
            head_line,
            replace_context_parts=tuple(replace_context_parts),
            replacement_context_parts=(guard,),
        )

    def _self_recursive_state_context_matches_key(
        self,
        *,
        predicate: str,
        args: Sequence[str],
        state_specs: Sequence[Dict[str, Any]],
        noop_state_key: Tuple[str, ...],
        progress_index: int,
    ) -> bool:
        for spec in state_specs:
            noop_head_args = tuple(str(arg) for arg in tuple(spec.get("head_args") or ()))
            state_predicate = str(spec.get("state_predicate") or "")
            state_args = tuple(str(arg) for arg in tuple(spec.get("state_args") or ()))
            if (
                not state_predicate
                or predicate != state_predicate
                or len(args) != len(state_args)
                or progress_index >= len(noop_head_args)
            ):
                continue
            key_index_by_arg = {
                arg: key_index
                for key_index, arg in enumerate(
                    arg
                    for index, arg in enumerate(noop_head_args)
                    if index != progress_index
                )
            }
            matched = True
            for state_arg, actual_arg in zip(state_args, args):
                if state_arg == noop_head_args[progress_index]:
                    continue
                key_index = key_index_by_arg.get(state_arg)
                if key_index is None or key_index >= len(noop_state_key):
                    matched = False
                    break
                if self._canonical_runtime_token(str(actual_arg)) != (
                    self._canonical_runtime_token(str(noop_state_key[key_index]))
                ):
                    matched = False
                    break
            if matched:
                return True
        return False

    def _self_recursive_state_progress_term(
        self,
        *,
        predicate: str,
        args: Sequence[str],
        state_specs: Sequence[Dict[str, Any]],
        progress_index: int,
    ) -> Optional[str]:
        for spec in state_specs:
            noop_head_args = tuple(str(arg) for arg in tuple(spec.get("head_args") or ()))
            state_predicate = str(spec.get("state_predicate") or "")
            state_args = tuple(str(arg) for arg in tuple(spec.get("state_args") or ()))
            if (
                not state_predicate
                or predicate != state_predicate
                or len(args) != len(state_args)
                or progress_index >= len(noop_head_args)
            ):
                continue
            head_index_by_arg = {
                str(arg): index
                for index, arg in enumerate(noop_head_args)
            }
            matched = True
            resolved_progress: Optional[str] = None
            for state_arg, actual_arg in zip(state_args, args):
                if state_arg == noop_head_args[progress_index]:
                    resolved_progress = str(actual_arg)
                    continue
                head_index = head_index_by_arg.get(state_arg)
                if head_index is None:
                    matched = False
                    break
                expected_arg = str(noop_head_args[head_index])
                if self._looks_like_asl_variable(expected_arg):
                    continue
                if self._canonical_runtime_token(str(actual_arg)) != (
                    self._canonical_runtime_token(expected_arg)
                ):
                    matched = False
                    break
            if matched and resolved_progress:
                return resolved_progress
        return None

    def _self_recursive_seed_sources_for_key(
        self,
        *,
        state_specs: Sequence[Dict[str, Any]],
        noop_state_key: Tuple[str, ...],
        progress_index: int,
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
    ) -> Tuple[str, ...]:
        seed_sources: set[str] = set()
        for spec in state_specs:
            noop_head_args = tuple(str(arg) for arg in tuple(spec.get("head_args") or ()))
            state_predicate = str(spec.get("state_predicate") or "")
            state_args = tuple(str(arg) for arg in tuple(spec.get("state_args") or ()))
            if not state_predicate or progress_index >= len(noop_head_args):
                continue
            facts = fact_index.get((state_predicate, len(state_args)), ())
            if not facts:
                continue
            key_index_by_arg = {
                arg: key_index
                for key_index, arg in enumerate(
                    arg
                    for index, arg in enumerate(noop_head_args)
                    if index != progress_index
                )
            }
            progress_arg = noop_head_args[progress_index]
            for fact_args in facts:
                resolved_progress: Optional[str] = None
                matched = True
                for arg, fact_value in zip(state_args, fact_args):
                    if arg == progress_arg:
                        if resolved_progress is None:
                            resolved_progress = str(fact_value)
                            continue
                        if self._canonical_runtime_token(resolved_progress) != (
                            self._canonical_runtime_token(str(fact_value))
                        ):
                            matched = False
                            break
                        continue
                    key_index = key_index_by_arg.get(arg)
                    if key_index is None or key_index >= len(noop_state_key):
                        matched = False
                        break
                    if self._canonical_runtime_token(str(fact_value)) != (
                        self._canonical_runtime_token(str(noop_state_key[key_index]))
                    ):
                        matched = False
                        break
                if matched and resolved_progress:
                    seed_sources.add(resolved_progress)
        return tuple(sorted(seed_sources))

    def _self_recursive_binding_atom_is_state_seed(
        self,
        *,
        predicate: str,
        args: Sequence[str],
        state_specs: Sequence[Dict[str, Any]],
        head_args: Sequence[str],
        progress_index: int,
        child_progress_term: str,
    ) -> bool:
        for spec in state_specs:
            noop_head_args = tuple(str(arg) for arg in tuple(spec.get("head_args") or ()))
            state_predicate = str(spec.get("state_predicate") or "")
            state_args = tuple(str(arg) for arg in tuple(spec.get("state_args") or ()))
            if (
                not state_predicate
                or predicate != state_predicate
                or len(args) != len(state_args)
                or progress_index >= len(noop_head_args)
            ):
                continue
            head_index_by_arg = {
                str(arg): index
                for index, arg in enumerate(noop_head_args)
            }
            matched = True
            for state_arg, actual_arg in zip(state_args, args):
                if state_arg == noop_head_args[progress_index]:
                    if str(actual_arg) != str(child_progress_term):
                        matched = False
                        break
                    continue
                head_index = head_index_by_arg.get(state_arg)
                if head_index is None or head_index >= len(head_args):
                    matched = False
                    break
                if str(actual_arg) != str(head_args[head_index]):
                    matched = False
                    break
            if matched:
                return True
        return False

    @staticmethod
    def _forward_reachable_states(
        adjacency: Dict[str, set[str]],
        seed_sources: Sequence[str],
    ) -> Tuple[str, ...]:
        if not seed_sources:
            return ()

        ordered_sources: List[str] = []
        seen_sources: set[str] = set()
        frontier: List[str] = []
        for source in seed_sources:
            rendered = str(source)
            if not rendered or rendered in seen_sources:
                continue
            seen_sources.add(rendered)
            ordered_sources.append(rendered)
            frontier.append(rendered)

        queue = deque(frontier)

        while queue:
            current = queue.popleft()
            for next_state in sorted(adjacency.get(current, ())):
                rendered = str(next_state)
                if not rendered or rendered in seen_sources:
                    continue
                seen_sources.add(rendered)
                ordered_sources.append(rendered)
                queue.append(rendered)

        return tuple(ordered_sources)

    def _guard_atom_exists_in_fact_index(
        self,
        guard: str,
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
    ) -> bool:
        parsed = self._parse_asl_context_conjunct(guard)
        if parsed is None or parsed.get("kind") != "atom":
            return False
        predicate = str(parsed.get("predicate") or "")
        args = tuple(str(arg) for arg in tuple(parsed.get("args") or ()))
        if not predicate:
            return False
        return args in fact_index.get((predicate, len(args)), ())

    def _source_state_guard_for_self_recursive_chunk(
        self,
        *,
        state_specs: Sequence[Dict[str, Any]],
        noop_state_key: Tuple[str, ...],
        progress_index: int,
        source: str,
        edge_binding: Dict[str, str],
    ) -> Optional[str]:
        for spec in state_specs:
            noop_head_args = tuple(str(arg) for arg in tuple(spec.get("head_args") or ()))
            state_args = tuple(str(arg) for arg in tuple(spec.get("state_args") or ()))
            if progress_index >= len(noop_head_args):
                continue
            key_index_by_arg = {
                arg: key_index
                for key_index, arg in enumerate(
                    arg
                    for index, arg in enumerate(noop_head_args)
                    if index != progress_index
                )
            }
            rendered_args: List[str] = []
            missing_arg = False
            for arg in state_args:
                if arg == noop_head_args[progress_index]:
                    rendered_args.append(source)
                    continue
                key_index = key_index_by_arg.get(arg)
                if key_index is not None and key_index < len(noop_state_key):
                    rendered_args.append(noop_state_key[key_index])
                    continue
                resolved = self._resolve_local_witness_term(arg, edge_binding)
                if resolved is None:
                    missing_arg = True
                    break
                rendered_args.append(resolved)
            if missing_arg:
                continue
            return self._format_asl_atom(str(spec.get("state_predicate") or ""), rendered_args)
        return None

    @staticmethod
    def _reverse_shortest_distances(
        adjacency: Dict[str, set[str]],
        target: str,
    ) -> Dict[str, int]:
        reverse_adjacency: Dict[str, set[str]] = {}
        for source, targets in adjacency.items():
            reverse_adjacency.setdefault(source, set())
            for next_node in targets:
                reverse_adjacency.setdefault(next_node, set()).add(source)

        distances: Dict[str, int] = {target: 0}
        frontier: List[str] = [target]
        while frontier:
            current = frontier.pop(0)
            next_distance = distances[current] + 1
            for predecessor in sorted(reverse_adjacency.get(current, ())):
                if predecessor in distances:
                    continue
                distances[predecessor] = next_distance
                frontier.append(predecessor)
        return distances

    @staticmethod
    def _format_asl_atom(predicate: str, args: Sequence[str]) -> str:
        name = str(predicate or "").strip()
        if not name:
            return ""
        rendered_args = ", ".join(str(arg) for arg in args)
        return f"{name}({rendered_args})"

    @staticmethod
    def _is_transition_task_name(task_name: str) -> bool:
        return str(task_name or "").strip().startswith("dfa_step_")

    def _specialise_chunks_from_noop_prefix_contexts(
        self,
        chunks: Sequence[str],
        *,
        max_candidates_per_chunk: int,
    ) -> List[str]:
        if not chunks:
            return []

        noop_contexts_by_task: Dict[str, List[Tuple[Tuple[str, ...], Tuple[str, ...]]]] = {}
        for chunk in chunks:
            lines = self._chunk_lines(chunk)
            if not lines:
                continue
            parsed_head = self._parse_asl_method_head(lines[0])
            if parsed_head is None or not self._chunk_is_noop_method_plan(lines[1:]):
                continue
            task_name, head_args, context_parts = parsed_head
            no_op_context_parts = tuple(
                part
                for part in context_parts
                if not str(part).strip().startswith("object_type(")
            )
            if not no_op_context_parts:
                continue
            noop_contexts_by_task.setdefault(task_name, []).append(
                (head_args, no_op_context_parts),
            )

        if not noop_contexts_by_task:
            return list(chunks)

        expanded_chunks: List[str] = []
        for chunk in chunks:
            lines = self._chunk_lines(chunk)
            if not lines:
                expanded_chunks.append(chunk)
                continue
            parsed_head = self._parse_asl_method_head(lines[0])
            if parsed_head is None or self._chunk_is_noop_method_plan(lines[1:]):
                expanded_chunks.append(chunk)
                continue
            parent_task_name = parsed_head[0]
            parent_context_parts = tuple(parsed_head[2] or ())

            chunk_vars = self._extract_asl_variables(chunk)
            prefix_context_variants: List[Tuple[str, ...]] = [()]
            candidate_chunks: List[str] = []
            seen_candidates = {chunk}
            for line in lines[1:]:
                goal = self._parse_asl_goal_call(line)
                if goal is None:
                    continue
                goal_task_name, goal_args = goal
                noop_specs = noop_contexts_by_task.get(goal_task_name)
                if not noop_specs:
                    break

                next_variants: List[Tuple[str, ...]] = []
                for prefix_context in prefix_context_variants:
                    for head_args, context_parts in noop_specs:
                        instantiated_context = self._instantiate_noop_prefix_context(
                            head_args=head_args,
                            goal_args=goal_args,
                            context_parts=context_parts,
                            chunk_vars=chunk_vars,
                        )
                        if not instantiated_context:
                            continue
                        if (
                            goal_task_name == parent_task_name
                            and self._self_recursive_noop_prefix_context_conflicts(
                                instantiated_context=instantiated_context,
                                head_context_parts=parent_context_parts,
                            )
                        ):
                            continue
                        combined_context = tuple(
                            dict.fromkeys([*prefix_context, *instantiated_context]),
                        )
                        if not combined_context or combined_context in next_variants:
                            continue
                        next_variants.append(combined_context)
                        if (
                            len(combined_context) < 2
                            and goal_task_name != parent_task_name
                        ):
                            continue
                        rewritten_head = self._append_asl_method_context_parts(
                            lines[0],
                            combined_context,
                        )
                        if rewritten_head == lines[0]:
                            continue
                        specialised_chunk = "\n".join([rewritten_head, *lines[1:]])
                        if specialised_chunk in seen_candidates:
                            continue
                        seen_candidates.add(specialised_chunk)
                        candidate_chunks.append(specialised_chunk)
                        if len(candidate_chunks) >= max_candidates_per_chunk:
                            break
                    if len(candidate_chunks) >= max_candidates_per_chunk:
                        break
                if not next_variants or len(candidate_chunks) >= max_candidates_per_chunk:
                    break
                prefix_context_variants = next_variants

            expanded_chunks.extend(candidate_chunks)
            expanded_chunks.append(chunk)
        return expanded_chunks

    def _self_recursive_noop_prefix_context_conflicts(
        self,
        *,
        instantiated_context: Sequence[str],
        head_context_parts: Sequence[str],
    ) -> bool:
        existing_atoms = [
            parsed
            for parsed in (
                self._parse_asl_context_conjunct(part)
                for part in tuple(head_context_parts or ())
            )
            if parsed is not None and parsed.get("kind") == "atom"
        ]
        if not existing_atoms:
            return False

        for part in tuple(instantiated_context or ()):
            parsed_candidate = self._parse_asl_context_conjunct(str(part))
            if parsed_candidate is None or parsed_candidate.get("kind") != "atom":
                continue
            candidate_predicate = str(parsed_candidate.get("predicate") or "")
            candidate_args = tuple(
                str(arg)
                for arg in tuple(parsed_candidate.get("args") or ())
            )
            if len(candidate_args) < 2:
                continue
            for parsed_existing in existing_atoms:
                existing_predicate = str(parsed_existing.get("predicate") or "")
                existing_args = tuple(
                    str(arg)
                    for arg in tuple(parsed_existing.get("args") or ())
                )
                if (
                    not candidate_predicate
                    or candidate_predicate != existing_predicate
                    or len(candidate_args) != len(existing_args)
                ):
                    continue
                equal_positions = sum(
                    1
                    for existing_arg, candidate_arg in zip(existing_args, candidate_args)
                    if self._canonical_runtime_token(existing_arg)
                    == self._canonical_runtime_token(candidate_arg)
                )
                if equal_positions >= len(candidate_args) - 1 and equal_positions < len(candidate_args):
                    return True
        return False

    def _promote_body_no_ancestor_guards_to_context(
        self,
        chunks: Sequence[str],
    ) -> List[str]:
        rewritten_chunks: List[str] = []
        for chunk in chunks:
            lines = self._chunk_lines(chunk)
            if not lines:
                rewritten_chunks.append(chunk)
                continue
            head_line = lines[0]
            guard_context_parts: List[str] = []
            for line in lines[1:]:
                statement = str(line or "").strip().rstrip(";.")
                if not statement.startswith("pipeline.no_ancestor_goal("):
                    continue
                if statement not in guard_context_parts:
                    guard_context_parts.append(statement)
            if not guard_context_parts:
                rewritten_chunks.append(chunk)
                continue
            rewritten_head = self._append_asl_method_context_parts(
                head_line,
                guard_context_parts,
            )
            if rewritten_head == head_line:
                rewritten_chunks.append(chunk)
                continue
            rewritten_chunks.append("\n".join([rewritten_head, *lines[1:]]))
        return rewritten_chunks

    def _discard_conflicting_head_context_chunks(
        self,
        chunks: Sequence[str],
    ) -> List[str]:
        retained_chunks: List[str] = []
        for chunk in chunks:
            if self._head_context_has_conflict(chunk):
                continue
            retained_chunks.append(chunk)
        return retained_chunks

    def _head_context_has_conflict(self, chunk: str) -> bool:
        lines = self._chunk_lines(chunk)
        if not lines:
            return False
        parsed_head = self._parse_asl_method_head(lines[0])
        if parsed_head is None:
            return False

        _task_name, _head_args, context_parts = parsed_head
        positive_atoms: Set[Tuple[str, Tuple[str, ...]]] = set()
        negative_atoms: Set[Tuple[str, Tuple[str, ...]]] = set()
        for part in tuple(context_parts or ()):
            parsed = self._parse_signed_asl_context_conjunct(part)
            if parsed is None:
                continue
            if parsed.get("kind") == "inequality":
                lhs = self._canonical_runtime_token(str(parsed.get("lhs") or ""))
                rhs = self._canonical_runtime_token(str(parsed.get("rhs") or ""))
                if lhs and rhs and lhs == rhs:
                    return True
                continue
            if parsed.get("kind") != "atom":
                continue
            predicate = str(parsed.get("predicate") or "").strip()
            args = tuple(
                self._canonical_runtime_token(str(arg))
                for arg in tuple(parsed.get("args") or ())
            )
            signature = (predicate, args)
            if bool(parsed.get("negated", False)):
                if signature in positive_atoms:
                    return True
                negative_atoms.add(signature)
                continue
            if signature in negative_atoms:
                return True
            positive_atoms.add(signature)
        return False

    def _guard_body_goals_against_blocked_runtime_goals(
        self,
        chunks: Sequence[str],
    ) -> List[str]:
        rewritten_chunks: List[str] = []
        for chunk in chunks:
            lines = self._chunk_lines(chunk)
            if not lines:
                rewritten_chunks.append(chunk)
                continue
            head_line = lines[0]
            blocked_goal_guards: List[str] = []
            for line in lines[1:]:
                goal = self._parse_asl_goal_call(line)
                if goal is None:
                    continue
                goal_name, goal_args = goal
                if str(goal_name).startswith("mark_target_"):
                    continue
                guard = self._blocked_runtime_goal_context(str(goal_name), tuple(goal_args))
                if guard not in blocked_goal_guards:
                    blocked_goal_guards.append(guard)
            if not blocked_goal_guards:
                rewritten_chunks.append(chunk)
                continue
            rewritten_head = self._append_asl_method_context_parts(
                head_line,
                blocked_goal_guards,
            )
            if rewritten_head == head_line:
                rewritten_chunks.append(chunk)
                continue
            rewritten_chunks.append("\n".join([rewritten_head, *lines[1:]]))
        return rewritten_chunks

    def _order_runtime_method_plan_chunks(
        self,
        chunks: Sequence[str],
        *,
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        current_fact_arg_pair_scores: Dict[Tuple[str, str], int],
        type_domains: Dict[str, Tuple[str, ...]],
        task_effect_predicates: Dict[str, FrozenSet[str]],
    ) -> List[str]:
        if not chunks:
            return []

        parsed_chunks: List[Dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            lines = self._chunk_lines(chunk)
            if not lines:
                parsed_chunks.append({"index": index, "chunk": chunk, "task_name": "", "sort_key": (0,)})
                continue
            parsed_head = self._parse_asl_method_head(lines[0])
            if parsed_head is None:
                parsed_chunks.append({"index": index, "chunk": chunk, "task_name": "", "sort_key": (0,)})
                continue
            task_name, head_args, context_parts = parsed_head
            body_lines = list(lines[1:])
            parsed_context_parts: List[Dict[str, Any]] = []
            caller_type_constraints: List[Tuple[str, str]] = []
            caller_inequalities: List[Tuple[str, str]] = []
            non_type_context_count = 0
            grounded_context_arg_count = 0
            for part in context_parts:
                rendered_part = str(part)
                if not rendered_part.strip().startswith("object_type("):
                    non_type_context_count += 1
                parsed_part = self._parse_asl_context_conjunct(rendered_part)
                if parsed_part is None:
                    continue
                parsed_context_parts.append(parsed_part)
                if parsed_part.get("kind") == "atom":
                    if str(parsed_part.get("predicate")) == "object_type":
                        args = tuple(parsed_part.get("args") or ())
                        if len(args) == 2:
                            caller_type_constraints.append((str(args[0]), str(args[1])))
                    grounded_context_arg_count += sum(
                        1
                        for arg in tuple(parsed_part.get("args") or ())
                        if not self._looks_like_asl_variable(str(arg))
                    )
                elif parsed_part.get("kind") == "inequality":
                    caller_inequalities.append((str(parsed_part["lhs"]), str(parsed_part["rhs"])))
                    grounded_context_arg_count += sum(
                        1
                        for arg in (str(parsed_part["lhs"]), str(parsed_part["rhs"]))
                        if not self._looks_like_asl_variable(arg)
                    )
            body_goals = [
                goal
                for goal in (
                    self._parse_asl_goal_call(line)
                    for line in body_lines
                )
                if goal is not None
            ]
            noop_body = self._chunk_is_noop_method_plan(body_lines)
            noop_effect_support = self._noop_context_supports_task_effect(
                task_name=task_name,
                head_args=head_args,
                parsed_context_parts=parsed_context_parts,
                task_effect_predicates=task_effect_predicates,
            )
            task_token_matches = self._task_name_grounded_token_match_count(
                task_name=task_name,
                head_args=head_args,
                parsed_context_parts=parsed_context_parts,
                body_goals=body_goals,
            )
            residual_var_count = self._residual_runtime_variable_count(
                head_args=head_args,
                parsed_context_parts=parsed_context_parts,
                body_goals=body_goals,
            )
            parsed_chunks.append(
                {
                    "index": index,
                    "chunk": chunk,
                    "task_name": task_name,
                    "head_args": head_args,
                    "context_parts": context_parts,
                    "parsed_context_parts": tuple(parsed_context_parts),
                    "caller_type_constraints": tuple(caller_type_constraints),
                    "caller_inequalities": tuple(caller_inequalities),
                    "non_type_context_count": non_type_context_count,
                    "grounded_context_arg_count": grounded_context_arg_count,
                    "body_lines": body_lines,
                    "body_goals": tuple(body_goals),
                    "noop_body": noop_body,
                    "noop_effect_support": noop_effect_support,
                    "task_token_match_count": task_token_matches,
                    "residual_var_count": residual_var_count,
                },
            )

        noop_specs_by_task: Dict[str, List[Dict[str, Any]]] = {}
        for item in parsed_chunks:
            if not bool(item.get("noop_body")):
                continue
            task_name = str(item.get("task_name") or "")
            if not task_name:
                continue
            context_atoms: List[Dict[str, Any]] = []
            inequalities: List[Tuple[str, str]] = []
            for parsed in item.get("parsed_context_parts") or ():
                if parsed.get("kind") == "atom":
                    context_atoms.append(parsed)
                elif parsed.get("kind") == "inequality":
                    inequalities.append((str(parsed["lhs"]), str(parsed["rhs"])))
            noop_specs_by_task.setdefault(task_name, []).append(
                {
                    "head_args": tuple(item.get("head_args") or ()),
                    "context_atoms": tuple(context_atoms),
                    "inequalities": tuple(inequalities),
                },
            )

        navigation_task_names: Set[str] = set()
        for task_name, specs in noop_specs_by_task.items():
            for spec in specs:
                context_atoms = tuple(spec.get("context_atoms") or ())
                non_type_atoms = [
                    atom
                    for atom in context_atoms
                    if str(atom.get("predicate") or "") != "object_type"
                ]
                if len(non_type_atoms) != 1:
                    continue
                if str(non_type_atoms[0].get("predicate") or "") == "at":
                    navigation_task_names.add(str(task_name))
                    break

        method_specs_by_task: Dict[str, List[Dict[str, Any]]] = {}
        for item in parsed_chunks:
            task_name = str(item.get("task_name") or "")
            if not task_name:
                continue
            method_specs_by_task.setdefault(task_name, []).append(item)
        compound_task_names = set(method_specs_by_task)
        noop_state_specs = self._noop_state_specs_by_task(chunks)

        task_navigation_output_specs_by_task: Dict[str, List[Dict[str, Any]]] = {}
        for item in parsed_chunks:
            task_name = str(item.get("task_name") or "")
            head_args = tuple(str(arg) for arg in tuple(item.get("head_args") or ()))
            if not task_name or not head_args:
                continue
            for goal_task_name, goal_args in tuple(item.get("body_goals") or ()):
                if str(goal_task_name) not in navigation_task_names:
                    continue
                output_arg_tuple = tuple(str(arg) for arg in tuple(goal_args or ()))
                if len(output_arg_tuple) < 2:
                    continue
                task_navigation_output_specs_by_task.setdefault(task_name, []).append(
                    {
                        "head_args": head_args,
                        "parsed_context_parts": tuple(item.get("parsed_context_parts") or ()),
                        "subject_arg": output_arg_tuple[0],
                        "location_arg": output_arg_tuple[1],
                    },
                )

        grouped_indexes: Dict[str, List[int]] = {}
        group_order: List[str] = []
        for index, item in enumerate(parsed_chunks):
            task_name = str(item.get("task_name") or f"__raw_{index}")
            if task_name not in grouped_indexes:
                grouped_indexes[task_name] = []
                group_order.append(task_name)
            grouped_indexes[task_name].append(index)

        ordered_chunks: List[str] = []
        noop_support_cache: Dict[
            Tuple[
                str,
                Tuple[str, ...],
                Tuple[Tuple[str, str], ...],
                Tuple[Tuple[str, str], ...],
            ],
            bool,
        ] = {}
        for task_name in group_order:
            group_items = [parsed_chunks[index] for index in grouped_indexes[task_name]]
            for item in group_items:
                body_lines = list(item.get("body_lines") or ())
                caller_type_constraints = tuple(item.get("caller_type_constraints") or ())
                caller_inequalities = tuple(item.get("caller_inequalities") or ())
                body_goals = list(item.get("body_goals") or ())
                weighted_noop_support = 0
                for goal_index, goal in enumerate(body_goals, start=1):
                    support_cache_key = (
                        str(goal[0]),
                        tuple(goal[1]),
                        tuple(caller_type_constraints),
                        tuple(caller_inequalities),
                    )
                    has_noop_support = noop_support_cache.get(support_cache_key)
                    if has_noop_support is None:
                        has_noop_support = self._goal_has_noop_runtime_support(
                            goal_task_name=str(goal[0]),
                            goal_args=tuple(goal[1]),
                            noop_specs_by_task=noop_specs_by_task,
                            fact_index=fact_index,
                            type_domains=type_domains,
                            caller_type_constraints=tuple(caller_type_constraints),
                            caller_inequalities=tuple(caller_inequalities),
                        )
                        noop_support_cache[support_cache_key] = has_noop_support
                    if has_noop_support:
                        weighted_noop_support += goal_index
                body_current_fact_pair_score = self._body_current_fact_pair_score(
                    body_goals,
                    current_fact_arg_pair_scores,
                )
                latent_navigation_handoff_score = self._latent_navigation_handoff_score(
                    body_goals=body_goals,
                    navigation_task_names=navigation_task_names,
                    task_navigation_output_specs_by_task=task_navigation_output_specs_by_task,
                    method_specs_by_task=method_specs_by_task,
                    noop_specs_by_task=noop_specs_by_task,
                    fact_index=fact_index,
                    type_domains=type_domains,
                    current_fact_arg_pair_scores=current_fact_arg_pair_scores,
                )
                first_body_goal_state_seed_support = self._first_body_goal_state_seed_support(
                    body_goals=body_goals,
                    parsed_context_parts=tuple(item.get("parsed_context_parts") or ()),
                    noop_state_specs=noop_state_specs,
                )
                leading_compound_followup = bool(body_goals) and (
                    str(body_goals[0][0]) in compound_task_names
                    and len(body_goals) > 1
                )
                leading_compound_followup_goal_count = (
                    len(body_goals)
                    if leading_compound_followup
                    else 0
                )
                grounded_head_arg_count = sum(
                    1
                    for arg in tuple(item.get("head_args") or ())
                    if not self._looks_like_asl_variable(str(arg))
                )
                non_type_context_count = int(item.get("non_type_context_count") or 0)
                grounded_context_arg_count = int(item.get("grounded_context_arg_count") or 0)
                task_token_match_count = int(item.get("task_token_match_count") or 0)
                residual_var_count = int(item.get("residual_var_count") or 0)
                noop_rank = 1
                if bool(item.get("noop_body")):
                    noop_rank = 0
                item["sort_key"] = (
                    noop_rank,
                    0 if not leading_compound_followup else 1,
                    -leading_compound_followup_goal_count,
                    -task_token_match_count,
                    -first_body_goal_state_seed_support,
                    residual_var_count,
                    -non_type_context_count,
                    -grounded_head_arg_count,
                    -grounded_context_arg_count,
                    -latent_navigation_handoff_score,
                    -body_current_fact_pair_score,
                    -weighted_noop_support,
                    int(item.get("index", 0)),
                )
            group_items.sort(key=lambda item: tuple(item.get("sort_key") or ()))
            ordered_chunks.extend(str(item["chunk"]) for item in group_items)

        return ordered_chunks

    def _first_body_goal_state_seed_support(
        self,
        *,
        body_goals: Sequence[Tuple[str, Tuple[str, ...]]],
        parsed_context_parts: Sequence[Dict[str, Any]],
        noop_state_specs: Dict[str, List[Dict[str, Any]]],
    ) -> int:
        if not body_goals:
            return 0

        goal_task_name, goal_args = body_goals[0]
        specs = noop_state_specs.get(str(goal_task_name), ())
        if not specs:
            return 0

        for spec in specs:
            head_args = tuple(str(arg) for arg in tuple(spec.get("head_args") or ()))
            if len(head_args) != len(tuple(goal_args or ())):
                continue
            head_binding = self._match_grounding_atom(
                head_args,
                tuple(str(arg) for arg in tuple(goal_args or ())),
                {},
            )
            if head_binding is None:
                continue
            state_predicate = str(spec.get("state_predicate") or "")
            state_args = tuple(
                head_binding.get(str(arg), str(arg))
                for arg in tuple(spec.get("state_args") or ())
            )
            for parsed_context in tuple(parsed_context_parts or ()):
                if parsed_context.get("kind") != "atom":
                    continue
                if str(parsed_context.get("predicate") or "") != state_predicate:
                    continue
                caller_args = tuple(str(arg) for arg in tuple(parsed_context.get("args") or ()))
                if self._match_grounding_atom(state_args, caller_args, {}) is not None:
                    return 1
        return 0

    def _body_navigation_support_score(
        self,
        *,
        body_goals: Sequence[Tuple[str, Tuple[str, ...]]],
        navigation_task_names: Set[str],
        method_specs_by_task: Dict[str, List[Dict[str, Any]]],
        noop_specs_by_task: Dict[str, List[Dict[str, Any]]],
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        cache: Dict[Tuple[str, Tuple[str, ...]], Optional[int]],
    ) -> Tuple[int, int]:
        if not body_goals or not navigation_task_names:
            return 0, 0

        unsupported_goal_count = 0
        support_depth_sum = 0
        for goal_task_name, goal_args in tuple(body_goals or ()):
            if str(goal_task_name) not in navigation_task_names:
                continue
            support_depth = self._navigation_goal_support_depth(
                goal_task_name=str(goal_task_name),
                goal_args=tuple(str(arg) for arg in tuple(goal_args or ())),
                navigation_task_names=navigation_task_names,
                method_specs_by_task=method_specs_by_task,
                noop_specs_by_task=noop_specs_by_task,
                fact_index=fact_index,
                type_domains=type_domains,
                cache=cache,
                active_goals=frozenset(),
            )
            if support_depth is None:
                unsupported_goal_count += 1
                continue
            support_depth_sum += support_depth
        return unsupported_goal_count, support_depth_sum

    def _navigation_goal_support_depth(
        self,
        *,
        goal_task_name: str,
        goal_args: Sequence[str],
        navigation_task_names: Set[str],
        method_specs_by_task: Dict[str, List[Dict[str, Any]]],
        noop_specs_by_task: Dict[str, List[Dict[str, Any]]],
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        cache: Dict[Tuple[str, Tuple[str, ...]], Optional[int]],
        active_goals: FrozenSet[Tuple[str, Tuple[str, ...]]],
    ) -> Optional[int]:
        canonical_goal = (
            str(goal_task_name),
            tuple(str(arg) for arg in tuple(goal_args or ())),
        )
        if canonical_goal in cache:
            return cache[canonical_goal]
        if canonical_goal in active_goals:
            return None

        if self._goal_has_noop_runtime_support(
            goal_task_name=canonical_goal[0],
            goal_args=canonical_goal[1],
            noop_specs_by_task=noop_specs_by_task,
            fact_index=fact_index,
            type_domains=type_domains,
            caller_type_constraints=(),
            caller_inequalities=(),
        ):
            cache[canonical_goal] = 0
            return 0

        best_depth: Optional[int] = None
        next_active_goals = active_goals | frozenset({canonical_goal})
        for spec in method_specs_by_task.get(canonical_goal[0], ()):
            if bool(spec.get("noop_body")):
                continue
            candidate_depth = self._navigation_method_support_depth(
                spec=spec,
                goal_args=canonical_goal[1],
                navigation_task_names=navigation_task_names,
                method_specs_by_task=method_specs_by_task,
                noop_specs_by_task=noop_specs_by_task,
                fact_index=fact_index,
                type_domains=type_domains,
                cache=cache,
                active_goals=next_active_goals,
            )
            if candidate_depth is None:
                continue
            if best_depth is None or candidate_depth < best_depth:
                best_depth = candidate_depth

        cache[canonical_goal] = best_depth
        return best_depth

    def _navigation_method_support_depth(
        self,
        *,
        spec: Dict[str, Any],
        goal_args: Sequence[str],
        navigation_task_names: Set[str],
        method_specs_by_task: Dict[str, List[Dict[str, Any]]],
        noop_specs_by_task: Dict[str, List[Dict[str, Any]]],
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        cache: Dict[Tuple[str, Tuple[str, ...]], Optional[int]],
        active_goals: FrozenSet[Tuple[str, Tuple[str, ...]]],
    ) -> Optional[int]:
        head_args = tuple(str(arg) for arg in tuple(spec.get("head_args") or ()))
        head_binding = self._match_grounding_atom(head_args, tuple(goal_args), {})
        if head_binding is None:
            return None

        instantiated_context_parts: List[Dict[str, Any]] = []
        for parsed_part in tuple(spec.get("parsed_context_parts") or ()):
            kind = str(parsed_part.get("kind") or "")
            if kind == "atom":
                instantiated_context_parts.append(
                    {
                        "kind": "atom",
                        "predicate": str(parsed_part.get("predicate") or ""),
                        "args": tuple(
                            head_binding.get(str(arg), str(arg))
                            for arg in tuple(parsed_part.get("args") or ())
                        ),
                    },
                )
                continue
            if kind == "inequality":
                instantiated_context_parts.append(
                    {
                        "kind": "inequality",
                        "lhs": head_binding.get(str(parsed_part.get("lhs") or ""), str(parsed_part.get("lhs") or "")),
                        "rhs": head_binding.get(str(parsed_part.get("rhs") or ""), str(parsed_part.get("rhs") or "")),
                    },
                )

        instantiated_body_goals = [
            (
                str(goal_task_name),
                tuple(
                    head_binding.get(str(arg), str(arg))
                    for arg in tuple(goal_args or ())
                ),
            )
            for goal_task_name, goal_args in tuple(spec.get("body_goals") or ())
        ]

        local_vars: Set[str] = set()
        for parsed_part in instantiated_context_parts:
            if parsed_part.get("kind") == "atom":
                for arg in tuple(parsed_part.get("args") or ()):
                    if self._looks_like_asl_variable(str(arg)):
                        local_vars.add(str(arg))
            elif parsed_part.get("kind") == "inequality":
                for term in (str(parsed_part.get("lhs") or ""), str(parsed_part.get("rhs") or "")):
                    if self._looks_like_asl_variable(term):
                        local_vars.add(term)
        for _, body_goal_args in instantiated_body_goals:
            for arg in tuple(body_goal_args or ()):
                if self._looks_like_asl_variable(str(arg)):
                    local_vars.add(str(arg))

        type_constraints: List[Tuple[str, str]] = []
        inequalities: List[Tuple[str, str]] = []
        ground_atoms: List[Dict[str, Any]] = []
        binding_atoms: List[Dict[str, Any]] = []
        for parsed_part in instantiated_context_parts:
            kind = str(parsed_part.get("kind") or "")
            if kind == "inequality":
                inequalities.append((str(parsed_part["lhs"]), str(parsed_part["rhs"])))
                continue
            if kind != "atom":
                continue
            predicate = str(parsed_part.get("predicate") or "")
            args = tuple(str(arg) for arg in tuple(parsed_part.get("args") or ()))
            if predicate == "object_type" and len(args) == 2:
                type_constraints.append((args[0], args[1]))
                continue
            atom_vars = {
                term
                for term in args
                if self._looks_like_asl_variable(term)
            }
            if atom_vars & local_vars:
                binding_atoms.append({"predicate": predicate, "args": args})
                continue
            ground_atoms.append({"predicate": predicate, "args": args})

        candidate_bindings: List[Dict[str, str]]
        if binding_atoms:
            binding_atoms.sort(
                key=lambda atom: (
                    len(fact_index.get((str(atom["predicate"]), len(tuple(atom["args"]))), ())),
                    -len(
                        [
                            term
                            for term in tuple(atom["args"])
                            if not self._looks_like_asl_variable(term)
                        ],
                    ),
                ),
            )
            candidate_bindings = self._candidate_bindings_for_local_witnesses(
                binding_atoms=binding_atoms,
                type_constraints=type_constraints,
                inequalities=inequalities,
                local_vars=tuple(sorted(local_vars)),
                fact_index=fact_index,
                type_domains=type_domains,
                max_candidates_per_clause=16,
            )
        else:
            candidate_bindings = [{}]

        best_depth: Optional[int] = None
        for binding in candidate_bindings:
            if not self._binding_satisfies_local_witness_filters(
                binding,
                type_constraints=type_constraints,
                type_domains=type_domains,
                inequalities=inequalities,
                local_vars=tuple(sorted(local_vars)),
                require_all_local_bindings=True,
            ):
                continue
            if not self._ground_context_atoms_hold(
                ground_atoms=ground_atoms,
                binding=binding,
                fact_index=fact_index,
            ):
                continue

            candidate_depth = 1
            supported = True
            for body_goal_task_name, body_goal_args in instantiated_body_goals:
                resolved_goal_args: List[str] = []
                for arg in tuple(body_goal_args or ()):
                    resolved = self._resolve_local_witness_term(str(arg), binding)
                    if resolved is None:
                        resolved = str(arg)
                    resolved_goal_args.append(str(resolved))
                if any(self._looks_like_asl_variable(arg) for arg in resolved_goal_args):
                    supported = False
                    break
                if str(body_goal_task_name) not in navigation_task_names:
                    continue
                body_goal_depth = self._navigation_goal_support_depth(
                    goal_task_name=str(body_goal_task_name),
                    goal_args=tuple(resolved_goal_args),
                    navigation_task_names=navigation_task_names,
                    method_specs_by_task=method_specs_by_task,
                    noop_specs_by_task=noop_specs_by_task,
                    fact_index=fact_index,
                    type_domains=type_domains,
                    cache=cache,
                    active_goals=active_goals,
                )
                if body_goal_depth is None:
                    supported = False
                    break
                candidate_depth += body_goal_depth
            if not supported:
                continue
            if best_depth is None or candidate_depth < best_depth:
                best_depth = candidate_depth
        return best_depth

    def _ground_context_atoms_hold(
        self,
        *,
        ground_atoms: Sequence[Dict[str, Any]],
        binding: Dict[str, str],
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
    ) -> bool:
        for atom in tuple(ground_atoms or ()):
            predicate = str(atom.get("predicate") or "")
            raw_args = tuple(str(arg) for arg in tuple(atom.get("args") or ()))
            resolved_args: List[str] = []
            for arg in raw_args:
                resolved = self._resolve_local_witness_term(arg, binding)
                if resolved is None:
                    resolved = arg
                if self._looks_like_asl_variable(str(resolved)):
                    return False
                resolved_args.append(str(resolved))
            facts = fact_index.get((predicate, len(resolved_args)), ())
            if not facts:
                return False
            if not any(
                self._match_grounding_atom(tuple(resolved_args), fact_args, {}) is not None
                for fact_args in facts
            ):
                return False
        return True

    def _latent_navigation_handoff_score(
        self,
        *,
        body_goals: Sequence[Tuple[str, Tuple[str, ...]]],
        navigation_task_names: Set[str],
        task_navigation_output_specs_by_task: Dict[str, List[Dict[str, Any]]],
        method_specs_by_task: Dict[str, List[Dict[str, Any]]],
        noop_specs_by_task: Dict[str, List[Dict[str, Any]]],
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        current_fact_arg_pair_scores: Dict[Tuple[str, str], int],
    ) -> int:
        if not body_goals or not navigation_task_names or not task_navigation_output_specs_by_task:
            return 0

        score = 0
        navigation_depth_cache_by_source: Dict[
            Tuple[str, str],
            Dict[Tuple[str, Tuple[str, ...]], Optional[int]],
        ] = {}
        for earlier_index, (goal_task_name, goal_args) in enumerate(tuple(body_goals or ())):
            output_pairs = self._task_navigation_output_pairs(
                goal_task_name=str(goal_task_name),
                goal_args=tuple(str(arg) for arg in tuple(goal_args or ())),
                task_navigation_output_specs_by_task=task_navigation_output_specs_by_task,
                fact_index=fact_index,
                type_domains=type_domains,
            )
            if not output_pairs:
                continue

            for later_goal_task_name, later_goal_args in tuple(body_goals[earlier_index + 1:]):
                if str(later_goal_task_name) not in navigation_task_names:
                    continue
                later_goal_arg_tuple = tuple(str(arg) for arg in tuple(later_goal_args or ()))
                if len(later_goal_arg_tuple) < 2:
                    continue
                later_subject = later_goal_arg_tuple[0]
                later_target = later_goal_arg_tuple[1]
                if (
                    self._looks_like_asl_variable(later_subject)
                    or self._looks_like_asl_variable(later_target)
                ):
                    continue
                canonical_later_subject = self._canonical_runtime_token(later_subject)
                canonical_later_target = self._canonical_runtime_token(later_target)
                best_pair_score: Optional[int] = None
                for output_subject, output_location in output_pairs:
                    if output_subject != canonical_later_subject:
                        continue
                    pair_score = current_fact_arg_pair_scores.get(
                        (output_location, canonical_later_target),
                        0,
                    )
                    source_key = (canonical_later_subject, output_location)
                    source_fact_index = self._fact_index_with_subject_location_override(
                        fact_index,
                        subject=canonical_later_subject,
                        location=output_location,
                    )
                    source_cache = navigation_depth_cache_by_source.setdefault(source_key, {})
                    navigation_depth = self._navigation_goal_support_depth(
                        goal_task_name=str(later_goal_task_name),
                        goal_args=tuple(later_goal_arg_tuple),
                        navigation_task_names=navigation_task_names,
                        method_specs_by_task=method_specs_by_task,
                        noop_specs_by_task=noop_specs_by_task,
                        fact_index=source_fact_index,
                        type_domains=type_domains,
                        cache=source_cache,
                        active_goals=frozenset(),
                    )
                    if navigation_depth is not None:
                        pair_score += max(1, 64 - navigation_depth)
                    else:
                        pair_score -= 128
                    if best_pair_score is None or pair_score > best_pair_score:
                        best_pair_score = pair_score
                if best_pair_score is not None:
                    score += best_pair_score
        return score

    def _task_navigation_output_pairs(
        self,
        *,
        goal_task_name: str,
        goal_args: Sequence[str],
        task_navigation_output_specs_by_task: Dict[str, List[Dict[str, Any]]],
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
    ) -> Set[Tuple[str, str]]:
        output_pairs: Set[Tuple[str, str]] = set()
        for spec in task_navigation_output_specs_by_task.get(str(goal_task_name), ()):
            head_args = tuple(str(arg) for arg in tuple(spec.get("head_args") or ()))
            head_binding = self._match_grounding_atom(head_args, tuple(goal_args), {})
            if head_binding is None:
                continue

            candidate_bindings = self._context_support_bindings(
                parsed_context_parts=tuple(spec.get("parsed_context_parts") or ()),
                base_binding=head_binding,
                extra_terms=(
                    str(spec.get("subject_arg") or ""),
                    str(spec.get("location_arg") or ""),
                ),
                fact_index=fact_index,
                type_domains=type_domains,
                max_candidates_per_clause=16,
            )
            for binding in candidate_bindings:
                subject_value = self._resolve_local_witness_term(
                    str(spec.get("subject_arg") or ""),
                    binding,
                )
                location_value = self._resolve_local_witness_term(
                    str(spec.get("location_arg") or ""),
                    binding,
                )
                if subject_value is None or location_value is None:
                    continue
                if self._looks_like_asl_variable(subject_value) or self._looks_like_asl_variable(location_value):
                    continue
                output_pairs.add(
                    (
                        self._canonical_runtime_token(subject_value),
                        self._canonical_runtime_token(location_value),
                    ),
                )
        return output_pairs

    def _context_support_bindings(
        self,
        *,
        parsed_context_parts: Sequence[Dict[str, Any]],
        base_binding: Dict[str, str],
        extra_terms: Sequence[str],
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        max_candidates_per_clause: int,
    ) -> List[Dict[str, str]]:
        instantiated_context_parts: List[Dict[str, Any]] = []
        for parsed_part in tuple(parsed_context_parts or ()):
            kind = str(parsed_part.get("kind") or "")
            if kind == "atom":
                instantiated_context_parts.append(
                    {
                        "kind": "atom",
                        "predicate": str(parsed_part.get("predicate") or ""),
                        "args": tuple(
                            base_binding.get(str(arg), str(arg))
                            for arg in tuple(parsed_part.get("args") or ())
                        ),
                    },
                )
                continue
            if kind == "inequality":
                instantiated_context_parts.append(
                    {
                        "kind": "inequality",
                        "lhs": base_binding.get(str(parsed_part.get("lhs") or ""), str(parsed_part.get("lhs") or "")),
                        "rhs": base_binding.get(str(parsed_part.get("rhs") or ""), str(parsed_part.get("rhs") or "")),
                    },
                )

        local_vars: Set[str] = set()
        for parsed_part in instantiated_context_parts:
            if parsed_part.get("kind") == "atom":
                for arg in tuple(parsed_part.get("args") or ()):
                    if self._looks_like_asl_variable(str(arg)):
                        local_vars.add(str(arg))
            elif parsed_part.get("kind") == "inequality":
                for term in (str(parsed_part.get("lhs") or ""), str(parsed_part.get("rhs") or "")):
                    if self._looks_like_asl_variable(term):
                        local_vars.add(term)
        for term in tuple(extra_terms or ()):
            resolved_term = base_binding.get(str(term), str(term))
            if self._looks_like_asl_variable(str(resolved_term)):
                local_vars.add(str(resolved_term))

        type_constraints: List[Tuple[str, str]] = []
        inequalities: List[Tuple[str, str]] = []
        ground_atoms: List[Dict[str, Any]] = []
        binding_atoms: List[Dict[str, Any]] = []
        for parsed_part in instantiated_context_parts:
            kind = str(parsed_part.get("kind") or "")
            if kind == "inequality":
                inequalities.append((str(parsed_part["lhs"]), str(parsed_part["rhs"])))
                continue
            if kind != "atom":
                continue
            predicate = str(parsed_part.get("predicate") or "")
            args = tuple(str(arg) for arg in tuple(parsed_part.get("args") or ()))
            if predicate == "object_type" and len(args) == 2:
                type_constraints.append((args[0], args[1]))
                continue
            atom_vars = {
                term
                for term in args
                if self._looks_like_asl_variable(term)
            }
            if atom_vars & local_vars:
                binding_atoms.append({"predicate": predicate, "args": args})
                continue
            ground_atoms.append({"predicate": predicate, "args": args})

        if binding_atoms:
            binding_atoms.sort(
                key=lambda atom: (
                    len(fact_index.get((str(atom["predicate"]), len(tuple(atom["args"]))), ())),
                    -len(
                        [
                            term
                            for term in tuple(atom["args"])
                            if not self._looks_like_asl_variable(term)
                        ],
                    ),
                ),
            )
            candidate_bindings = self._candidate_bindings_for_local_witnesses(
                binding_atoms=binding_atoms,
                type_constraints=type_constraints,
                inequalities=inequalities,
                local_vars=tuple(sorted(local_vars)),
                fact_index=fact_index,
                type_domains=type_domains,
                max_candidates_per_clause=max_candidates_per_clause,
            )
        else:
            candidate_bindings = [dict()]

        resolved_bindings: List[Dict[str, str]] = []
        for binding in candidate_bindings:
            merged_binding = dict(base_binding)
            merged_binding.update(binding)
            if not self._binding_satisfies_local_witness_filters(
                merged_binding,
                type_constraints=type_constraints,
                type_domains=type_domains,
                inequalities=inequalities,
                local_vars=tuple(sorted(local_vars)),
                require_all_local_bindings=True,
            ):
                continue
            if not self._ground_context_atoms_hold(
                ground_atoms=ground_atoms,
                binding=merged_binding,
                fact_index=fact_index,
            ):
                continue
            resolved_bindings.append(merged_binding)
        return resolved_bindings

    @staticmethod
    def _fact_index_with_subject_location_override(
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        *,
        subject: str,
        location: str,
    ) -> Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]]:
        overridden = dict(fact_index)
        at_facts = [
            fact_args
            for fact_args in tuple(fact_index.get(("at", 2), ()))
            if len(tuple(fact_args)) == 2
            and ASLMethodLowering._canonical_runtime_token(str(fact_args[0])) != ASLMethodLowering._canonical_runtime_token(subject)
        ]
        at_facts.append((subject, location))
        overridden[("at", 2)] = tuple(at_facts)
        return overridden

    def _task_name_grounded_token_match_count(
        self,
        *,
        task_name: str,
        head_args: Sequence[str],
        parsed_context_parts: Sequence[Dict[str, Any]],
        body_goals: Sequence[Tuple[str, Tuple[str, ...]]],
    ) -> int:
        if not str(task_name).startswith("dfa_step_"):
            return 0

        task_tokens = {
            token
            for token in re.split(r"[^A-Za-z0-9]+", str(task_name))
            if token
        }
        if not task_tokens:
            return 0

        matches: Set[str] = set()
        for arg in tuple(head_args or ()):
            rendered = str(arg)
            if self._looks_like_asl_variable(rendered):
                continue
            if rendered in task_tokens:
                matches.add(rendered)

        for parsed in tuple(parsed_context_parts or ()):
            kind = parsed.get("kind")
            if kind == "atom":
                for arg in tuple(parsed.get("args") or ()):
                    rendered = str(arg)
                    if self._looks_like_asl_variable(rendered):
                        continue
                    if rendered in task_tokens:
                        matches.add(rendered)
            elif kind == "inequality":
                for arg in (str(parsed.get("lhs") or ""), str(parsed.get("rhs") or "")):
                    if not arg or self._looks_like_asl_variable(arg):
                        continue
                    if arg in task_tokens:
                        matches.add(arg)

        for goal_task_name, goal_args in tuple(body_goals or ()):
            if goal_task_name in task_tokens:
                matches.add(str(goal_task_name))
            for arg in tuple(goal_args or ()):
                rendered = str(arg)
                if self._looks_like_asl_variable(rendered):
                    continue
                if rendered in task_tokens:
                    matches.add(rendered)

        return len(matches)

    def _residual_runtime_variable_count(
        self,
        *,
        head_args: Sequence[str],
        parsed_context_parts: Sequence[Dict[str, Any]],
        body_goals: Sequence[Tuple[str, Tuple[str, ...]]],
    ) -> int:
        residual_vars: Set[str] = set()

        for arg in tuple(head_args or ()):
            rendered = str(arg)
            if self._looks_like_asl_variable(rendered):
                residual_vars.add(rendered)

        for parsed in tuple(parsed_context_parts or ()):
            kind = str(parsed.get("kind") or "")
            if kind == "atom":
                for arg in tuple(parsed.get("args") or ()):
                    rendered = str(arg)
                    if self._looks_like_asl_variable(rendered):
                        residual_vars.add(rendered)
                continue
            if kind == "inequality":
                for term in (str(parsed.get("lhs") or ""), str(parsed.get("rhs") or "")):
                    if self._looks_like_asl_variable(term):
                        residual_vars.add(term)

        for _, goal_args in tuple(body_goals or ()):
            for arg in tuple(goal_args or ()):
                rendered = str(arg)
                if self._looks_like_asl_variable(rendered):
                    residual_vars.add(rendered)

        return len(residual_vars)

    def _specialise_method_chunk_local_witnesses(
        self,
        chunk: Sequence[str],
        *,
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        max_candidates_per_clause: int,
    ) -> List[str]:
        original = "\n".join(chunk)
        if not chunk:
            return [original]

        parsed_head = self._parse_asl_method_head(chunk[0])
        if parsed_head is None:
            return [original]
        _, head_args, context_parts = parsed_head
        chunk_text = "\n".join(chunk)
        trigger_vars = {
            term
            for term in head_args
            if self._looks_like_asl_variable(term)
        }
        all_vars = self._extract_asl_variables(chunk_text)
        local_vars = sorted(all_vars - trigger_vars)
        if not local_vars:
            return [original]

        body_goals = [
            goal
            for goal in (
                self._parse_asl_goal_call(line)
                for line in chunk[1:]
            )
            if goal is not None
        ]
        if any(
            str(goal_name) == str(parsed_head[0])
            and any(str(arg) in local_vars for arg in tuple(goal_args or ()))
            for goal_name, goal_args in body_goals
        ):
            return [original]

        type_constraints: List[Tuple[str, str]] = []
        inequalities: List[Tuple[str, str]] = []
        binding_atoms: List[Dict[str, Any]] = []
        local_var_set = set(local_vars)
        for part in context_parts:
            parsed = self._parse_asl_context_conjunct(part)
            if parsed is None:
                continue
            kind = parsed.get("kind")
            if kind == "inequality":
                inequalities.append((str(parsed["lhs"]), str(parsed["rhs"])))
                continue
            if kind != "atom":
                continue
            predicate = str(parsed["predicate"])
            args = tuple(str(arg) for arg in parsed["args"])
            if predicate == "object_type" and len(args) == 2:
                type_constraints.append((args[0], args[1]))
                continue
            atom_vars = {
                term
                for term in args
                if self._looks_like_asl_variable(term)
            }
            if atom_vars & local_var_set:
                binding_atoms.append({"predicate": predicate, "args": args})

        binding_atoms.sort(
            key=lambda atom: (
                len(fact_index.get((str(atom["predicate"]), len(tuple(atom["args"]))), ())),
                -len(
                    [
                        term
                        for term in tuple(atom["args"])
                        if not self._looks_like_asl_variable(term)
                    ],
                ),
            ),
        )
        if not binding_atoms:
            return [original]

        candidate_bindings = self._candidate_bindings_for_local_witnesses(
            binding_atoms=binding_atoms,
            type_constraints=type_constraints,
            inequalities=inequalities,
            local_vars=local_vars,
            fact_index=fact_index,
            type_domains=type_domains,
            max_candidates_per_clause=max_candidates_per_clause,
        )
        if not candidate_bindings:
            return [original]

        specialised_chunks: List[str] = []
        seen_chunks: set[str] = set()
        for binding in candidate_bindings:
            specialised_chunk = "\n".join(
                self._substitute_asl_bindings(line, binding)
                for line in chunk
            )
            if specialised_chunk == original or specialised_chunk in seen_chunks:
                continue
            seen_chunks.add(specialised_chunk)
            specialised_chunks.append(specialised_chunk)

        if not specialised_chunks:
            return [original]
        specialised_chunks.append(original)
        return specialised_chunks

    def _candidate_bindings_for_local_witnesses(
        self,
        *,
        binding_atoms: Sequence[Dict[str, Any]],
        type_constraints: Sequence[Tuple[str, str]],
        inequalities: Sequence[Tuple[str, str]],
        local_vars: Sequence[str],
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        max_candidates_per_clause: int,
    ) -> List[Dict[str, str]]:
        bindings: List[Dict[str, str]] = [{}]
        for atom in binding_atoms:
            next_bindings: List[Dict[str, str]] = []
            facts = fact_index.get((str(atom["predicate"]), len(tuple(atom["args"]))), ())
            if not facts:
                return []
            for binding in bindings:
                for fact_args in facts:
                    matched = self._match_grounding_atom(
                        tuple(atom["args"]),
                        fact_args,
                        binding,
                    )
                    if matched is None:
                        continue
                    if not self._binding_satisfies_local_witness_filters(
                        matched,
                        type_constraints=type_constraints,
                        type_domains=type_domains,
                        inequalities=inequalities,
                        local_vars=local_vars,
                        require_all_local_bindings=False,
                    ):
                        continue
                    next_bindings.append(matched)
            if not next_bindings:
                return []
            bindings = self._prune_local_witness_bindings(
                next_bindings,
                max_candidates=max_candidates_per_clause,
            )

        completed_bindings: List[Dict[str, str]] = [dict(binding) for binding in bindings]
        for variable in local_vars:
            expanded_bindings: List[Dict[str, str]] = []
            for binding in completed_bindings:
                if variable in binding:
                    expanded_bindings.append(dict(binding))
                    continue

                domain = self._local_witness_type_domain(
                    variable,
                    type_constraints=type_constraints,
                    type_domains=type_domains,
                )
                if domain is None or not domain:
                    continue

                for value in domain:
                    candidate = dict(binding)
                    candidate[variable] = value
                    if not self._binding_satisfies_local_witness_filters(
                        candidate,
                        type_constraints=type_constraints,
                        type_domains=type_domains,
                        inequalities=inequalities,
                        local_vars=local_vars,
                        require_all_local_bindings=False,
                    ):
                        continue
                    expanded_bindings.append(candidate)
            if not expanded_bindings:
                return []
            completed_bindings = self._prune_local_witness_bindings(
                expanded_bindings,
                max_candidates=max_candidates_per_clause,
            )

        unique: Dict[Tuple[Tuple[str, str], ...], Dict[str, str]] = {}
        local_var_set = set(local_vars)
        for binding in completed_bindings:
            if not self._binding_satisfies_local_witness_filters(
                binding,
                type_constraints=type_constraints,
                type_domains=type_domains,
                inequalities=inequalities,
                local_vars=local_vars,
                require_all_local_bindings=True,
            ):
                continue
            signature = tuple(sorted(
                (item, value)
                for item, value in binding.items()
                if item in local_var_set or value
            ))
            unique.setdefault(signature, binding)
        return list(unique.values())

    def _body_goal_context_bindings(
        self,
        *,
        goal_args: Sequence[str],
        callee_head_args: Sequence[str],
        callee_context_parts: Sequence[str],
        caller_context_parts: Sequence[Dict[str, Any]],
        chunk_vars: FrozenSet[str],
    ) -> List[Dict[str, str]]:
        if len(goal_args) != len(callee_head_args):
            return []

        head_binding = self._match_grounding_atom(
            tuple(callee_head_args),
            tuple(goal_args),
            {},
        )
        if head_binding is None:
            return []

        bindings: List[Dict[str, str]] = []
        seen_signatures: Set[Tuple[Tuple[str, str], ...]] = set()
        candidate_local_vars = set(chunk_vars)
        for part in callee_context_parts:
            parsed = self._parse_asl_context_conjunct(str(part))
            if parsed is None or parsed.get("kind") != "atom":
                continue
            predicate = str(parsed.get("predicate") or "")
            if predicate in {"object", "object_type"}:
                continue
            instantiated_args = tuple(
                head_binding.get(str(arg), str(arg))
                for arg in tuple(parsed.get("args") or ())
            )
            if not any(str(arg) in candidate_local_vars for arg in instantiated_args):
                continue
            for caller_part in caller_context_parts:
                if caller_part.get("kind") != "atom":
                    continue
                if str(caller_part.get("predicate") or "") != predicate:
                    continue
                caller_args = tuple(str(arg) for arg in tuple(caller_part.get("args") or ()))
                if len(caller_args) != len(instantiated_args):
                    continue
                binding = self._match_chunk_local_aliases(
                    pattern_args=instantiated_args,
                    context_args=caller_args,
                    candidate_local_vars=candidate_local_vars,
                )
                if not binding:
                    continue
                signature = tuple(sorted((str(key), str(value)) for key, value in binding.items()))
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                bindings.append(binding)
        return bindings

    def _prefix_goal_support_bindings(
        self,
        *,
        unresolved_var: str,
        previous_goal_args: Sequence[str],
        caller_context_parts: Sequence[Dict[str, Any]],
        allowed_predicates: Set[str],
    ) -> List[Dict[str, str]]:
        bindings: List[Dict[str, str]] = []
        seen_signatures: Set[Tuple[Tuple[str, str], ...]] = set()
        previous_goal_terms = {
            str(term)
            for term in tuple(previous_goal_args or ())
        }
        predicate_filter = {
            str(predicate).strip()
            for predicate in tuple(allowed_predicates or ())
            if str(predicate).strip() and str(predicate).strip() not in {"object", "object_type"}
        }
        for caller_part in caller_context_parts:
            if caller_part.get("kind") != "atom":
                continue
            predicate = str(caller_part.get("predicate") or "")
            if predicate in {"object", "object_type"}:
                continue
            if predicate_filter and predicate not in predicate_filter:
                continue
            caller_args = tuple(str(arg) for arg in tuple(caller_part.get("args") or ()))
            shared_terms = [arg for arg in caller_args if arg in previous_goal_terms]
            if not shared_terms:
                continue
            extra_terms = [arg for arg in caller_args if arg not in previous_goal_terms]
            if len(extra_terms) != 1:
                continue
            binding = {str(unresolved_var): str(extra_terms[0])}
            signature = tuple(sorted((str(key), str(value)) for key, value in binding.items()))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            bindings.append(binding)
        return bindings

    def _prune_local_witness_bindings(
        self,
        bindings: Sequence[Dict[str, str]],
        *,
        max_candidates: int,
    ) -> List[Dict[str, str]]:
        unique_bindings: List[Dict[str, str]] = []
        seen_signatures: Set[Tuple[Tuple[str, str], ...]] = set()
        for binding in bindings:
            signature = tuple(sorted((str(key), str(value)) for key, value in binding.items()))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_bindings.append(dict(binding))

        if max_candidates < 0 or len(unique_bindings) <= max_candidates:
            return unique_bindings

        variable_values: Dict[str, Set[str]] = {}
        for binding in unique_bindings:
            for variable, value in binding.items():
                variable_values.setdefault(str(variable), set()).add(str(value))

        pivot_variable = ""
        pivot_distinct = 1
        for variable, values in variable_values.items():
            if len(values) <= 1:
                continue
            if len(values) > pivot_distinct or (
                len(values) == pivot_distinct and variable < pivot_variable
            ):
                pivot_variable = variable
                pivot_distinct = len(values)

        if not pivot_variable:
            return unique_bindings[:max_candidates]

        grouped_bindings: Dict[str, deque[Dict[str, str]]] = {}
        group_order: List[str] = []
        for binding in unique_bindings:
            group_key = str(binding.get(pivot_variable, ""))
            if group_key not in grouped_bindings:
                grouped_bindings[group_key] = deque()
                group_order.append(group_key)
            grouped_bindings[group_key].append(dict(binding))

        pruned: List[Dict[str, str]] = []
        while len(pruned) < max_candidates and group_order:
            next_group_order: List[str] = []
            for group_key in group_order:
                bucket = grouped_bindings[group_key]
                if not bucket:
                    continue
                pruned.append(bucket.popleft())
                if len(pruned) >= max_candidates:
                    break
                if bucket:
                    next_group_order.append(group_key)
            group_order = next_group_order

        return pruned

    def _binding_satisfies_local_witness_filters(
        self,
        binding: Dict[str, str],
        *,
        type_constraints: Sequence[Tuple[str, str]],
        type_domains: Dict[str, Tuple[str, ...]],
        inequalities: Sequence[Tuple[str, str]],
        local_vars: Sequence[str],
        require_all_local_bindings: bool,
    ) -> bool:
        if require_all_local_bindings and any(var not in binding for var in local_vars):
            return False

        for term, type_name in type_constraints:
            resolved = self._resolve_local_witness_term(term, binding)
            if resolved is None:
                continue
            domain = type_domains.get(str(type_name))
            if domain is None or resolved not in domain:
                return False

        for lhs, rhs in inequalities:
            left_value = self._resolve_local_witness_term(lhs, binding)
            right_value = self._resolve_local_witness_term(rhs, binding)
            if left_value is None or right_value is None:
                continue
            if self._canonical_runtime_token(left_value) == self._canonical_runtime_token(right_value):
                return False
        return True

    def _local_witness_type_domain(
        self,
        variable: str,
        *,
        type_constraints: Sequence[Tuple[str, str]],
        type_domains: Dict[str, Tuple[str, ...]],
    ) -> Optional[Tuple[str, ...]]:
        required_types = [
            str(type_name)
            for term, type_name in type_constraints
            if term == variable
        ]
        if not required_types:
            return None

        domain_sets = [
            set(type_domains.get(type_name, ()))
            for type_name in required_types
        ]
        if not domain_sets:
            return ()
        domain = set.intersection(*domain_sets)
        return tuple(sorted(domain))

    @staticmethod
    def _match_grounding_atom(
        pattern_args: Sequence[str],
        fact_args: Sequence[str],
        binding: Dict[str, str],
    ) -> Optional[Dict[str, str]]:
        if len(pattern_args) != len(fact_args):
            return None

        candidate = dict(binding)
        for pattern_term, fact_term in zip(pattern_args, fact_args):
            if ASLMethodLowering._looks_like_asl_variable(pattern_term):
                existing = candidate.get(pattern_term)
                if existing is not None:
                    if ASLMethodLowering._canonical_runtime_token(existing) != (
                        ASLMethodLowering._canonical_runtime_token(fact_term)
                    ):
                        return None
                    continue
                candidate[pattern_term] = fact_term
                continue
            if ASLMethodLowering._canonical_runtime_token(pattern_term) != (
                ASLMethodLowering._canonical_runtime_token(fact_term)
            ):
                return None
        return candidate

    @staticmethod
    def _match_chunk_local_aliases(
        *,
        pattern_args: Sequence[str],
        context_args: Sequence[str],
        candidate_local_vars: Set[str],
    ) -> Optional[Dict[str, str]]:
        if len(pattern_args) != len(context_args):
            return None

        binding: Dict[str, str] = {}
        for pattern_term, context_term in zip(pattern_args, context_args):
            rendered_pattern = str(pattern_term)
            rendered_context = str(context_term)
            if (
                ASLMethodLowering._canonical_runtime_token(rendered_pattern)
                == ASLMethodLowering._canonical_runtime_token(rendered_context)
            ):
                continue
            if rendered_pattern in candidate_local_vars:
                existing = binding.get(rendered_pattern)
                if existing is not None:
                    if (
                        ASLMethodLowering._canonical_runtime_token(existing)
                        != ASLMethodLowering._canonical_runtime_token(rendered_context)
                    ):
                        return None
                    continue
                binding[rendered_pattern] = rendered_context
                continue
            return None

        return binding or None

    def _runtime_fact_index_for_local_witness_grounding(
        self,
        *,
        seed_facts: Sequence[str],
        runtime_objects: Sequence[str],
        object_types: Dict[str, str],
        type_parent_map: Dict[str, Optional[str]],
    ) -> Tuple[
        Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        Dict[str, Tuple[str, ...]],
    ]:
        facts_by_predicate: Dict[Tuple[str, int], set[Tuple[str, ...]]] = {}
        type_domains: Dict[str, set[str]] = {}

        for fact in seed_facts:
            atom = self._hddl_fact_to_atom(fact)
            parsed = self._parse_runtime_fact_atom(atom)
            if parsed is None:
                continue
            predicate, args = parsed
            facts_by_predicate.setdefault((predicate, len(args)), set()).add(args)

        for obj in runtime_objects:
            rendered_object = self._runtime_atom_term(str(obj))
            facts_by_predicate.setdefault(("object", 1), set()).add((rendered_object,))
            for type_name in self._type_closure(object_types.get(str(obj)), type_parent_map):
                type_atom = self._type_atom(type_name)
                facts_by_predicate.setdefault(("object_type", 2), set()).add(
                    (rendered_object, type_atom),
                )
                type_domains.setdefault(type_atom, set()).add(rendered_object)

        return (
            {
                key: tuple(sorted(values))
                for key, values in facts_by_predicate.items()
            },
            {
                type_name: tuple(sorted(values))
                for type_name, values in type_domains.items()
            },
        )

    @staticmethod
    @lru_cache(maxsize=131072)
    def _parse_runtime_fact_atom(atom: Optional[str]) -> Optional[Tuple[str, Tuple[str, ...]]]:
        text = str(atom or "").strip()
        if not text:
            return None
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)(?:\((.*)\))?", text)
        if match is None:
            return None
        predicate = match.group(1).strip()
        args_text = (match.group(2) or "").strip()
        if not args_text:
            return predicate, ()
        return predicate, ASLMethodLowering._split_asl_arguments(args_text)

    @staticmethod
    @lru_cache(maxsize=131072)
    def _parse_asl_method_head(
        head_line: str,
    ) -> Optional[Tuple[str, Tuple[str, ...], Tuple[str, ...]]]:
        match = re.match(
            r"^\s*\+!([^\s(:]+)(?:\(([^)]*)\))?\s*:\s*(.*?)\s*<-\s*$",
            head_line,
        )
        if match is None:
            return None
        task_name = match.group(1).strip()
        args_text = (match.group(2) or "").strip()
        context_text = (match.group(3) or "").strip()
        head_args = ASLMethodLowering._split_asl_arguments(args_text)
        context_parts = tuple(
            part.strip()
            for part in context_text.split("&")
            if part.strip()
        )
        return task_name, head_args, context_parts

    @staticmethod
    def _parse_asl_context_conjunct(part: str) -> Optional[Dict[str, Any]]:
        text = str(part or "").strip()
        if not text or text == "true":
            return None
        if "\\==" in text:
            lhs, rhs = text.split("\\==", 1)
            return {"kind": "inequality", "lhs": lhs.strip(), "rhs": rhs.strip()}
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)(?:\((.*)\))?", text)
        if match is None:
            return {"kind": "other", "text": text}
        predicate = match.group(1).strip()
        args_text = (match.group(2) or "").strip()
        args = ASLMethodLowering._split_asl_arguments(args_text) if args_text else ()
        return {"kind": "atom", "predicate": predicate, "args": args}

    @classmethod
    def _parse_signed_asl_context_conjunct(
        cls,
        part: str,
    ) -> Optional[Dict[str, Any]]:
        text = str(part or "").strip()
        if not text:
            return None
        negated = False
        if text.startswith("not "):
            negated = True
            text = text[4:].strip()
        parsed = cls._parse_asl_context_conjunct(text)
        if parsed is None:
            return None
        return {
            **parsed,
            "negated": negated,
        }

    @staticmethod
    def _task_effect_predicates_by_task(method_library: Any | None) -> Dict[str, FrozenSet[str]]:
        if method_library is None:
            return {}

        mapping: Dict[str, FrozenSet[str]] = {}
        for task in tuple(getattr(method_library, "compound_tasks", ()) or ()):
            task_name = str(getattr(task, "name", "") or "").strip()
            if not task_name:
                continue
            predicates = frozenset(
                str(predicate).strip()
                for predicate in tuple(getattr(task, "source_predicates", ()) or ())
                if str(predicate).strip()
            )
            if predicates:
                mapping[task_name] = predicates
        return mapping

    @staticmethod
    def _callee_context_predicates(
        task_name: str,
        callee_context_specs: Dict[str, List[Dict[str, Any]]],
    ) -> Set[str]:
        predicates: Set[str] = set()
        for spec in callee_context_specs.get(str(task_name), ()):
            for part in tuple(spec.get("context_parts") or ()):
                parsed = ASLMethodLowering._parse_asl_context_conjunct(str(part))
                if parsed is None or parsed.get("kind") != "atom":
                    continue
                predicate = str(parsed.get("predicate") or "")
                if predicate and predicate not in {"object", "object_type"}:
                    predicates.add(predicate)
        return predicates

    def _callee_context_specs_from_agentspeak_prefix(
        self,
        prefix: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        specs_by_task: Dict[str, List[Dict[str, Any]]] = {}
        current_chunk: List[str] = []
        for line in str(prefix or "").splitlines():
            if not line.strip():
                if current_chunk:
                    self._record_callee_context_spec(current_chunk, specs_by_task)
                    current_chunk = []
                continue
            if line.strip().startswith("/*"):
                if current_chunk:
                    self._record_callee_context_spec(current_chunk, specs_by_task)
                    current_chunk = []
                continue
            current_chunk.append(line)
        if current_chunk:
            self._record_callee_context_spec(current_chunk, specs_by_task)
        return specs_by_task

    def _record_callee_context_spec(
        self,
        chunk_lines: Sequence[str],
        specs_by_task: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        if not chunk_lines:
            return
        parsed_head = self._parse_asl_method_head(str(chunk_lines[0]))
        if parsed_head is None:
            return
        task_name, head_args, context_parts = parsed_head
        specs_by_task.setdefault(str(task_name), []).append(
            {
                "head_args": tuple(head_args),
                "context_parts": tuple(context_parts),
            },
        )

    def _noop_context_supports_task_effect(
        self,
        *,
        task_name: str,
        head_args: Sequence[str],
        parsed_context_parts: Sequence[Dict[str, Any]],
        task_effect_predicates: Dict[str, FrozenSet[str]],
    ) -> bool:
        effect_predicates = task_effect_predicates.get(str(task_name or "").strip())
        if not effect_predicates:
            return False

        head_signature = tuple(
            self._canonical_runtime_token(str(arg))
            for arg in tuple(head_args or ())
        )
        for parsed in tuple(parsed_context_parts or ()):
            if parsed.get("kind") != "atom":
                continue
            if str(parsed.get("predicate") or "").strip() not in effect_predicates:
                continue
            context_signature = tuple(
                self._canonical_runtime_token(str(arg))
                for arg in tuple(parsed.get("args") or ())
            )
            if context_signature == head_signature:
                return True
        return False

    @staticmethod
    def _chunk_is_noop_method_plan(body_lines: Sequence[str]) -> bool:
        statements = [
            line.strip().rstrip(";.")
            for line in body_lines
            if line.strip()
        ]
        if not statements:
            return False
        if statements[0].startswith('.print("runtime trace method flat "'):
            statements = statements[1:]
        if statements == ["true"]:
            return True
        if len(statements) != 1:
            return False
        statement = statements[0]
        return (
            statement == "!nop"
            or statement == "!noop"
            or statement.startswith("!nop(")
            or statement.startswith("!noop(")
        )

    @staticmethod
    @lru_cache(maxsize=131072)
    def _parse_asl_goal_call(line: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
        text = str(line or "").strip().rstrip(";.")
        if not text.startswith("!"):
            return None
        match = re.fullmatch(r"!([A-Za-z][A-Za-z0-9_]*)(?:\((.*)\))?", text)
        if match is None:
            return None
        task_name = match.group(1).strip()
        args_text = (match.group(2) or "").strip()
        args = ASLMethodLowering._split_asl_arguments(args_text) if args_text else ()
        return task_name, args

    def _goal_has_noop_runtime_support(
        self,
        *,
        goal_task_name: str,
        goal_args: Sequence[str],
        noop_specs_by_task: Dict[str, List[Dict[str, Any]]],
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        caller_type_constraints: Sequence[Tuple[str, str]] = (),
        caller_inequalities: Sequence[Tuple[str, str]] = (),
    ) -> bool:
        for spec in noop_specs_by_task.get(str(goal_task_name), ()):
            if self._noop_support_exists(
                head_args=tuple(spec.get("head_args") or ()),
                goal_args=tuple(goal_args),
                context_atoms=tuple(spec.get("context_atoms") or ()),
                inequalities=tuple(spec.get("inequalities") or ()),
                fact_index=fact_index,
                type_domains=type_domains,
                caller_type_constraints=caller_type_constraints,
                caller_inequalities=caller_inequalities,
            ):
                return True
        return False

    def _noop_support_exists(
        self,
        *,
        head_args: Sequence[str],
        goal_args: Sequence[str],
        context_atoms: Sequence[Dict[str, Any]],
        inequalities: Sequence[Tuple[str, str]],
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        caller_type_constraints: Sequence[Tuple[str, str]],
        caller_inequalities: Sequence[Tuple[str, str]],
    ) -> bool:
        if len(head_args) != len(goal_args):
            return False

        head_binding = {
            str(pattern): str(actual)
            for pattern, actual in zip(head_args, goal_args)
            if self._looks_like_asl_variable(str(pattern))
        }
        instantiated_atoms = [
            {
                "predicate": str(atom.get("predicate") or ""),
                "args": tuple(
                    head_binding.get(str(arg), str(arg))
                    for arg in tuple(atom.get("args") or ())
                ),
            }
            for atom in context_atoms
        ]
        instantiated_inequalities = tuple(
            (
                head_binding.get(str(lhs), str(lhs)),
                head_binding.get(str(rhs), str(rhs)),
            )
            for lhs, rhs in inequalities
        )
        combined_inequalities = instantiated_inequalities + tuple(caller_inequalities)
        instantiated_atoms.sort(
            key=lambda atom: len(
                fact_index.get((str(atom["predicate"]), len(tuple(atom["args"]))), ()),
            ),
        )

        def has_support(atom_index: int, binding: Dict[str, str]) -> bool:
            if atom_index >= len(instantiated_atoms):
                return self._binding_satisfies_local_witness_filters(
                    binding,
                    type_constraints=caller_type_constraints,
                    type_domains=type_domains,
                    inequalities=combined_inequalities,
                    local_vars=(),
                    require_all_local_bindings=False,
                )

            atom = instantiated_atoms[atom_index]
            facts = fact_index.get((str(atom["predicate"]), len(tuple(atom["args"]))), ())
            if not facts:
                return False

            for fact_args in facts:
                matched = self._match_grounding_atom(
                    tuple(atom["args"]),
                    fact_args,
                    binding,
                )
                if matched is None:
                    continue
                if not self._binding_satisfies_local_witness_filters(
                    matched,
                    type_constraints=caller_type_constraints,
                    type_domains=type_domains,
                    inequalities=combined_inequalities,
                    local_vars=(),
                    require_all_local_bindings=False,
                ):
                    continue
                if has_support(atom_index + 1, matched):
                    return True
            return False

        return has_support(0, {})

    def _noop_support_bindings(
        self,
        *,
        head_args: Sequence[str],
        goal_args: Sequence[str],
        context_atoms: Sequence[Dict[str, Any]],
        inequalities: Sequence[Tuple[str, str]],
        fact_index: Dict[Tuple[str, int], Tuple[Tuple[str, ...], ...]],
        type_domains: Dict[str, Tuple[str, ...]],
        caller_type_constraints: Sequence[Tuple[str, str]],
        caller_inequalities: Sequence[Tuple[str, str]],
        max_bindings: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        if len(head_args) != len(goal_args):
            return []

        head_binding = {
            str(pattern): str(actual)
            for pattern, actual in zip(head_args, goal_args)
            if self._looks_like_asl_variable(str(pattern))
        }
        instantiated_atoms: List[Dict[str, Any]] = []
        for atom in context_atoms:
            instantiated_atoms.append(
                {
                    "predicate": str(atom.get("predicate") or ""),
                    "args": tuple(
                        head_binding.get(str(arg), str(arg))
                        for arg in tuple(atom.get("args") or ())
                    ),
                },
            )
        instantiated_inequalities = tuple(
            (
                head_binding.get(str(lhs), str(lhs)),
                head_binding.get(str(rhs), str(rhs)),
            )
            for lhs, rhs in inequalities
        )
        combined_inequalities = instantiated_inequalities + tuple(caller_inequalities)
        instantiated_atoms.sort(
            key=lambda atom: len(
                fact_index.get((str(atom["predicate"]), len(tuple(atom["args"]))), ()),
            ),
        )

        collected_bindings: List[Dict[str, str]] = []
        seen_binding_signatures: set[Tuple[Tuple[str, str], ...]] = set()

        def collect_bindings(atom_index: int, binding: Dict[str, str]) -> None:
            if max_bindings is not None and len(collected_bindings) >= max_bindings:
                return
            if atom_index >= len(instantiated_atoms):
                if not self._binding_satisfies_local_witness_filters(
                    binding,
                    type_constraints=caller_type_constraints,
                    type_domains=type_domains,
                    inequalities=combined_inequalities,
                    local_vars=(),
                    require_all_local_bindings=False,
                ):
                    return
                signature = tuple(sorted(binding.items()))
                if signature in seen_binding_signatures:
                    return
                seen_binding_signatures.add(signature)
                collected_bindings.append(dict(binding))
                return

            atom = instantiated_atoms[atom_index]
            facts = fact_index.get((str(atom["predicate"]), len(tuple(atom["args"]))), ())
            if not facts:
                return

            for fact_args in facts:
                matched = self._match_grounding_atom(
                    tuple(atom["args"]),
                    fact_args,
                    binding,
                )
                if matched is None:
                    continue
                if not self._binding_satisfies_local_witness_filters(
                    matched,
                    type_constraints=caller_type_constraints,
                    type_domains=type_domains,
                    inequalities=combined_inequalities,
                    local_vars=(),
                    require_all_local_bindings=False,
                ):
                    continue
                collect_bindings(atom_index + 1, matched)
                if max_bindings is not None and len(collected_bindings) >= max_bindings:
                    return

        collect_bindings(0, {})
        return collected_bindings

    @staticmethod
    def _merge_asl_bindings(
        first: Dict[str, str],
        second: Dict[str, str],
    ) -> Optional[Dict[str, str]]:
        merged = dict(first)
        for variable, value in second.items():
            existing = merged.get(variable)
            if existing is not None and (
                ASLMethodLowering._canonical_runtime_token(existing)
                != ASLMethodLowering._canonical_runtime_token(value)
            ):
                return None
            merged[variable] = value
        return merged

    @staticmethod
    @lru_cache(maxsize=131072)
    def _extract_asl_variables(text: str) -> FrozenSet[str]:
        return frozenset(
            token
            for token in re.findall(r"\b[A-Z][A-Z0-9_]*\b", str(text or ""))
            if ASLMethodLowering._looks_like_asl_variable(token)
        )

    @staticmethod
    @lru_cache(maxsize=131072)
    def _looks_like_asl_variable(token: str) -> bool:
        return re.fullmatch(r"[A-Z][A-Z0-9_]*", str(token or "").strip()) is not None

    @staticmethod
    @lru_cache(maxsize=131072)
    def _split_asl_arguments(args_text: str) -> Tuple[str, ...]:
        text = str(args_text or "").strip()
        if not text:
            return ()
        parts: List[str] = []
        current: List[str] = []
        depth = 0
        quote: Optional[str] = None
        for character in text:
            if quote is not None:
                current.append(character)
                if character == quote:
                    quote = None
                continue
            if character in {'"', "'"}:
                quote = character
                current.append(character)
                continue
            if character == "(":
                depth += 1
                current.append(character)
                continue
            if character == ")":
                depth = max(0, depth - 1)
                current.append(character)
                continue
            if character == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(character)
        part = "".join(current).strip()
        if part:
            parts.append(part)
        return tuple(parts)

    def _resolve_local_witness_term(
        self,
        term: str,
        binding: Dict[str, str],
    ) -> Optional[str]:
        token = str(term or "").strip()
        if self._looks_like_asl_variable(token):
            return binding.get(token)
        return token

    def _substitute_asl_bindings(
        self,
        text: str,
        binding: Dict[str, str],
    ) -> str:
        rendered = str(text)
        for variable, value in sorted(binding.items(), key=lambda item: len(item[0]), reverse=True):
            if variable not in rendered:
                continue
            pattern = self._asl_binding_token_pattern(variable)
            rendered = pattern.sub(value, rendered)
        return rendered

    @staticmethod
    @lru_cache(maxsize=131072)
    def _asl_binding_token_pattern(variable: str) -> re.Pattern[str]:
        return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(str(variable))}(?![A-Za-z0-9_])")

    @staticmethod
    def _append_asl_method_context_parts(
        head_line: str,
        extra_context_parts: Sequence[str],
    ) -> str:
        if not extra_context_parts:
            return head_line
        match = re.match(r"^(\s*\+![^\s(:]+(?:\([^)]*\))?\s*:\s*)(.*?)(\s*<-\s*)$", head_line)
        if match is None:
            return head_line
        prefix, context_text, suffix = match.groups()
        context_parts = [
            part.strip()
            for part in str(context_text or "").split("&")
            if part.strip() and part.strip() != "true"
        ]
        seen = set(context_parts)
        for part in extra_context_parts:
            rendered = str(part or "").strip()
            if not rendered or rendered in seen:
                continue
            context_parts.append(rendered)
            seen.add(rendered)
        rewritten_context = " & ".join(context_parts) if context_parts else "true"
        return f"{prefix}{rewritten_context}{suffix}"

    @staticmethod
    def _replace_asl_method_context_parts(
        head_line: str,
        *,
        replace_context_parts: Sequence[str],
        replacement_context_parts: Sequence[str],
    ) -> str:
        match = re.match(r"^(\s*\+![^\s(:]+(?:\([^)]*\))?\s*:\s*)(.*?)(\s*<-\s*)$", head_line)
        if match is None:
            return head_line
        prefix, context_text, suffix = match.groups()
        replace_set = {
            str(part).strip()
            for part in replace_context_parts
            if str(part).strip()
        }
        context_parts = [
            part.strip()
            for part in str(context_text or "").split("&")
            if part.strip() and part.strip() != "true"
        ]
        rewritten_parts: List[str] = []
        seen_parts: set[str] = set()
        for part in context_parts:
            if part in replace_set:
                continue
            if part in seen_parts:
                continue
            rewritten_parts.append(part)
            seen_parts.add(part)
        for part in replacement_context_parts:
            rendered = str(part or "").strip()
            if not rendered or rendered in seen_parts:
                continue
            rewritten_parts.append(rendered)
            seen_parts.add(rendered)
        rewritten_context = " & ".join(rewritten_parts) if rewritten_parts else "true"
        return f"{prefix}{rewritten_context}{suffix}"

    def _instantiate_noop_prefix_context(
        self,
        *,
        head_args: Sequence[str],
        goal_args: Sequence[str],
        context_parts: Sequence[str],
        chunk_vars: Set[str],
    ) -> Tuple[str, ...]:
        if len(head_args) != len(goal_args):
            return ()
        binding: Dict[str, str] = {}
        for head_arg, goal_arg in zip(head_args, goal_args):
            if self._looks_like_asl_variable(str(head_arg)):
                binding[str(head_arg)] = str(goal_arg)
                continue
            if self._looks_like_asl_variable(str(goal_arg)):
                continue
            if self._canonical_runtime_token(str(head_arg)) != (
                self._canonical_runtime_token(str(goal_arg))
            ):
                return ()

        instantiated_parts: List[str] = []
        for part in context_parts:
            substituted = self._substitute_asl_bindings(str(part), binding).strip()
            if (
                not substituted
                or substituted == "true"
                or substituted.startswith("object_type(")
            ):
                continue
            introduced_vars = self._extract_asl_variables(substituted) - set(chunk_vars)
            if introduced_vars:
                return ()
            instantiated_parts.append(substituted)
        return tuple(dict.fromkeys(instantiated_parts))

    def _runtime_fact_arg_pair_scores(
        self,
        seed_facts: Sequence[str],
    ) -> Dict[Tuple[str, str], int]:
        pair_scores: Dict[Tuple[str, str], int] = {}
        for fact_index, fact in enumerate(tuple(seed_facts or ()), start=1):
            atom = self._hddl_fact_to_atom(fact)
            parsed = self._parse_runtime_fact_atom(atom)
            if parsed is None:
                continue
            predicate, args = parsed
            if predicate in {"object", "object_type"}:
                continue
            ground_args = [
                self._canonical_runtime_token(str(arg))
                for arg in tuple(args)
                if not self._looks_like_asl_variable(str(arg))
            ]
            for left_index, left in enumerate(ground_args):
                for right in ground_args[left_index + 1:]:
                    if left == right:
                        continue
                    pair_scores[(left, right)] = fact_index
                    pair_scores[(right, left)] = fact_index
        return pair_scores

    def _body_current_fact_pair_score(
        self,
        body_goals: Sequence[Tuple[str, Tuple[str, ...]]],
        current_fact_arg_pair_scores: Dict[Tuple[str, str], int],
    ) -> int:
        score = 0
        for _, goal_args in body_goals:
            ground_args = [
                self._canonical_runtime_token(str(arg))
                for arg in tuple(goal_args)
                if not self._looks_like_asl_variable(str(arg))
            ]
            matched_pairs: Set[Tuple[str, str]] = set()
            for left_index, left in enumerate(ground_args):
                for right in ground_args[left_index + 1:]:
                    if left == right:
                        continue
                    pair = (left, right)
                    if pair in current_fact_arg_pair_scores:
                        matched_pairs.add(pair)
            score += sum(current_fact_arg_pair_scores.get(pair, 0) for pair in matched_pairs)
        return score

    @staticmethod
    def _type_atom(type_name: str) -> str:
        return ASLMethodLowering._sanitize_name(str(type_name or "object")).lower() or "object"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip()).strip("_") or "term"

    @staticmethod
    def _asl_string(text: str) -> str:
        return json.dumps(str(text))

    @classmethod
    def _asl_atom_or_string(cls, text: str) -> str:
        token = str(text).strip()
        if re.fullmatch(r"[a-z][a-z0-9_]*", token):
            return token
        return cls._asl_string(token)

    @classmethod
    def _runtime_atom_term(cls, text: str) -> str:
        token = str(text).strip()
        if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
            return token
        return cls._asl_atom_or_string(token)

    @classmethod
    def _blocked_runtime_goal_context(
        cls,
        goal_name: str,
        goal_args: Sequence[str],
    ) -> str:
        terms = [cls._asl_atom_or_string(goal_name)]
        terms.extend(str(arg) for arg in goal_args)
        return f"not blocked_runtime_goal({', '.join(terms)})"

    @classmethod
    def _type_closure(
        cls,
        type_name: Optional[str],
        type_parent_map: Dict[str, Optional[str]],
    ) -> Tuple[str, ...]:
        if not type_name:
            return ()

        closure: List[str] = []
        visited: set[str] = set()
        cursor: Optional[str] = str(type_name).strip()
        while cursor and cursor not in visited:
            visited.add(cursor)
            if cursor != "object":
                closure.append(cursor)
            cursor = type_parent_map.get(cursor)
        return tuple(closure)

    @staticmethod
    def _canonical_runtime_token(token: str) -> str:
        value = str(token or "").strip()
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            return value[1:-1]
        return value

    @staticmethod
    @lru_cache(maxsize=131072)
    def _hddl_fact_to_atom(fact: str) -> Optional[str]:
        text = (fact or "").strip()
        if not text.startswith("(") or not text.endswith(")"):
            return None
        inner = text[1:-1].strip()
        if not inner or inner.startswith("not "):
            return None
        tokens = inner.split()
        if not tokens:
            return None
        predicate, args = tokens[0], tokens[1:]
        if predicate == "=":
            return None
        functor = ASLMethodLowering._sanitize_name(predicate)
        if not args:
            return functor
        rendered_args = [ASLMethodLowering._runtime_atom_term(arg) for arg in args]
        return f"{functor}({','.join(rendered_args)})"
