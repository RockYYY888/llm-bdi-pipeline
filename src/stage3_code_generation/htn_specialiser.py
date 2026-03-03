"""
Preferred specialisation for HTN decomposition traces.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Set

from stage3_code_generation.htn_schema import (
    DecompositionNode,
    DecompositionTrace,
    HTNLiteral,
    SpecialisationResult,
)


class HTNSpecialiser:
    """Reduce a decomposition trace to a non-redundant, still-abstract cut."""

    def specialise(
        self,
        trace: DecompositionTrace,
        target_literal: HTNLiteral,
    ) -> SpecialisationResult:
        parent_map = self._build_parent_map(trace.root)
        all_nodes = list(trace.iter_nodes())
        leaves = trace.leaf_nodes()
        needed: Set[str] = {target_literal.to_signature()}
        relevant_leaf_ids: List[str] = []

        for leaf in reversed(leaves):
            if leaf.kind == "primitive":
                produced = {literal.to_signature() for literal in leaf.effects}
                if produced & needed:
                    relevant_leaf_ids.append(leaf.node_id)
                    needed -= produced
                    needed.update(
                        literal.to_signature()
                        for literal in leaf.preconditions
                        if literal.is_positive
                    )
            else:
                available = {
                    literal.to_signature()
                    for literal in (leaf.context or ((leaf.literal,) if leaf.literal else ()))
                    if literal is not None
                }
                if available & needed:
                    relevant_leaf_ids.append(leaf.node_id)
                    needed -= available

        if not relevant_leaf_ids:
            relevant_leaf_ids = [trace.root.node_id]

        relevant_ids = set(relevant_leaf_ids)
        for leaf_id in list(relevant_leaf_ids):
            current = leaf_id
            while current in parent_map:
                current = parent_map[current]
                relevant_ids.add(current)

        @lru_cache(maxsize=None)
        def subtree_fully_relevant(node_id: str) -> bool:
            node = self._find_node(trace.root, node_id)
            if node is None or node_id not in relevant_ids:
                return False
            if not node.children:
                return True
            return all(subtree_fully_relevant(child.node_id) for child in node.children)

        retained_cut = self._collect_cut(trace.root, relevant_ids, subtree_fully_relevant)
        dropped = [node.node_id for node in all_nodes if node.node_id not in relevant_ids]

        return SpecialisationResult(
            relevant_node_ids=sorted(relevant_ids),
            retained_cut_node_ids=retained_cut,
            relevant_leaf_ids=list(reversed(relevant_leaf_ids)),
            dropped_node_ids=dropped,
            metrics={
                "total_nodes": len(all_nodes),
                "relevant_nodes": len(relevant_ids),
                "leaf_nodes": len(leaves),
                "relevant_leaves": len(relevant_leaf_ids),
                "cut_size": len(retained_cut),
            },
        )

    def _build_parent_map(self, root: DecompositionNode) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        stack = [root]
        while stack:
            node = stack.pop()
            for child in node.children:
                mapping[child.node_id] = node.node_id
                stack.append(child)
        return mapping

    def _collect_cut(
        self,
        node: DecompositionNode,
        relevant_ids: Set[str],
        subtree_fully_relevant,
    ) -> List[str]:
        if node.node_id not in relevant_ids:
            return []
        if not node.children:
            return [node.node_id]
        if all(subtree_fully_relevant(child.node_id) for child in node.children):
            return [node.node_id]

        cut: List[str] = []
        for child in node.children:
            cut.extend(self._collect_cut(child, relevant_ids, subtree_fully_relevant))
        return cut or [node.node_id]

    def _find_node(self, root: DecompositionNode, node_id: str) -> DecompositionNode | None:
        stack = [root]
        while stack:
            node = stack.pop()
            if node.node_id == node_id:
                return node
            stack.extend(reversed(node.children))
        return None
