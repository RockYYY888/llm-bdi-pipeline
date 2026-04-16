"""Generic task naming helpers shared by the current pipeline."""

from __future__ import annotations


def sanitize_identifier(value: str) -> str:
	text = str(value or "").strip().replace("-", "_")
	buffer: list[str] = []
	for character in text:
		if character.isalnum() or character == "_":
			buffer.append(character.lower())
		else:
			buffer.append("_")
	sanitized = "".join(buffer).strip("_")
	while "__" in sanitized:
		sanitized = sanitized.replace("__", "_")
	if not sanitized:
		sanitized = "item"
	if not sanitized[0].isalpha():
		sanitized = f"t_{sanitized}"
	return sanitized


def query_root_alias_task_name(index: int, source_task_name: str) -> str:
	return f"query_root_{int(index)}_{sanitize_identifier(source_task_name)}"
