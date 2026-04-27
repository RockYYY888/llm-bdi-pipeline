"""
Compatibility helpers for legacy runtime entrypoints.
"""

from plan_library import PlanLibraryArtifactBundle, load_plan_library_artifact_bundle


def bundle_to_legacy_domain_library_artifact(
	library_artifact,
):
	"""Convert a plan-library bundle into the legacy method-library artifact."""

	bundle = load_plan_library_artifact_bundle(library_artifact)
	if not isinstance(bundle, PlanLibraryArtifactBundle):
		raise TypeError("Expected a plan-library artifact bundle.")
	return bundle.as_domain_library_artifact()


__all__ = [
	"bundle_to_legacy_domain_library_artifact",
]
