"""Applicability domain.

``ApplicabilityDomainArtifact`` (inference) imports eagerly and needs only numpy and
onnxruntime. ``ApplicabilityDomain`` (fit) imports scikit-learn, so it is resolved lazily
via ``__getattr__`` — importing this package on a base install must not pull scikit-learn
in, or every Ersilia Model Hub template breaks at load time.
"""

from .artifact import ApplicabilityDomainArtifact as ApplicabilityDomainArtifact

__all__ = ["ApplicabilityDomain", "ApplicabilityDomainArtifact"]


def __getattr__(name):
    if name == "ApplicabilityDomain":
        from .domain import ApplicabilityDomain

        return ApplicabilityDomain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
