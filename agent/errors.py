"""Typed domain errors that the UI can turn into honest user-facing states."""


class VerificationRequestError(ValueError):
    """Base class for a request that must not be executed as supplied."""


class NoDataError(VerificationRequestError):
    """The request is meaningful, but no matching data exists."""


class AmbiguousCaseError(VerificationRequestError):
    """More than one forecast initialization is available and none was selected."""


class DataContractError(VerificationRequestError):
    """Input metadata violates the one-product/one-initialization contract."""
