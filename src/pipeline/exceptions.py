class PipelineError(Exception):
    """Base pipeline exception."""


class StageExecutionError(PipelineError):
    """Raised when a stage fails."""

