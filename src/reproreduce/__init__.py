"""Public ReproReduce API."""

__version__ = "0.1.0"

from .api import reduce
from .core.result import ReductionResult
from .oracle.base import FailureOracle, OracleResult
from .oracle.exception import ExceptionOracle, FailureFingerprint
from .oracle.gradient import GradientDifferenceOracle
from .oracle.numerical import CompileDifferenceOracle, ExecutionOutcome

__all__ = [
    "CompileDifferenceOracle",
    "ExceptionOracle",
    "ExecutionOutcome",
    "FailureFingerprint",
    "FailureOracle",
    "GradientDifferenceOracle",
    "OracleResult",
    "ReductionResult",
    "__version__",
    "reduce",
]
