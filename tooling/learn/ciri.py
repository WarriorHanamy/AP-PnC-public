"""CIRI is not implemented.

The user's CIRI representation is not specified in this checkout. Do not
substitute a range image, an occupancy cloud, or a convex corridor.
"""

from __future__ import annotations


class CiriUnavailable(NotImplementedError):
    """Raised when a caller asks for a CIRI observation that has no schema."""


def require_ciri() -> None:
    raise CiriUnavailable(
        "CIRI input is unfinished. This checkout does not define that representation, "
        "so RL is not wired to a guessed CIRI tensor."
    )
