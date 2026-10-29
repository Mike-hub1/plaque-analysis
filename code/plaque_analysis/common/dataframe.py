"""Column assignment support for copied tables owned by native kernels.

Call only for independently copied output tables. An ordinary Python frame
lets pandas identify the owned table during its chained-assignment checks.
"""


def assign_columns(frame, key, value):
    """Preserve pandas column assignment, including label alignment and dtype."""
    frame[key] = value


def and_column(frame, key, value):
    """Preserve the existing in-place Boolean-column update."""
    frame[key] &= value
