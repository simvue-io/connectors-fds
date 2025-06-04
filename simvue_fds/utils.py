"""Utils.

Helper functions for the FDS connector.
"""

import os
import sys


class HiddenPrints:
    """Stop prints being displayed on the console, for use as a context manager."""

    def __enter__(self):
        """Redirect stdout to devnull so it doesn't show up."""
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Resume normal printing behaviour."""
        sys.stdout.close()
        sys.stdout = self._original_stdout
