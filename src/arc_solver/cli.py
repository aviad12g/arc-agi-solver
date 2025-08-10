"""Legacy CLI entry stub.

This module is kept for backward compatibility but delegates to the modern CLI
at `arc_solver.cli.main`. Prefer invoking the console script `arc-solver`.
"""

from arc_solver.cli.main import main as main

if __name__ == "__main__":  # pragma: no cover
    # Delegate to the new CLI
    main()