"""Allow ``python -m nenpi`` to run the drain report."""

import sys

from nenpi.drain import main

if __name__ == "__main__":
    sys.exit(main())
