from __future__ import annotations

import sys

from automodelling import main as automodellingMain


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in {"run", "search", "agent", "inspect", "init-program"}:
        sys.argv = ["automodelling.py", *argv]
    else:
        sys.argv = ["automodelling.py", "agent", *argv]
    automodellingMain()


if __name__ == "__main__":
    main()
