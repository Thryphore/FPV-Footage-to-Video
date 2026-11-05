"""python -m fpv_triage launches the app; add --cli for the command line."""

from __future__ import annotations

import sys


def main() -> int:
    # Desktop app is the default. Use --cli (or pass --source) for the terminal tool.
    if "--cli" in sys.argv or "--source" in sys.argv:
        argv = [a for a in sys.argv[1:] if a != "--cli"]
        from .cli import main as cli_main

        return cli_main(argv)

    from .app import main as app_main

    app_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
