"""Automatic English → Polish video dubbing."""


def main() -> None:
    import sys

    from .cli import main as cli_main

    sys.exit(cli_main())
