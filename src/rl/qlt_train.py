import sys
from pathlib import Path
from typing import List, Optional

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from rl.train_qlearning import main as _train_main


def _has_agent_flag(argv: List[str]) -> bool:
    for arg in argv:
        if arg == "--agent":
            return True
        if arg.startswith("--agent="):
            return True
    return False


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    injected = [] if _has_agent_flag(argv) else ["--agent", "tabular"]
    return _train_main(injected + argv)


if __name__ == "__main__":
    raise SystemExit(main())
