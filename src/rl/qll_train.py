import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from rl.train_qlearning import main as _train_main

_APPROX_DEFAULTS: Dict[str, str] = {
    "--num_episodes": "5000",
    "--decay_rate": "0.0005",
    "--learning_rate": "0.001",
    "--gamma": "0.95",
    "--max_steps": "500",
    "--batch_size": "64",
    "--hidden_dim": "64",
}

_AGENT_TYPE_MAP = {
    "linear": "linear",
    "replay": "neural",
    "neural": "neural",
}

_AGENT_ALIASES = {
    "replay": "neural",
}


def _strip_agent_type(argv: List[str]) -> Tuple[List[str], Optional[str]]:
    clean: List[str] = []
    agent_type: Optional[str] = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--agent_type":
            if i + 1 >= len(argv):
                raise ValueError("--agent_type requires a value")
            agent_type = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--agent_type="):
            agent_type = arg.split("=", 1)[1]
            i += 1
            continue
        clean.append(arg)
        i += 1
    return clean, agent_type


def _has_flag(argv: List[str], flag: str) -> bool:
    for arg in argv:
        if arg == flag or arg.startswith(f"{flag}="):
            return True
    return False


def _find_agent(argv: List[str]) -> Optional[str]:
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--agent":
            if i + 1 < len(argv):
                return argv[i + 1]
            return None
        if arg.startswith("--agent="):
            return arg.split("=", 1)[1]
        i += 1
    return None


def _inject_defaults(argv: List[str], agent: str) -> List[str]:
    if agent not in ("linear", "replay", "neural"):
        return argv

    defaults: List[str] = []
    for flag, value in _APPROX_DEFAULTS.items():
        if not _has_flag(argv, flag):
            defaults.extend([flag, value])
    return defaults + argv


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    cleaned_args, agent_type = _strip_agent_type(argv)
    agent_flag_present = _has_flag(cleaned_args, "--agent")

    mapped_agent: Optional[str] = None
    if agent_type is not None:
        mapped_agent = _AGENT_TYPE_MAP.get(agent_type)
        if mapped_agent is None:
            raise ValueError(f"Unknown agent type: {agent_type}")

    if not agent_flag_present:
        agent_choice = mapped_agent or "linear"
        cleaned_args = ["--agent", agent_choice] + cleaned_args

    agent = _find_agent(cleaned_args) or "linear"
    canonical_agent = _AGENT_ALIASES.get(agent, agent)
    if canonical_agent != agent:
        # replace first occurrence of --agent to keep canonical form for downstream defaults
        updated: List[str] = []
        i = 0
        replaced = False
        while i < len(cleaned_args):
            arg = cleaned_args[i]
            if not replaced and arg == "--agent":
                updated.extend(["--agent", canonical_agent])
                replaced = True
                i += 2
                continue
            if not replaced and arg.startswith("--agent="):
                updated.append(f"--agent={canonical_agent}")
                replaced = True
                i += 1
                continue
            updated.append(arg)
            i += 1
        cleaned_args = updated
        agent = canonical_agent

    final_args = _inject_defaults(cleaned_args, agent)
    return _train_main(final_args)


if __name__ == "__main__":
    raise SystemExit(main())
