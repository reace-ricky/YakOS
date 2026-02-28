"""YakOS Core - lineup validation."""
from typing import Dict, Any, List, Tuple

from .config import DK_LINEUP_SIZE, DK_POS_SLOTS, SALARY_CAP

POSITION_TEMPLATES = {
    ("NBA", "main"): {"slots": DK_LINEUP_SIZE, "positions": DK_POS_SLOTS},
    ("PGA", "main"): {"slots": 6, "positions": None},
}


def validate_lineup(
    lineup: Dict[str, Any],
    sport: str,
    contest_type: str,
    salary_cap: int = SALARY_CAP,
) -> Tuple[bool, List[str]]:
    """Validate a single lineup dict. Returns (is_valid, errors)."""
    errors: List[str] = []

    key = (sport, contest_type)
    if key not in POSITION_TEMPLATES:
        errors.append(f"No position template for sport={sport}, contest_type={contest_type}")
        return False, errors

    template = POSITION_TEMPLATES[key]
    required_slots = template["slots"]
    players = lineup.get("players") or []

    if len(players) != required_slots:
        errors.append(f"Expected {required_slots} players, got {len(players)}")

    total_salary = 0
    for idx, p in enumerate(players):
        sal = p.get("salary")
        if sal is None:
            errors.append(f"Player {idx} missing salary")
        elif sal < 0:
            errors.append(f"Player {idx} has negative salary {sal}")
        else:
            total_salary += sal

    if total_salary > salary_cap:
        errors.append(f"Salary cap exceeded: {total_salary} > {salary_cap}")

    is_valid = len(errors) == 0
    return is_valid, errors
