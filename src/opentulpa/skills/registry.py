"""Skill registry: register and resolve skills by name."""

from opentulpa.skills.base import Skill

_registry: dict[str, type[Skill]] = {}


def register_skill(skill_class: type[Skill]) -> type[Skill]:
    """Register a skill class."""
    if skill_class.name:
        _registry[skill_class.name] = skill_class
    return skill_class


def get_skill(name: str) -> type[Skill] | None:
    return _registry.get(name)


def get_registry() -> dict[str, type[Skill]]:
    return dict(_registry)
