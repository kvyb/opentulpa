"""Dynamic loader for agent-created FastAPI routers in tulpa_stuff."""

from __future__ import annotations

import ast
import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)


class TulpaRouterLoader:
    """Load and hot-reload APIRouter objects from the tulpa_stuff package."""

    def __init__(
        self,
        project_root: Path,
        mount_router: APIRouter,
    ) -> None:
        self.project_root = project_root.resolve()
        self.mount_router = mount_router
        self.package_name = "tulpa_stuff"
        self.package_dir = self.project_root / self.package_name

    def _ensure_importable(self) -> None:
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        existing = sys.modules.get(self.package_name)
        if existing is not None:
            module_file = str(getattr(existing, "__file__", "") or "").strip()
            module_path = Path(module_file).resolve() if module_file else None
            if module_path is None or not (
                module_path == self.package_dir / "__init__.py" or self.project_root in module_path.parents
            ):
                for name in list(sys.modules):
                    if name == self.package_name or name.startswith(f"{self.package_name}."):
                        sys.modules.pop(name, None)
        self.package_dir.mkdir(parents=True, exist_ok=True)
        init_file = self.package_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text(
                '"""Agent-created integrations and skills."""\n',
                encoding="utf-8",
            )

    def _module_names(self) -> list[str]:
        modules: list[str] = []
        for file in sorted(self.package_dir.glob("*.py")):
            name = file.stem
            if name == "__init__" or name.startswith("_"):
                continue
            modules.append(name)
        return modules

    def _appears_to_export_router(self, module_name: str) -> bool:
        path = self.package_dir / f"{module_name}.py"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except Exception:
            return False
        return any(_is_router_assignment(stmt) for stmt in _module_level_statements(tree))

    def _top_level_import_warning(self, module_name: str) -> str | None:
        path = self.package_dir / f"{module_name}.py"
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except Exception:
            return "module could not be parsed"
        for stmt in tree.body:
            if _is_safe_module_level_statement(stmt):
                continue
            lineno = getattr(stmt, "lineno", "?")
            return f"unsafe top-level statement at line {lineno}: {type(stmt).__name__}"
        return None

    def _import_module(self, module_name: str) -> ModuleType:
        full_name = f"{self.package_name}.{module_name}"
        if full_name in sys.modules:
            return importlib.reload(sys.modules[full_name])
        return importlib.import_module(full_name)

    def reload(self) -> dict[str, Any]:
        """Reload all tulpa_stuff module routers onto the mount router."""
        self._ensure_importable()
        self.mount_router.routes.clear()

        loaded: list[str] = []
        errors: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []

        for module_name in self._module_names():
            if not self._appears_to_export_router(module_name):
                continue
            safety_warning = self._top_level_import_warning(module_name)
            if safety_warning is not None:
                logger.warning(
                    "Skipping tulpa module %s due to import safety guard: %s",
                    module_name,
                    safety_warning,
                )
                warnings.append(
                    {
                        "module": module_name,
                        "warning": f"import safety guard: {safety_warning}",
                    }
                )
                continue
            try:
                module = self._import_module(module_name)
                router_obj = getattr(module, "router", None)
                if not isinstance(router_obj, APIRouter):
                    raise TypeError("missing APIRouter 'router' export")
                self.mount_router.include_router(
                    router_obj,
                    prefix=f"/{module_name}",
                    tags=["tulpa"],
                )
                loaded.append(module_name)
            except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
                missing = str(getattr(exc, "name", "")).strip() or str(exc)
                logger.warning(
                    "Skipping tulpa module %s due to missing dependency: %s",
                    module_name,
                    missing,
                )
                warnings.append(
                    {
                        "module": module_name,
                        "warning": f"missing dependency: {missing}",
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.exception("Failed to load tulpa module %s: %s", module_name, exc)
                errors.append({"module": module_name, "error": str(exc)})

        return {
            "ok": True,
            "loaded": loaded,
            "warnings": warnings,
            "errors": errors,
            "mount_prefix": "/tulpa/<module_name>",
        }


def _module_level_statements(tree: ast.Module) -> list[ast.stmt]:
    out: list[ast.stmt] = []
    for stmt in tree.body:
        out.append(stmt)
        if isinstance(stmt, ast.Try):
            out.extend(stmt.body)
            for handler in stmt.handlers:
                out.extend(handler.body)
            out.extend(stmt.orelse)
    return out


def _is_safe_module_level_statement(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Import | ast.ImportFrom | ast.FunctionDef | ast.AsyncFunctionDef):
        return True
    if isinstance(stmt, ast.Expr):
        return isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)
    if isinstance(stmt, ast.Assign):
        return _is_router_assignment(stmt) or _is_safe_constant_assignment(stmt)
    if isinstance(stmt, ast.AnnAssign):
        return _is_safe_literal(stmt.value)
    return isinstance(stmt, ast.If) and _is_main_guard(stmt.test)


def _is_safe_constant_assignment(stmt: ast.Assign) -> bool:
    return _is_safe_literal(stmt.value)


def _is_safe_literal(value: ast.AST | None) -> bool:
    if value is None:
        return True
    if isinstance(value, ast.Constant):
        return True
    if isinstance(value, ast.List | ast.Tuple | ast.Set):
        return all(_is_safe_literal(item) for item in value.elts)
    if isinstance(value, ast.Dict):
        return all(
            _is_safe_literal(key) and _is_safe_literal(item)
            for key, item in zip(value.keys, value.values, strict=False)
        )
    return False


def _is_main_guard(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    comparator = test.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def _is_router_assignment(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Assign):
        if not any(_is_router_target(target) for target in stmt.targets):
            return False
        return _is_apirouter_call(stmt.value)
    if isinstance(stmt, ast.AnnAssign):
        if not _is_router_target(stmt.target):
            return False
        return _is_apirouter_call(stmt.value)
    return False


def _is_router_target(target: ast.expr) -> bool:
    return isinstance(target, ast.Name) and target.id == "router"


def _is_apirouter_call(value: ast.expr | None) -> bool:
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    if isinstance(func, ast.Name):
        return func.id == "APIRouter"
    return isinstance(func, ast.Attribute) and func.attr == "APIRouter"
