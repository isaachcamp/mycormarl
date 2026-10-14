"""Protect independent core installation after extraction."""
import ast
from pathlib import Path
import importlib.util
import mycormarl


def test_core_has_no_downstream_runtime_imports():
    for path in Path(mycormarl.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(name.split(".")[0] in {"redbyrbd", "mycormarl_animations", "scripts", "manim"} for name in names), path


def test_named_research_modules_are_absent_from_installed_core():
    for name in ["study", "trade_only_study", "static_controls", "resource_pressure_screen", "resource_pressure_analysis", "plant_growth_qualification", "soil.absorber_diagnostic", "soil.depletion_gradient_diagnostic", "soil.hyphal_density_sweep"]:
        assert importlib.util.find_spec("mycormarl." + name) is None, name
