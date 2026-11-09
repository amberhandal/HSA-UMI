"""
Pin the environment nuc/ reads at import, so a shell with HSA_ROBOT or HSA_TOOL exported cannot change a result.

hsa_calib builds its frames and payload from both when imported; the launch files and robots.get() read them too.
Every test starts from neither being set (the FER with its wrist), and ``hsa_env`` sets them the way
nuc/station_env.sh would.
"""

import importlib
import sys

import pytest


def _reload_hsa_calib():
    if 'hsa_calib' in sys.modules:
        return importlib.reload(sys.modules['hsa_calib'])
    return importlib.import_module('hsa_calib')


@pytest.fixture(autouse=True)
def _no_hsa_env(monkeypatch):
    monkeypatch.delenv('HSA_ROBOT', raising=False)
    monkeypatch.delenv('HSA_TOOL', raising=False)
    _reload_hsa_calib()
    yield
    monkeypatch.undo()
    monkeypatch.delenv('HSA_ROBOT', raising=False)
    monkeypatch.delenv('HSA_TOOL', raising=False)
    _reload_hsa_calib()


@pytest.fixture
def hsa_env(monkeypatch):
    """Return a function exporting HSA_ROBOT / HSA_TOOL (empty = unset) and re-importing hsa_calib under them."""

    def export(robot: str = '', tool: str = ''):
        for name, value in (('HSA_ROBOT', robot), ('HSA_TOOL', tool)):
            if value:
                monkeypatch.setenv(name, value)
            else:
                monkeypatch.delenv(name, raising=False)
        return _reload_hsa_calib()

    return export
