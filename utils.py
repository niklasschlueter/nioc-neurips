from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from typing import TYPE_CHECKING, Type

import numpy as np
import toml
from munch import munchify
from scipy.spatial.transform import Rotation as R

from mpcc.control.controller import BaseController

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from munch import Munch
    from numpy.typing import NDArray

def additional_config_processing(config):
    # Create the dec Vicon name of the drones depeding on the uris in the config file.
    # A little bit messy but otherwise one would have to change both dec and hex in config file.
    config.deploy.drone_names = [
        f"cf{int(str(link_uri)[-2:], 16)}" for link_uri in config.deploy.uris
    ]
    print(f"Using drone names: {config.deploy.drone_names}")
    return config

def load_controller(path: Path) -> Type[BaseController]:
    """Load the controller module from the given path and return the Controller class.

    Args:
        path: Path to the controller module.
    """
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)

    def filter(mod: Any) -> bool:
        """Filter function to identify valid controller classes.

        Args:
            mod: Any attribute of the controller module to be checked.
        """
        subcls = inspect.isclass(mod) and issubclass(mod, BaseController)
        return subcls and mod.__module__ == controller_module.__name__

    controllers = inspect.getmembers(controller_module, filter)
    controllers = [c for _, c in controllers if issubclass(c, BaseController)]
    assert (
        len(controllers) > 0
    ), f"No controller found in {path}. Have you subclassed BaseController?"
    assert len(controllers) == 1, f"Multiple controllers found in {path}. Only one is allowed."
    controller_module.Controller = controllers[0]
    assert issubclass(controller_module.Controller, BaseController)

    try:
        return controller_module.Controller
    except ImportError as e:
        raise e


def load_config(path: Path) -> Munch:
    """Load the race config file.

    Args:
        path: Path to the config file.

    Returns:
        The munchified config dict.
    """
    assert path.exists(), f"Configuration file not found: {path}"
    assert path.suffix == ".toml", f"Configuration file has to be a TOML file: {path}"
    with open(path, "r") as f:
        return munchify(toml.load(f))

