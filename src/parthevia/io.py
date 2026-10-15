from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from .model import Package, State, Truck

def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default

def package_from_dict(data: Mapping[str, Any]) -> Package:
    return Package(
        id=_first(data, "id", "ID"),
        dimensions=_first(data, "dimensions", "Dimensions"),
        orientation=_first(data, "orientation", "Orientation"),
        value=_first(data, "value", "Value", default=0),
        weight=_first(data, "weight", "Weight", default=0),
        is_fragile=_first(data, "isFragile", "is_fragile", "IsFragile", default=False),
        eta=_first(data, "ETA", "eta", default=0),
        position=_first(data, "position", "Position", default=None),
    )

def truck_from_dict(data: Mapping[str, Any]) -> Truck:
    return Truck(
        dimensions=_first(data, "dimensions", "Dimensions"),
        max_capacity=_first(data, "maxCapacity", "max_capacity", "MaxCapacity", "capacity"),
    )

def load_problem_data(data: Mapping[str, Any]) -> tuple[Truck, list[Package]]:
    truck_data = _first(data, "truck", "Truck")
    packages_data = _first(data, "packages", "Packages", default=[])
    if truck_data is None:
        raise ValueError("input must contain a truck")
    return truck_from_dict(truck_data), [package_from_dict(item) for item in packages_data]

def load_problem(path: str | Path) -> tuple[Truck, list[Package]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return load_problem_data(json.load(handle))

def state_from_dict(data: Mapping[str, Any], truck: Truck | None = None) -> State:
    if truck is None:
        truck_data = _first(data, "truck", "Truck")
        if truck_data is None:
            raise ValueError("state input must contain a truck when truck is not supplied")
        truck = truck_from_dict(truck_data)
    packages_data = _first(data, "packages", "Packages", default=[])
    return State(truck, [package_from_dict(item) for item in packages_data])

def load_state(path: str | Path, truck: Truck | None = None) -> State:
    with Path(path).open("r", encoding="utf-8") as handle:
        return state_from_dict(json.load(handle), truck)

def save_state(state: State, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, indent=2)
        handle.write("\n")

def save_json(data: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
