from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Any, Iterable, Mapping, Sequence, cast


AXES = ("x", "y", "z")
LOCAL_AXES = ("w", "l", "h")
_AXIS_NAMES = {"x", "y", "z"}


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return result


def _read_dimensions(value: Mapping[str, Any] | Sequence[Any], name: str) -> dict[str, int]:
    if isinstance(value, Mapping):
        width = value.get("w", value.get("width"))
        length = value.get("l", value.get("d", value.get("length", value.get("depth"))))
        height = value.get("h", value.get("height"))
        if width is None or length is None or height is None:
            raise ValueError(f"{name} must contain w/l/h dimensions")
        raw = (width, length, height)
    else:
        if len(value) != 3:
            raise ValueError(f"{name} must have exactly three dimensions")
        raw = tuple(value)
    result = {axis: _as_int(item, f"{name}.{axis}") for axis, item in zip(LOCAL_AXES, raw)}
    if any(item <= 0 for item in result.values()):
        raise ValueError(f"{name} dimensions must be positive")
    return result


def _normalize_position(position: Any) -> tuple[int, int, int] | None:
    if position is None or position in ("outside", "out", "none"):
        return None
    if isinstance(position, Mapping):
        raw = (position.get("x"), position.get("y"), position.get("z"))
    else:
        raw = tuple(position)
    if len(raw) != 3 or any(item is None for item in raw):
        raise ValueError("position must contain x/y/z coordinates")
    result = tuple(_as_int(item, f"position[{index}]") for index, item in enumerate(raw))
    return cast(tuple[int, int, int], result)


def _normalize_orientation(value: Any) -> tuple[str, str, str]:
    if value is None:
        return AXES
    if isinstance(value, Mapping):
        if all(key in value for key in LOCAL_AXES):
            raw = tuple(value[key] for key in LOCAL_AXES)
        elif all(key in value for key in AXES):
            inverse = {value[key]: key for key in AXES}
            raw = tuple(inverse[axis] for axis in LOCAL_AXES)
        else:
            raise ValueError("orientation must map w/l/h or x/y/z")
    else:
        raw = tuple(value)
    normalized = tuple(str(axis).lower() for axis in raw)
    if len(normalized) != 3 or set(normalized) != _AXIS_NAMES:
        raise ValueError("orientation must be a permutation of x, y, z")
    return cast(tuple[str, str, str], normalized)


@dataclass
class Package:
    id: str
    dimensions: Mapping[str, Any] | Sequence[Any]
    orientation: Any = None
    value: int | float = 0
    weight: int | float = 0
    is_fragile: bool = False
    eta: int | float = 0
    position: Sequence[Any] | Mapping[str, Any] | str | None = None
    _dimensions: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("package id cannot be empty")
        self._dimensions = _read_dimensions(self.dimensions, f"package {self.id}")
        self.orientation = _normalize_orientation(self.orientation)
        self.position = _normalize_position(self.position)
        if self.weight < 0:
            raise ValueError("package weight cannot be negative")

    @property
    def dims(self) -> dict[str, int]:
        return dict(self._dimensions)

    @property
    def effective_dimensions(self) -> tuple[int, int, int]:
        by_local = dict(zip(LOCAL_AXES, (self._dimensions[axis] for axis in LOCAL_AXES)))
        global_dims = {axis: by_local[LOCAL_AXES[index]] for index, axis in enumerate(self.orientation)}
        return tuple(global_dims[axis] for axis in AXES)

    @property
    def is_inside(self) -> bool:
        return self.position is not None

    def copy(self) -> "Package":
        return Package(
            id=self.id,
            dimensions=self._dimensions,
            orientation=self.orientation,
            value=self.value,
            weight=self.weight,
            is_fragile=self.is_fragile,
            eta=self.eta,
            position=self.position,
        )

    def with_position(self, position: Sequence[int] | None) -> "Package":
        result = self.copy()
        result.position = _normalize_position(position)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dimensions": self.dims,
            "orientation": list(self.orientation),
            "value": self.value,
            "weight": self.weight,
            "isFragile": self.is_fragile,
            "ETA": self.eta,
            "position": list(self.position) if self.position is not None else None,
        }


@dataclass(frozen=True)
class Truck:
    dimensions: Mapping[str, Any] | Sequence[Any]
    max_capacity: int | float

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", _read_dimensions(self.dimensions, "truck"))
        if self.max_capacity < 0:
            raise ValueError("truck max_capacity cannot be negative")

    @property
    def effective_dimensions(self) -> tuple[int, int, int]:
        return tuple(self.dimensions[axis] for axis in LOCAL_AXES)

    @property
    def width(self) -> int:
        return self.effective_dimensions[0]

    @property
    def length(self) -> int:
        return self.effective_dimensions[1]

    @property
    def height(self) -> int:
        return self.effective_dimensions[2]

    @property
    def capacity(self) -> int | float:
        return self.max_capacity

    def to_dict(self) -> dict[str, Any]:
        return {"dimensions": dict(self.dimensions), "maxCapacity": self.max_capacity}


@dataclass
class State:
    truck: Truck
    packages: dict[str, Package] | Iterable[Package]

    def __post_init__(self) -> None:
        if not isinstance(self.packages, dict):
            package_list = list(self.packages)
            package_ids = [package.id for package in package_list]
            if len(package_ids) != len(set(package_ids)):
                raise ValueError("package ids must be unique")
            self.packages = {package.id: package for package in package_list}
        else:
            self.packages = dict(self.packages)
        if len(self.packages) != len(set(self.packages)):
            raise ValueError("package ids must be unique")
        for package_id, package in self.packages.items():
            if package_id != package.id:
                raise ValueError("state package keys must match package ids")

    def copy(self) -> "State":
        return State(self.truck, {package_id: package.copy() for package_id, package in self.packages.items()})

    def __iter__(self):
        return iter(self.packages.values())

    def __len__(self) -> int:
        return len(self.packages)

    def __getitem__(self, package_id: str) -> Package:
        return self.packages[package_id]

    @property
    def inside_packages(self) -> list[Package]:
        return [package for package in self if package.is_inside]

    @property
    def outside_packages(self) -> list[Package]:
        return [package for package in self if not package.is_inside]

    @property
    def total_value(self) -> float:
        return sum(package.value for package in self.inside_packages)

    @property
    def total_weight(self) -> float:
        return sum(package.weight for package in self.inside_packages)

    def replace_package(self, package: Package) -> "State":
        if package.id not in self.packages:
            raise KeyError(package.id)
        result = self.copy()
        result.packages[package.id] = package.copy()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"truck": self.truck.to_dict(), "packages": [package.to_dict() for package in self]}


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    violations: tuple[str, ...] = ()


def _horizontal_overlap(a: Package, b: Package) -> bool:
    if a.position is None or b.position is None:
        return False
    aw, al, _ = a.effective_dimensions
    bw, bl, _ = b.effective_dimensions
    ax, ay, _ = a.position
    bx, by, _ = b.position
    return min(ax + aw, bx + bw) > max(ax, bx) and min(ay + al, by + bl) > max(ay, by)


def _boxes_overlap(a: Package, b: Package) -> bool:
    if a.position is None or b.position is None:
        return False
    ad = a.effective_dimensions
    bd = b.effective_dimensions
    return all(
        min(a.position[index] + ad[index], b.position[index] + bd[index])
        > max(a.position[index], b.position[index])
        for index in range(3)
    )


class ConstraintChecker:
    def __init__(self, truck: Truck):
        self.truck = truck

    def check(self, state: State) -> ValidationResult:
        if state.truck != self.truck:
            return ValidationResult(False, ("state truck does not match checker truck",))
        violations: list[str] = []
        inside = state.inside_packages
        for package in inside:
            if package.position is None:
                continue
            x, y, z = package.position
            width, length, height = package.effective_dimensions
            if x < 0 or y < 0 or z < 0 or x + width > self.truck.width or y + length > self.truck.length or z + height > self.truck.height:
                violations.append(f"package {package.id} crosses the truck wall")

        for index, first in enumerate(inside):
            for second in inside[index + 1 :]:
                if _boxes_overlap(first, second):
                    violations.append(f"packages {first.id} and {second.id} overlap")

        for package in inside:
            assert package.position is not None
            if package.position[2] == 0:
                continue
            supported = any(
                other.id != package.id
                and other.position is not None
                and other.position[2] + other.effective_dimensions[2] == package.position[2]
                and _horizontal_overlap(package, other)
                for other in inside
            )
            if not supported:
                violations.append(f"package {package.id} is floating")

        for lower in inside:
            if not lower.is_fragile or lower.position is None:
                continue
            lower_top = lower.position[2] + lower.effective_dimensions[2]
            for upper in inside:
                if upper.id == lower.id or upper.position is None:
                    continue
                if upper.position[2] == lower_top and _horizontal_overlap(lower, upper):
                    violations.append(f"fragile package {lower.id} supports package {upper.id}")

        if state.total_weight > self.truck.max_capacity:
            violations.append("truck maximum capacity is exceeded")
        return ValidationResult(not violations, tuple(violations))

    def is_valid(self, state: State) -> bool:
        return self.check(state).valid


def is_valid(state: State, checker: ConstraintChecker | None = None) -> bool:
    return (checker or ConstraintChecker(state.truck)).is_valid(state)


def objective(
    state: State,
    checker: ConstraintChecker | None = None,
    invalid_penalty: float | None = None,
) -> float:
    if not is_valid(state, checker):
        return -inf if invalid_penalty is None else invalid_penalty
    return float(state.total_value)


class ObjectiveFunction:
    def __init__(self, checker: ConstraintChecker | None = None, invalid_penalty: float | None = None):
        self.checker = checker
        self.invalid_penalty = invalid_penalty

    def __call__(self, state: State) -> float:
        return objective(state, self.checker, self.invalid_penalty)
