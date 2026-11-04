from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog

from app.schemas.lubricant import LubricantTargetProperties


@dataclass(frozen=True)
class SyntheticComponent:
    material_id: str
    name: str
    material_type: str
    cost_per_kg: float
    available_kg: float
    max_percentage: float
    viscosity_kv100_cst: float
    viscosity_index: float
    pour_point_c: float


SYNTHETIC_COMPONENTS = (
    SyntheticComponent(
        "SYN-BASE-01", "Demo light base oil", "base_oil", 1.10, 5000, 1.0, 4.5, 95, -18
    ),
    SyntheticComponent(
        "SYN-BASE-02", "Demo medium base oil", "base_oil", 1.40, 3500, 1.0, 6.0, 105, -24
    ),
    SyntheticComponent(
        "SYN-BASE-03", "Demo high-VI base oil", "base_oil", 2.00, 2000, 1.0, 9.0, 125, -45
    ),
    SyntheticComponent(
        "SYN-ADD-01", "Demo viscosity-index improver", "additive", 8.00, 500, 0.04, 0, 300, 0
    ),
    SyntheticComponent(
        "SYN-ADD-02", "Demo pour-point depressant", "additive", 6.00, 300, 0.02, 0, 0, -250
    ),
)

DEMO_WARNINGS = [
    "All component data and additive effects are synthetic demonstration values.",
    "The optimizer uses a simplified linear blending approximation, not validated lubricant blending rules.",
    "Do not use this result for production, purchasing, or standards-compliance decisions.",
]


def synthetic_catalog() -> list[dict[str, Any]]:
    return [
        {
            "material_id": component.material_id,
            "name": component.name,
            "material_type": component.material_type,
            "cost_per_kg": component.cost_per_kg,
            "available_kg": component.available_kg,
            "max_percentage": component.max_percentage * 100,
            "viscosity_kv100_cst": component.viscosity_kv100_cst,
            "viscosity_index": component.viscosity_index,
            "pour_point_c": component.pour_point_c,
        }
        for component in SYNTHETIC_COMPONENTS
    ]


def optimize_lubricant(
    targets: LubricantTargetProperties, batch_size_kg: float
) -> dict[str, Any]:
    components = SYNTHETIC_COMPONENTS
    costs = np.array([component.cost_per_kg for component in components], dtype=float)
    kv100 = np.array(
        [component.viscosity_kv100_cst for component in components], dtype=float
    )
    viscosity_index = np.array(
        [component.viscosity_index for component in components], dtype=float
    )
    pour_point = np.array(
        [component.pour_point_c for component in components], dtype=float
    )
    availability_bounds = [
        (
            0,
            min(component.max_percentage, component.available_kg / batch_size_kg),
        )
        for component in components
    ]

    result = linprog(
        c=costs,
        A_ub=np.vstack(
            [
                kv100,
                -kv100,
                -viscosity_index,
                pour_point,
            ]
        ),
        b_ub=np.array(
            [
                targets.viscosity_kv100_max_cst,
                -targets.viscosity_kv100_min_cst,
                -targets.viscosity_index_min,
                targets.pour_point_max_c,
            ],
            dtype=float,
        ),
        A_eq=np.ones((1, len(components))),
        b_eq=np.array([1.0]),
        bounds=availability_bounds,
        method="highs",
    )
    if not result.success:
        if result.status == 2:
            raise ValueError("No feasible formulation meets the requested demo targets.")
        raise RuntimeError(f"Demo optimizer failed: {result.message}")

    fractions = np.asarray(result.x, dtype=float)
    formulation = [
        {
            "material_id": component.material_id,
            "name": component.name,
            "material_type": component.material_type,
            "percentage": float(fraction * 100),
            "amount_kg": float(fraction * batch_size_kg),
        }
        for component, fraction in zip(components, fractions, strict=True)
        if fraction > 1e-8
    ]
    return {
        "formulation": formulation,
        "batch_size_kg": float(batch_size_kg),
        "estimated_cost": float(result.fun * batch_size_kg),
        "cost_unit": "synthetic cost units",
        "predicted_properties": {
            "viscosity_kv100_cst": float(fractions @ kv100),
            "viscosity_index": float(fractions @ viscosity_index),
            "pour_point_c": float(fractions @ pour_point),
        },
        "data_source": "synthetic_demo",
        "operational_use_allowed": False,
        "warnings": DEMO_WARNINGS,
    }
