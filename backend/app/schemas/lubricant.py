from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class LubricantTargetProperties(BaseModel):
    viscosity_kv100_min_cst: float = Field(gt=0)
    viscosity_kv100_max_cst: float = Field(gt=0)
    viscosity_index_min: float = Field(ge=0)
    pour_point_max_c: float

    @model_validator(mode="after")
    def validate_viscosity_range(self) -> LubricantTargetProperties:
        if self.viscosity_kv100_min_cst > self.viscosity_kv100_max_cst:
            raise ValueError("Minimum viscosity must not exceed maximum viscosity.")
        return self


class LubricantOptimizationRequest(BaseModel):
    target_properties: LubricantTargetProperties
    batch_size_kg: float = Field(default=1000, gt=0)


class LubricantComponentAllocation(BaseModel):
    material_id: str
    name: str
    material_type: str
    percentage: float
    amount_kg: float


class LubricantCatalogComponent(BaseModel):
    material_id: str
    name: str
    material_type: str
    cost_per_kg: float
    available_kg: float
    max_percentage: float
    viscosity_kv100_cst: float
    viscosity_index: float
    pour_point_c: float


class LubricantCatalogResponse(BaseModel):
    data_source: Literal["synthetic_demo"]
    operational_use_allowed: Literal[False]
    components: list[LubricantCatalogComponent]
    warnings: list[str]


class LubricantPredictedProperties(BaseModel):
    viscosity_kv100_cst: float
    viscosity_index: float
    pour_point_c: float


class LubricantOptimizationResponse(BaseModel):
    formulation: list[LubricantComponentAllocation]
    batch_size_kg: float
    estimated_cost: float
    cost_unit: str
    predicted_properties: LubricantPredictedProperties
    data_source: Literal["synthetic_demo"]
    operational_use_allowed: Literal[False]
    warnings: list[str]
