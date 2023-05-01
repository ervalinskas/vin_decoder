from typing import List

from fastapi import Query
from pydantic import BaseModel, validator


class Vin(BaseModel):
    vin: str = Query(None, min_length=1)


class Payload(BaseModel):
    vins: List[Vin]

    @validator("vins")
    def list_must_not_be_empty(cls, value):
        if not len(value):
            raise ValueError("List of vins to classify cannot be empty.")
        return value

    class Config:
        schema_extra = {
            "example": {
                "vins": [
                    {"vin": "WAUZZZ8T39A0XXXXX"},
                    {"vin": "WAUZZZ4G0BN0XXXXX"},
                ]
            }
        }
