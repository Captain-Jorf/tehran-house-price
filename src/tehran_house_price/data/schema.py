"""
Data contract.

دو سطح validation داریم:
  1. DataFrame schema (pandera): برای کل dataset
  2. Record schema (pydantic): برای یک listing تنها، مثلاً در API

این جدا بودن عمدی است. pandera با DataFrame ها راحت‌تر کار می‌کند و
pydantic برای request validation در FastAPI استاندارد است.
"""

from __future__ import annotations

from datetime import datetime

import pandera as pa
from pandera.typing import Series
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tehran_house_price.data import constants as const

# ===================================================================
#  pandera schema: برای DataFrame validation
# ===================================================================


class HouseListingSchema(pa.DataFrameModel):
    """
    Schema for the processed Tehran house listings DataFrame.

    این schema روی processed data اعمال می‌شود، نه روی raw. raw data
    معمولاً کثیف است و این validation روی آن fail می‌شود (که خوب است).

    Note on nullable numeric columns (year_built, floor, total_floors):
        pandas nullable integers (Int64) و numpy int64 با هم سازگار نیستند
        وقتی مقادیر NA داشته باشیم. به همین دلیل این ستون‌ها را به عنوان
        float تعریف می‌کنیم - float می‌تواند NaN داشته باشد و pandera
        بدون مشکل coerce می‌کند. در pydantic schema این ستون‌ها همچنان
        int | None هستند چون pydantic record-level کار می‌کند.
    """

    listing_id: Series[str] = pa.Field(unique=True, nullable=False)
    source: Series[str] = pa.Field(isin=list(const.VALID_SOURCES), nullable=False)

    district: Series[str] = pa.Field(nullable=False)
    neighborhood: Series[str] = pa.Field(nullable=True)

    area_m2: Series[float] = pa.Field(
        ge=const.MIN_AREA_M2,
        le=const.MAX_AREA_M2,
        nullable=False,
    )
    rooms: Series[int] = pa.Field(
        ge=const.MIN_ROOMS,
        le=const.MAX_ROOMS,
        nullable=False,
    )

    # year_built, floor, total_floors: float نه int
    # دلیل: این ستون‌ها می‌توانند NaN داشته باشند. numpy int64 نمی‌تواند NaN
    # داشته باشد. float64 می‌تواند و pandera بدون مشکل coerce می‌کند.
    year_built: Series[float] = pa.Field(
        ge=float(const.MIN_YEAR_BUILT),
        le=float(const.MAX_YEAR_BUILT),
        nullable=True,
    )
    floor: Series[float] = pa.Field(nullable=True)
    total_floors: Series[float] = pa.Field(nullable=True)

    has_elevator: Series[bool] = pa.Field(nullable=True)
    has_parking: Series[bool] = pa.Field(nullable=True)
    has_storage: Series[bool] = pa.Field(nullable=True)

    total_price: Series[float] = pa.Field(
        ge=const.MIN_PRICE_IRR,
        le=const.MAX_PRICE_IRR,
        nullable=False,
    )
    price_per_m2: Series[float] = pa.Field(
        ge=0,
        nullable=True,
    )

    published_at: Series[pa.DateTime] = pa.Field(nullable=True)
    ingested_at: Series[pa.DateTime] = pa.Field(nullable=False)

    class Config:
        strict = False  # اجازه می‌دهیم ستون اضافه باشد
        coerce = True  # سعی کن type را cast کنی
        ordered = False


# ===================================================================
#  pydantic schema: برای record validation (مثلاً API)
# ===================================================================


class HouseListing(BaseModel):
    """A single house listing. Used for API I/O and unit-level validation."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="ignore",
    )

    listing_id: str = Field(min_length=1)
    source: str

    district: str = Field(min_length=1)
    neighborhood: str | None = None

    area_m2: float = Field(ge=const.MIN_AREA_M2, le=const.MAX_AREA_M2)
    rooms: int = Field(ge=const.MIN_ROOMS, le=const.MAX_ROOMS)

    # pydantic record-level کار می‌کند، NaN مفهوم ندارد -> int | None درست است
    year_built: int | None = Field(default=None, ge=const.MIN_YEAR_BUILT, le=const.MAX_YEAR_BUILT)
    floor: int | None = None
    total_floors: int | None = None

    has_elevator: bool | None = None
    has_parking: bool | None = None
    has_storage: bool | None = None

    total_price: float = Field(ge=const.MIN_PRICE_IRR, le=const.MAX_PRICE_IRR)
    price_per_m2: float | None = Field(default=None, ge=0)

    published_at: datetime | None = None
    ingested_at: datetime

    @field_validator("source")
    @classmethod
    def _source_must_be_valid(cls, v: str) -> str:
        v_norm = v.strip().lower()
        if v_norm not in const.VALID_SOURCES:
            raise ValueError(f"invalid source '{v}'. expected one of {sorted(const.VALID_SOURCES)}")
        return v_norm

    @field_validator("district")
    @classmethod
    def _district_lowercase(cls, v: str) -> str:
        return v.strip()
