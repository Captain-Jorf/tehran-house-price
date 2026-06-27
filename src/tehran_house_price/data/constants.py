"""
Data constants.

اسم ستون‌های canonical اینجاست. هر دو منبع (Kaggle, Divar) باید به این
اسم‌ها نگاشت شوند. این کار باعث می‌شود بقیه pipeline یک schema واحد ببیند.

دلیل اینکه enum یا dataclass استفاده نکردم: این‌ها فقط constant هستند
و در 99% موارد به صورت string لازم می‌شوند (column access در pandas).
"""

from typing import Final

# ----- canonical column names -----

LISTING_ID: Final[str] = "listing_id"
SOURCE: Final[str] = "source"

# location
DISTRICT: Final[str] = "district"
NEIGHBORHOOD: Final[str] = "neighborhood"

# physical attributes
AREA_M2: Final[str] = "area_m2"
ROOMS: Final[str] = "rooms"
YEAR_BUILT: Final[str] = "year_built"
FLOOR: Final[str] = "floor"
TOTAL_FLOORS: Final[str] = "total_floors"

# amenities (boolean)
HAS_ELEVATOR: Final[str] = "has_elevator"
HAS_PARKING: Final[str] = "has_parking"
HAS_STORAGE: Final[str] = "has_storage"

# price (target)
TOTAL_PRICE: Final[str] = "total_price"  # IRR (toman)
PRICE_PER_M2: Final[str] = "price_per_m2"  # IRR per m2

# metadata
PUBLISHED_AT: Final[str] = "published_at"
INGESTED_AT: Final[str] = "ingested_at"


# ----- column groups -----

NUMERIC_COLS: Final[list[str]] = [
    AREA_M2,
    ROOMS,
    YEAR_BUILT,
    FLOOR,
    TOTAL_FLOORS,
    TOTAL_PRICE,
    PRICE_PER_M2,
]

BOOLEAN_COLS: Final[list[str]] = [
    HAS_ELEVATOR,
    HAS_PARKING,
    HAS_STORAGE,
]

CATEGORICAL_COLS: Final[list[str]] = [
    SOURCE,
    DISTRICT,
    NEIGHBORHOOD,
]

DATETIME_COLS: Final[list[str]] = [
    PUBLISHED_AT,
    INGESTED_AT,
]

REQUIRED_COLS: Final[list[str]] = [
    LISTING_ID,
    SOURCE,
    DISTRICT,
    AREA_M2,
    ROOMS,
    TOTAL_PRICE,
    INGESTED_AT,
]


# ----- valid sources -----

SOURCE_KAGGLE: Final[str] = "kaggle"
SOURCE_DIVAR: Final[str] = "divar"

VALID_SOURCES: Final[set[str]] = {SOURCE_KAGGLE, SOURCE_DIVAR}


# ----- sanity bounds -----
# این‌ها برای validation هستند. اگر یک رکورد بیرون این بازه‌ها بود،
# با احتمال زیاد data quality issue است (یا outlier واقعی، که باز هم
# باید بررسی شود).

MIN_AREA_M2: Final[float] = 15.0
MAX_AREA_M2: Final[float] = 2000.0

MIN_ROOMS: Final[int] = 0
MAX_ROOMS: Final[int] = 15

MIN_YEAR_BUILT: Final[int] = 1300  # شمسی، حدوداً 1920 میلادی
MAX_YEAR_BUILT: Final[int] = 1430  # حدود 2050 میلادی، حاشیه گذاشتم

MIN_PRICE_IRR: Final[float] = 1e8  # 100 million toman, lower bound sanity
MAX_PRICE_IRR: Final[float] = 1e13  # 10 trillion, upper bound sanity
