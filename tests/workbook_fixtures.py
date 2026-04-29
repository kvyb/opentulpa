from __future__ import annotations

from io import BytesIO
from pathlib import Path

from openpyxl import Workbook

SAMPLE_VEHICLE_SERVICES_XLSX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
SAMPLE_VEHICLE_SERVICES_XLSX_FILENAME = "sample_vehicle_services_price.xlsx"


def sample_vehicle_services_xlsx_bytes() -> bytes:
    workbook = Workbook()
    wash = workbook.active
    wash.title = "Мойка"
    wash.append(
        [
            "Услуга",
            "Описание",
            "C, D, E-Class / Mini SUV",
            "C, D, E-Class / Mini SUV",
            "S-Class / SUV / Jeep 3-doors / Mini Minivan",
            "S-Class / SUV / Jeep 3-doors / Mini Minivan",
            "Jeep / Minivan / Pick-Up",
            "Jeep / Minivan / Pick-Up",
            "Jeep XXL / Minivan XXL / Pick-Up XXL / MicroBus",
            "Jeep XXL / Minivan XXL / Pick-Up XXL / MicroBus",
        ]
    )
    wash.append(
        [
            "Экспресс мойка кузова",
            "бесконтакт + ковры",
            700,
            700,
            900,
            900,
            1000,
            1000,
            1200,
            1200,
        ]
    )
    wash.append(
        [
            "2х-фазная мойка кузова",
            "бесконтакт + ковры + обезжириватель + ручная мойка",
            1000,
            1000,
            1200,
            1200,
            1200,
            1200,
            1500,
            1500,
        ]
    )
    wash.append(
        [
            "3х-фазная мойка кузова",
            "деликатная мойка + дополнительная химия",
            1400,
            1400,
            1700,
            1700,
            1900,
            1900,
            2300,
            2300,
        ]
    )

    tires = workbook.create_sheet("Шиномонтаж")
    tires.append(
        [
            "Услуга",
            "Размер",
            "Седан",
            "Внедорожник / кросовер",
            "Внедорожник / кросовер + низкий профиль",
        ]
    )
    tires.append(["Комплект 18`R", "18`R", 2800, 3300, 3700])
    tires.append(["Комплект 19`R", "19`R", 3000, 3500, 4000])
    tires.append(["Комплект 20`R", "20`R", 3400, 3900, 4500])

    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def write_sample_vehicle_services_xlsx(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sample_vehicle_services_xlsx_bytes())
    return path
