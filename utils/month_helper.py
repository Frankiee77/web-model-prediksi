from typing import List, Tuple

MONTH_NAMES = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

def get_previous_months(year: int, month: int, n: int = 3) -> List[Tuple[int, int]]:
    """
    Mengembalikan daftar (tahun, bulan) untuk n bulan sebelumnya dari 
    bulan dan tahun yang diberikan.
    """
    months: List[Tuple[int, int]] = []
    for i in range(n, 0, -1):
        y = year
        m = month - i
        while m <= 0:
            m += 12
            y -= 1
        months.append((y, m))
    return months

def month_name(month: int) -> str:
    """
    Mengembalikan nama bulan dalam Bahasa Indonesia untuk nomor bulan yang diberikan.
    """
    if 1 <= month <= 12:
        return MONTH_NAMES[month - 1]
    return ""

def format_month_year(year: int, month: int) -> str:
    """
    Mengembalikan format string 'Bulan Tahun' seperti 'Januari 2027'.
    """
    return f"{month_name(month)} {year}"
