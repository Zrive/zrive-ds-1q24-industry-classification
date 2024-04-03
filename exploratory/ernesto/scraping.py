from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List
import pandas as pd
import requests

BASE_URL = "https://naics.askkodiak.com/naics/2022/"
BASE_CODES = [
    "11",
    "21",
    "22",
    "23",
    "31-33",
    "42",
    "44-45",
    "48-49",
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "61",
    "62",
    "71",
    "72",
    "81",
    "92",
]


@dataclass
class BaseNaics:
    code: str
    title: str
    description: str
    descendants_code: List[str]

    @classmethod
    def from_code(cls, code: str):
        url = f"{BASE_URL}{code}"
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("h1").get_text()

        overview = soup.find(attrs={"class": "overview subsection"}).find_all("p")
        description = "\n".join([p.get_text() for p in overview])
        description = description.strip()

        descendants = soup.find(attrs={"class": "descendants subsection"}).find_all(
            "li"
        )
        all_href = [descendant.a["href"] for descendant in descendants]
        all_codes = [descendant.split("/")[-1] for descendant in all_href]

        return cls(code, title, description, all_codes)


@dataclass
class Naics:
    code: str
    title: str
    description: str

    @classmethod
    def from_code(cls, code: str):
        url = f"{BASE_URL}{code}"
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("h1").get_text()

        overview = soup.find(attrs={"class": "overview subsection"}).find_all("p")
        description = "\n".join([p.get_text() for p in overview])
        description = description.strip()

        return cls(code, title, description)


def main():
    all_base_naics: List[BaseNaics] = []

    for sector in BASE_CODES:
        all_base_naics.append(BaseNaics.from_code(sector))

    all_desc: List[Naics] = []

    for base in all_base_naics:
        for desc_code in base.descendants_code:
            try:
                all_desc.append(Naics.from_code(desc_code))
            except Exception as e:
                print(f"Failed to get {desc_code}", e)

    base_naics_df = pd.DataFrame(all_base_naics)
    desc_naics_df = pd.DataFrame(all_desc)
    naics_df = pd.concat([base_naics_df, desc_naics_df], ignore_index=True)

    naics_df.to_csv("naics.csv", index=False)
