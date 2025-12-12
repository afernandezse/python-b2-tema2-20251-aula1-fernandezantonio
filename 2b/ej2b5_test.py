import pandas as pd
import requests

from ej2b5 import read_population_data, get_table_by_string_match, count_tables

# URL for the Wikipedia page with population data
URL = "https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population"


def test_read_population_data():
    # Check if the Wikipedia page is reachable
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(URL, headers=headers)
    assert response.status_code == 200, "Wikipedia page not reachable"

    # Check if tables can be read
    tables = read_population_data(URL)
    assert isinstance(tables, list), "Should return a list"
    assert all(
        isinstance(table, pd.DataFrame) for table in tables
    ), "Each item should be a DataFrame"


def test_get_table_by_string_match():
    tables = read_population_data(URL)
    # Use a match_text that is known to exist in the tables
    match_text = "Spain"
    matched_table = get_table_by_string_match(tables, match_text)
    assert matched_table is not None, f"Should find a table with text {match_text}"
    assert (
        match_text in matched_table.to_string()
    ), f"The table should contain the text {match_text}"


def test_count_tables():
    tables = read_population_data(URL)
    assert isinstance(count_tables(tables), int), "Should return an integer"
    assert count_tables(tables) > 0, "Should find one or more tables"
