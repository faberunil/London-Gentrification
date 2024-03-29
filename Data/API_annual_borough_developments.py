import requests
import json
import csv

base_url = "https://planningdata.london.gov.uk/api-guest/applications/_search"

# Define the range of years to query
start_year = 2010
end_year = 2020

for year in range(start_year, end_year + 1):
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"status.raw": "Completed"}},
                    {"range": {
                        "actual_completion_date": {
                            "gte": f"01/01/{year}",
                            "lte": f"31/12/{year}"
                        }
                    }}
                ],
                "should": [
                    {"terms": {"development_type": [
                        "Major dwellings", "Major offices-R and D-light industry", 
                        "Major general industry-storage-warehousing", "Major retail and service", 
                        "Major traveller caravan pitches", "Major all other major developments", 
                        "Minor dwellings", "Minor Offices-R and D-light industry", 
                        "Minor general industry-storage-warehousing", "Minor retail and service", 
                        "County Matters", "New Build", "Conversion"
                    ]}}
                ],
                "minimum_should_match": 1
            }
        },
        "_source": ["lpa_name", "borough", "actual_completion_date", "development_type"],
        "size": 10000
    }

    response = requests.post(base_url, json=query)
    if response.status_code == 200:
        print(f"Query for {year} successful!")
        data = response.json()

        # Flatten the nested JSON structure into a list of dictionaries
        flattened_data = []
        for hit in data.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            flattened_data.append({
                "lpa_name": source.get("lpa_name"),
                "borough": source.get("borough"),
                "actual_completion_date": source.get("actual_completion_date"),
                "development_type": source.get("development_type")
            })

        # Write the flattened data to a CSV file
        csv_filename = f"Borough_developments_{year}.csv"
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["lpa_name", "borough", "actual_completion_date", "development_type"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for item in flattened_data:
                writer.writerow(item)

        print(f"Data for {year} saved to {csv_filename}")

    else:
        print(f"Failed to execute query for {year}.")