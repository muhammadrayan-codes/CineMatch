import requests

def test_wiki(title):
    try:
        clean = title.split(" (")[0]
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + clean.replace(" ", "_")
        print(f"Testing: {title} -> {url}")
        res = requests.get(url, timeout=5)
        data = res.json()
        poster = data.get("thumbnail", {}).get("source", "NOT FOUND")
        print(f"Poster: {poster}")
    except Exception as e:
        print(f"Error: {e}")

movies = ["Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)", "Heat (1995)"]
for m in movies:
    test_wiki(m)
