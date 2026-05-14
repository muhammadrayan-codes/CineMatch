import requests

OMDB_API_KEY = "1461bea6"

def test_omdb(title):
    try:
        clean = title.split(" (")[0]
        url = f"http://www.omdbapi.com/?t={clean}&apikey={OMDB_API_KEY}"
        print(f"Testing OMDB: {title} -> {url}")
        res = requests.get(url, timeout=5)
        print(f"Status: {res.status_code}")
        data = res.json()
        poster = data.get("Poster", "NOT FOUND")
        print(f"Poster: {poster}")
    except Exception as e:
        print(f"Error: {e}")

movies = ["Toy Story (1995)", "Heat (1995)"]
for m in movies:
    test_omdb(m)
