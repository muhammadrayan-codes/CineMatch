import requests

def test_wiki_improved(title):
    headers = {
        'User-Agent': 'CineMatch/1.0 (contact@example.com)'
    }
    try:
        clean = title.split(" (")[0]
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + clean.replace(" ", "_")
        print(f"Testing: {title} -> {url}")
        res = requests.get(url, headers=headers, timeout=5)
        print(f"Status: {res.status_code}")
        if res.status_code == 200:
            data = res.json()
            poster = data.get("thumbnail", {}).get("source", "NOT FOUND")
            print(f"Poster: {poster}")
        else:
            print(f"Failed with status: {res.status_code}")
    except Exception as e:
        print(f"Error: {e}")

movies = ["Toy Story (1995)", "Jumanji (1995)"]
for m in movies:
    test_wiki_improved(m)
