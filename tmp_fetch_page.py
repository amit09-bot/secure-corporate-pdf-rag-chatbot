import urllib.request
url = 'http://127.0.0.1:8501'
with urllib.request.urlopen(url) as r:
    print(r.status)
    print(r.read(500).decode('utf-8', 'ignore'))
