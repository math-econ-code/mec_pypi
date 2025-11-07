import json,urllib.request, requests, numpy as np
#from datetime import datetime
#from zoneinfo import ZoneInfo  # Py3.9+

def url_fetch(url,field):
    try:
        with urllib.request.urlopen(url + ("&nocache=1" if "?" in url else "?nocache=1")) as r:
            data = json.load(r)
            return  data[field]
    except Exception as e:
        return None

def show_score(url):
    history = url_fetch(url,"history")
    hist = [(i,j) for (i,j,_) in history]
    nbObs = len(hist)
    return np.array([[hist.count((i,j)) for j in range(3)] for i in range(3)])/nbObs