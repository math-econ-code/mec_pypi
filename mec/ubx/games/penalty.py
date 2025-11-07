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


ua = "https://raw.githubusercontent.com/alfredgalichon/game0-admin_output/main/action.json"
history = url_fetch(ua,"history")

hist = [(i,j) for (i,j,_) in history]
nbObs = len(hist)
np.set_printoptions(precision=2, suppress=True)
np.array([[hist.count((i,j)) for j in range(3)] for i in range(3)])/nbObs