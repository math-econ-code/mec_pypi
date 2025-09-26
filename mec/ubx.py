def get_token(platform = 'colab',ENV_PATH=None):
    if platform == 'colab':
        return get_token_colab(ENV_PATH)
    else:
        raise Exception("Platform "+platform+' is not supported.')

def get_token_colab( ENV_PATH=None):
    from google.colab import drive
    drive.mount('/content/drive')
    if ENV_PATH is None:
        ENV_PATH = "/content/drive/MyDrive/secrets/github.env"
    import os
    from dotenv import load_dotenv
    # Load token from Drive .env
    loaded = load_dotenv(ENV_PATH, override=True)
    assert loaded, f"Could not load {ENV_PATH}"
    TOKEN = os.getenv("GITHUB_TOKEN")
    assert TOKEN, "Missing GITHUB_TOKEN in .env"
    return TOKEN


def create_gist(username,TOKEN,verbose = 0):
    from github import Github, Auth, InputFileContent
    import os, requests
    filename = username + "-actions.txt"
    # (Optional) sanity check: see scopes returned by API headers
    resp = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json"},
        timeout=20,
    )
    resp.raise_for_status()
    if verbose>0:
        print("Authenticated as:", resp.json().get("login"))
        print("Token scopes (X-OAuth-Scopes):", resp.headers.get("X-OAuth-Scopes"))

    # Use new-style auth to avoid deprecation warning
    g = Github(auth=Auth.Token(TOKEN))

    # Create a local file to upload
    FILEPATH = "/content/" + filename
    with open(FILEPATH, "w", encoding="utf-8") as f:
        f.write("0")

    # Read content
    with open(FILEPATH, "r", encoding="utf-8") as f:
        content = f.read()
    filename = os.path.basename(FILEPATH)

    # Create the gist
    DESCRIPTION = "No action posted"
    gist = g.get_user().create_gist(
        public=True,
        files={filename: InputFileContent(content)},
        description=DESCRIPTION,
    )
    if verbose>0:
        print("Gist created!")
        print("Web view:", gist.html_url)
        print("Raw file:", gist.files[filename].raw_url)
    return filename

def update_gist(username,TOKEN,NEW_CONTENT,verbose = 0):
    FILENAME = username+"-actions.txt"      # must match the file inside the Gist
    import os, requests
    from datetime import datetime, timezone, timedelta
    from github import Github, Auth, InputFileContent

    # --- Config you provide ---
    now_paris = datetime.now(timezone(timedelta(hours=2)))
    date_str = now_paris.strftime("%Y-%m-%d %H:%M:%S")
    FILENAME_MATCH = FILENAME   # <-- put your filename here
    NEW_DESCRIPTION = "Updated action"

    # --- Optional: confirm identity & scopes (helps debug 403/404) ---
    r = requests.get("https://api.github.com/user",
                    headers={"authorization": f"token {TOKEN}",
                            "Accept":"application/vnd.github+json"},
                    timeout=20)
    r.raise_for_status()
    print("Authenticated as:", r.json().get("login"))
    print("Token scopes (X-OAuth-Scopes):", r.headers.get("X-OAuth-Scopes"))

    # --- Auth (new-style) ---
    g = Github(auth=Auth.Token(TOKEN))
    me = g.get_user()

    # --- Find gists that contain the filename, choose the most recently updated ---
    candidates = []
    for gs in me.get_gists():  # includes your secret gists
        if FILENAME_MATCH in gs.files:
            candidates.append(gs)

    if not candidates:
        raise RuntimeError(f"No gist owned by you contains a file named '{FILENAME_MATCH}'.")

    # pick the most recently updated gist
    candidates.sort(key=lambda x: x.updated_at or datetime(1970,1,1, tzinfo=timezone.utc), reverse=True)
    gist = candidates[0]
    print(f"Resolved gist: {gist.id} | desc: {gist.description!r} | updated_at: {gist.updated_at}")

    # --- Update that file (adds the file if it didn't exist; here it does) ---
    gist.edit(
        description=NEW_DESCRIPTION,
        files={FILENAME_MATCH: InputFileContent(NEW_CONTENT)}
    )
    if verbose>0:
        print("Gist updated!")
        print("Web view:", gist.html_url)
        print("Raw file:", gist.files[FILENAME_MATCH].raw_url)

    return 
