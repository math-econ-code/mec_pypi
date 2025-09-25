def create_gist(username, platform = 'colab', verbose = 0):
   if platform == 'colab':
      create_gist_colab(username,verbose=verbose)
    else:
      raise Exception("Platform "+platform+' is not supported.')
      

  

def create_gist_colab(username,verbose = 0, ENV_PATH=None):
   filename = username + "-actions.txt"
   if ENV_PATH is None:
      ENV_PATH = "/content/drive/MyDrive/secrets/github.env"
    # Install deps
    !pip -q install PyGithub python-dotenv

    import os, requests
    from dotenv import load_dotenv
    from github import Github, Auth, InputFileContent

    # Load token from Drive .env
    loaded = load_dotenv(ENV_PATH, override=True)
    assert loaded, f"Could not load {ENV_PATH}"
    TOKEN = os.getenv("GITHUB_TOKEN")
    assert TOKEN, "Missing GITHUB_TOKEN in .env"

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
        f.write("hello, world\n")

    # Read content
    with open(FILEPATH, "r", encoding="utf-8") as f:
        content = f.read()
    filename = os.path.basename(FILEPATH)

    import smtplib
    from email.mime.text import MIMEText

    # Create the gist
    DESCRIPTION = "Example gist created from Colab via .env and PyGithub"
    gist = g.get_user().create_gist(
        public=True,
        files={filename: InputFileContent(content)},
        description=DESCRIPTION,
    )

    print("Gist created!")
    print("Web view:", gist.html_url)
    print("Raw file:", gist.files[filename].raw_url)
   return()
