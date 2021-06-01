"""
Search environment variables for secrets and create `.pypirc` for pipeline.
"""

import os
from pathlib import Path


def main():
    repo = os.getenv("PIP_REPOSITORY")
    user = os.getenv("PIP_USERNAME")
    passw = os.getenv("PIP_PASSWORD")
    configpath = os.path.expanduser("~/.config/pip")
    print(os.getcwd())

    if repo is user is passw is None:
        print("No secrets provided, doing nothing.")
    elif None in [repo, user, passw]:
        raise Exception("PIP environment variables incomplete.")
    else:
        if os.path.exists(f"{configpath}/pip.conf"):
            raise Exception(f"{configpath}/pip.conf exists, refusing to overwrite.")
        Path(configpath).mkdir(parents=True, exist_ok=True)
        with open(f"{configpath}/pip.conf", "w") as outfile:
            outfile.write(
                f"[global]\n"
                f"index = https://{user}:{passw}@{repo}/pypi\n"
                f"index-url = https://{user}:{passw}@{repo}/simple\n"
            )
        print("Created config ~/.config/pip/pip.conf")


if __name__ == "__main__":
    main()
