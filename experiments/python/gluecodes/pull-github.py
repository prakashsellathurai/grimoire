"""
GitHub Projects Lister (made with clanker)
======================
Fetches all your GitHub repositories and lists them with
descriptions and languages/technologies used.

SETUP:
  1. Set your GitHub username below (required)
  2. Optionally set a Personal Access Token for private repos
     → Create one at: https://github.com/settings/tokens
       (needs the 'repo' scope for private repos, or 'public_repo' for public only)
  3. Choose your output format

USAGE:
  pip install requests
  python github_projects.py
"""

import sys
import requests
import json
import csv
import os
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURATION — edit these values
# ─────────────────────────────────────────────
GITHUB_USERNAME = "your_username_here"       # ← required
GITHUB_TOKEN    = ""                         # ← optional (leave empty for public repos only)
OUTPUT_FORMAT   = "markdown"                 # options: "terminal", "markdown", "json", "csv"
OUTPUT_FILE     = "github_projects"          # filename (without extension)
INCLUDE_FORKS   = False                      # set True to include forked repos
INCLUDE_ARCHIVED = False                     # set True to include archived repos
SORT_BY         = "updated"                  # options: "updated", "created", "pushed", "full_name"
# ─────────────────────────────────────────────


def fetch_all_repos(username: str, token: str = "") -> list[dict]:
    """Fetch all repos for a user, handling pagination."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    repos = []
    page = 1

    while True:
        url = f"https://api.github.com/users/{username}/repos"
        params = {
            "per_page": 100,
            "page": page,
            "sort": SORT_BY,
            "direction": "desc",
            "type": "owner",  # only repos the user owns (not starred etc.)
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 401:
            raise ValueError("❌ Invalid or expired GitHub token.")
        elif response.status_code == 404:
            raise ValueError(f"❌ GitHub user '{username}' not found.")
        elif response.status_code != 200:
            raise ValueError(f"❌ GitHub API error {response.status_code}: {response.text}")

        page_repos = response.json()
        if not page_repos:
            break

        repos.extend(page_repos)
        page += 1

        # If less than 100 results, we've hit the last page
        if len(page_repos) < 100:
            break

    return repos


def fetch_repo_languages(repo: dict, headers: dict) -> list[str]:
    """Fetch the language breakdown for a single repo."""
    lang_url = repo.get("languages_url", "")
    if not lang_url:
        return []
    resp = requests.get(lang_url, headers=headers)
    if resp.status_code == 200:
        langs = resp.json()
        # Return languages sorted by bytes of code (most used first)
        return sorted(langs, key=langs.get, reverse=True)
    return []


def build_project_list(repos: list[dict], token: str = "") -> list[dict]:
    """Filter repos and enrich them with language data."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    projects = []
    total = len(repos)

    for i, repo in enumerate(repos, 1):
        # Apply filters
        if not INCLUDE_FORKS and repo.get("fork"):
            continue
        if not INCLUDE_ARCHIVED and repo.get("archived"):
            continue

        print(f"  Fetching languages... ({i}/{total}) {repo['name']}", end="\r")

        languages = fetch_repo_languages(repo, headers)
        primary_lang = repo.get("language") or (languages[0] if languages else "—")

        projects.append({
            "name":        repo["name"],
            "description": repo.get("description") or "No description provided.",
            "url":         repo["html_url"],
            "primary_language": primary_lang,
            "all_languages": languages if languages else ([primary_lang] if primary_lang != "—" else []),
            "topics":      repo.get("topics", []),
            "stars":       repo.get("stargazers_count", 0),
            "forks":       repo.get("forks_count", 0),
            "is_private":  repo.get("private", False),
            "updated_at":  repo.get("updated_at", "")[:10],  # YYYY-MM-DD
            "created_at":  repo.get("created_at", "")[:10],
        })

    print()  # newline after progress indicator
    return projects


# ─────────────────────────────────────────────
#  OUTPUT FUNCTIONS
# ─────────────────────────────────────────────

def format_languages(project: dict) -> str:
    langs = project["all_languages"]
    if not langs:
        return "—"
    return ", ".join(langs)


def output_terminal(projects: list[dict]):
    print(f"\n{'─'*70}")
    print(f"  GitHub Projects for @{GITHUB_USERNAME}  ({len(projects)} repos)")
    print(f"{'─'*70}\n")

    for p in projects:
        visibility = "🔒" if p["is_private"] else "🌐"
        stars = f"⭐ {p['stars']}" if p["stars"] else ""
        print(f"{visibility}  {p['name']}  {stars}")
        print(f"   {p['description']}")
        print(f"   🛠  {format_languages(p)}")
        if p["topics"]:
            print(f"   🏷  {', '.join(p['topics'])}")
        print(f"   🔗  {p['url']}  │  Updated: {p['updated_at']}")
        print()


def output_markdown(projects: list[dict], filepath: str):
    lines = [
        f"# GitHub Projects — @{GITHUB_USERNAME}",
        f"\n> Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}  •  {len(projects)} repositories\n",
        "---\n",
    ]

    for p in projects:
        visibility = "🔒 Private" if p["is_private"] else "🌐 Public"
        stars = f" · ⭐ {p['stars']}" if p["stars"] else ""
        lines.append(f"## [{p['name']}]({p['url']})")
        lines.append(f"*{visibility}{stars} · Updated {p['updated_at']}*\n")
        lines.append(f"{p['description']}\n")
        lines.append(f"**Languages / Technologies:** {format_languages(p)}")
        if p["topics"]:
            lines.append(f"\n**Topics:** {', '.join(p['topics'])}")
        lines.append("\n---\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅  Markdown saved → {filepath}")


def output_json(projects: list[dict], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(projects, f, indent=2, ensure_ascii=False)
    print(f"✅  JSON saved → {filepath}")


def output_csv(projects: list[dict], filepath: str):
    fieldnames = ["name", "description", "primary_language", "all_languages",
                  "topics", "stars", "forks", "is_private", "updated_at", "url"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in projects:
            row = {**p}
            row["all_languages"] = ", ".join(p["all_languages"])
            row["topics"] = ", ".join(p["topics"])
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"✅  CSV saved → {filepath}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        GITHUB_USERNAME = sys.argv[1]

    if GITHUB_USERNAME == "your_username_here":
        print("⚠️  Please pass your GITHUB_USERNAME as argument.")
        return

    print(f"\n🔍 Fetching repos for @{GITHUB_USERNAME}...")
    repos = fetch_all_repos(GITHUB_USERNAME, GITHUB_TOKEN)
    print(f"   Found {len(repos)} repo(s). Enriching with language data...")

    projects = build_project_list(repos, GITHUB_TOKEN)
    print(f"   {len(projects)} project(s) after filtering.\n")

    if not projects:
        print("No projects to display after applying filters.")
        return

    fmt = OUTPUT_FORMAT.lower()

    if fmt == "terminal":
        output_terminal(projects)

    elif fmt == "markdown":
        path = f"{OUTPUT_FILE}.md"
        output_terminal(projects)   # also print to terminal
        output_markdown(projects, path)

    elif fmt == "json":
        path = f"{OUTPUT_FILE}.json"
        output_json(projects, path)

    elif fmt == "csv":
        path = f"{OUTPUT_FILE}.csv"
        output_csv(projects, path)

    else:
        print(f"❌ Unknown OUTPUT_FORMAT '{fmt}'. Choose: terminal, markdown, json, csv")


if __name__ == "__main__":
    main()