#!/usr/bin/env python3
"""
Fetch the last ~year of GitHub contributions for $GH_USER via GraphQL and
emit _data/contributions.json. Used by the Jekyll site to render a
GitHub-style heatmap.

Required env:
  GH_USER   - GitHub login (default: sumin1ee)
  GH_TOKEN  - PAT (read:user scope) for private counts, OR repo's
              built-in GITHUB_TOKEN (public-only).

Output: _data/contributions.json
  {
    "user":   "sumin1ee",
    "fetched_at": "2026-05-13T...Z",
    "total":  1234,
    "weeks":  [{ "days": [{ "date": "2025-05-12", "count": 0, "level": 0 }, ... ] }, ...],
    "streak": { "current": 12, "longest": 47 }
  }
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

USER = os.environ.get("GH_USER", "sumin1ee")
TOKEN = os.environ.get("GH_TOKEN")
OUT = "_data/contributions.json"

if not TOKEN:
    print("ERROR: GH_TOKEN is not set", file=sys.stderr)
    sys.exit(1)

QUERY = """
query($login: String!) {
  user(login: $login) {
    contributionsCollection {
      contributionCalendar {
        totalContributions
        weeks {
          contributionDays {
            date
            contributionCount
            contributionLevel
          }
        }
      }
    }
  }
}
"""

LEVEL_MAP = {
    "NONE": 0,
    "FIRST_QUARTILE": 1,
    "SECOND_QUARTILE": 2,
    "THIRD_QUARTILE": 3,
    "FOURTH_QUARTILE": 4,
}


def fetch():
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": QUERY, "variables": {"login": USER}}).encode("utf-8"),
        headers={
            "Authorization": f"bearer {TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": f"contrib-fetch ({USER})",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def compute_streaks(weeks):
    """Returns (current_streak, longest_streak) measured in days."""
    days = []
    for w in weeks:
        for d in w["days"]:
            days.append((d["date"], d["count"]))
    # walk chronologically
    days.sort(key=lambda x: x[0])
    longest = 0
    cur = 0
    for _date, count in days:
        if count > 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    # current streak: count backwards from today
    cur = 0
    for _date, count in reversed(days):
        if count > 0:
            cur += 1
        else:
            break
    return cur, longest


def main():
    try:
        payload = fetch()
    except urllib.error.HTTPError as e:
        print(f"HTTP error {e.code}: {e.read().decode('utf-8', errors='ignore')[:400]}", file=sys.stderr)
        sys.exit(1)

    if "errors" in payload:
        print(f"GraphQL errors: {payload['errors']}", file=sys.stderr)
        sys.exit(1)

    cal = payload["data"]["user"]["contributionsCollection"]["contributionCalendar"]
    weeks_out = []
    for w in cal["weeks"]:
        days_out = []
        for d in w["contributionDays"]:
            days_out.append({
                "date": d["date"],
                "count": d["contributionCount"],
                "level": LEVEL_MAP.get(d["contributionLevel"], 0),
            })
        weeks_out.append({"days": days_out})

    current, longest = compute_streaks(weeks_out)

    out = {
        "user": USER,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "total": cal["totalContributions"],
        "weeks": weeks_out,
        "streak": {"current": current, "longest": longest},
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {OUT}: {out['total']} contributions over {len(weeks_out)} weeks, "
          f"streak now={current}, longest={longest}")


if __name__ == "__main__":
    main()
