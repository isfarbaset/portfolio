"""Daily check that the portfolio and every project site are up and not flagged.

Google Safe Browsing judges the whole isfarbaset.github.io domain, so one
flagged project takes the portfolio down with it. This script asks Google's
Transparency Report for the domain's status and confirms each site loads.
Exits non-zero with a summary on stdout if anything is wrong.
"""
import json
import sys
import urllib.parse
import urllib.request

DOMAIN = "isfarbaset.github.io"
SITES = [
    "portfolio/",
    "netflix-recap/",
    "github-insights/",
    "almanac/",
    "dmv-blooms/",
    "fall-2024-project-team-29/",
    "story-project/",
    "ev-insights/",
]

# Transparency Report status codes (unofficial, but stable for years).
UNSAFE = {2: "unsafe content found", 3: "some pages are unsafe"}

# Words that make a github.io page look like brand phishing to Safe Browsing.
BRAND_WORDS = ["sign in", "log in", "verify your account", "netflix", "paypal", "apple id"]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "site-health-check"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read().decode("utf-8", "ignore")


def safe_browsing(site):
    q = urllib.parse.quote(site, safe="")
    _, body = get(f"https://transparencyreport.google.com/transparencyreport/api/v3/safebrowsing/status?site={q}")
    return json.loads(body.split("\n", 1)[1])[0][1]


problems = []

status = safe_browsing(DOMAIN)
print(f"Safe Browsing status for {DOMAIN}: {status}")
if status in UNSAFE:
    problems.append(f"Google Safe Browsing: {UNSAFE[status]} on {DOMAIN}. "
                    "Check Search Console > Security issues.")

for path in SITES:
    url = f"https://{DOMAIN}/{path}"
    try:
        code, body = get(url)
    except Exception as e:
        problems.append(f"{url} failed to load: {e}")
        continue
    print(f"{code} {url}")
    lower = body.lower()
    hits = [w for w in BRAND_WORDS if w in lower]
    if hits:
        print(f"  note: page mentions {hits} (fine in a disclaimer, risky in a heading or button)")

if problems:
    print("\n".join(["", "PROBLEMS:"] + problems))
    sys.exit(1)
