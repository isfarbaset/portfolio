"""Rebuild docs/archive: the day-olds case on the homepage.

Each earlier design of the homepage is pulled out of git as it was, with only
the local files it needs, so it keeps working as a page of its own at
docs/archive/<batch>/. Three things are changed on the way out:

  - links to sibling pages point at the live copy (../../aqi.html), or at
    nothing if that page is gone, so nobody lands on a 404
  - search engines are told to skip it
  - opened on its own (not inside the case on the homepage), it wears a small
    tag saying what it is, with a way back

Run from anywhere:  python3 code/build_day_olds.py
It only ever writes inside docs/archive.
"""
import os
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(REPO, 'docs')
OUT = os.path.join(DOCS, 'archive')
GIT = next((g for g in ('/Library/Developer/CommandLineTools/usr/bin/git', '/usr/bin/git') if os.path.exists(g)), 'git')

# folder, the last commit of that design, what the tag calls it
BATCHES = [
    ('2025-02', '70086cd', 'February 2025'),
    ('2025-04', '8c96ad6', 'April 2025'),
    ('2025-08', '418f103', 'August 2025'),
    ('2025-10', 'fabd0e0', 'October 2025'),
    ('2026-05', '28ce306', 'May 2026'),
]

ASSET = re.compile(r'\.(css|js|png|jpe?g|gif|webp|svg|ico|woff2?|ttf|eot)$', re.I)
REMOTE = re.compile(r'^(https?:|mailto:|tel:|//|data:|#|javascript:)')

TAG = '''
<!-- added by code/build_day_olds.py -->
<script>
if (window.top === window) addEventListener('DOMContentLoaded', function () {
  var a = document.createElement('a');
  a.href = '../../#day-olds';
  a.textContent = 'A day-old, baked %s. Back to today\\u2019s \\u2192';
  a.style.cssText = 'position:fixed;left:50%%;bottom:18px;z-index:2147483647;transform:translateX(-50%%) rotate(-1.5deg);' +
    'padding:9px 18px;background:#f4efe6;color:#231d19;font:italic 15px/1.3 Georgia,serif;text-decoration:none;white-space:nowrap;' +
    'box-shadow:0 1px 1px rgba(30,18,8,.12),0 14px 24px -14px rgba(30,18,8,.5)';
  document.body.appendChild(a);
});
</script>
'''


def show(commit, path):
    r = subprocess.run([GIT, '-C', REPO, 'show', f'{commit}:docs/{path}'], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def save(dest, data):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as f:
        f.write(data)


def build(name, commit, label):
    out = os.path.join(OUT, name)
    html = show(commit, 'index.html').decode('utf-8')

    # every local file the page asks for, in attributes or in its scripts
    refs = {os.path.normpath(u) for u in re.findall(r'(?:src|href)="([^"#?]+)', html) if not REMOTE.match(u) and ASSET.search(u)}
    refs |= set(re.findall(r'''images/[^"'`<>)]+?\.(?:jpe?g|png|webp|gif)''', html))

    seen, missing, size = set(), [], 0
    while refs:
        p = refs.pop()
        if p in seen or p.startswith('..'):
            continue
        seen.add(p)
        data = show(commit, p)
        if data is None:
            missing.append(p)
            continue
        save(os.path.join(out, p), data)
        size += len(data)
        if p.endswith('.css'):
            for u in re.findall(r'url\(["\']?([^)"\']+)', data.decode('utf-8', 'replace')):
                u = re.sub(r'[?#].*', '', u)
                if u and not REMOTE.match(u):
                    refs.add(os.path.normpath(os.path.join(os.path.dirname(p), u)))

    # sibling pages: the live copy if there still is one, otherwise nowhere
    def page(m):
        target = m.group(2)
        clean = target[2:] if target.startswith('./') else target
        if clean == 'index.html':
            return m.group(0)
        live = os.path.exists(os.path.join(DOCS, clean))
        return f'{m.group(1)}"{"../../" + clean if live else "#"}"'
    html = re.sub(r'(href=)"((?:\./)?[^"#?:]+\.(?:html|pdf))"', page, html)

    html = html.replace('</head>', '<meta name="robots" content="noindex" />\n</head>', 1)
    html = html.replace('</body>', TAG % label + '</body>', 1)
    save(os.path.join(out, 'index.html'), html.encode('utf-8'))
    print(f'{name}  {commit}  {len(seen) + 1:3d} files  {size / 1048576:5.1f} MB' + (f'  missing: {missing}' if missing else ''))


if __name__ == '__main__':
    for batch in BATCHES:
        build(*batch)
