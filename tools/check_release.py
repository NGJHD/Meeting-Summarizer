r"""Check a published release the way a user's copy would see it.

    runtime\python.exe tools\check_release.py

Run this after `gh release create`. It asks GitHub the same question the in-app
update button asks and reports what an installed copy would actually do, which
is the part you cannot verify by looking at the releases page.

Development only; excluded from both release zips. See RELEASE_GUIDE.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import updater, version  # noqa: E402

# DOWNLOAD_MODELS.bat fetches from these by name. Deleting either breaks every
# future install, and the whisper one is our own compile -- it exists nowhere
# else, because whisper.cpp publishes no Vulkan build for Windows.
LOAD_BEARING = ("runtime-cpython-3.12.14", "whisper-vulkan-b4938")

RELEASES_API = "https://api.github.com/repos/%s/releases" % version.GITHUB_REPO

failures: list[str] = []
notes: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print("  %-4s %s%s" % ("ok" if ok else "FAIL", label,
                           ("  --  " + detail) if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def main() -> int:
    installed = version.APP_VERSION
    print("\nInstalled version in server/version.py: %s\n" % installed)

    print("What GitHub reports as the latest release")
    try:
        data = updater._get_json(version.RELEASES_API)
    except Exception as exc:  # noqa: BLE001
        print("  FAIL could not reach GitHub  --  %s" % exc)
        print("\nCannot check anything else without that. Is the network up?")
        return 1

    tag = str(data.get("tag_name") or "")
    want_tag = "v%s" % installed
    check(tag == want_tag, "tag matches the version in version.py",
          "release is %r, expected %r" % (tag, want_tag))

    assets = data.get("assets") or []
    by_name = {a["name"]: a for a in assets}
    print("  %-4s assets attached: %s" % ("--", ", ".join(by_name) or "none"))

    small = "Meeting-Summariser-%s.zip" % want_tag
    full = "Meeting-Summariser-%s-full.zip" % want_tag
    check(small in by_name, "update payload attached", small)
    if not check(full in by_name, "full bundle attached", full):
        notes.append("Without the full bundle a new install has no runtime or "
                     "binaries and cannot start.")

    for name in (small, full):
        a = by_name.get(name)
        if a and a.get("state") != "uploaded":
            check(False, "%s finished uploading" % name, "state=%s" % a.get("state"))

    print("\nWhat an installed copy would do")
    older = "%d.%d.%d" % (lambda v: (v[0], v[1], max(0, v[2] - 1)))(
        version.parse_version(installed) or (0, 0, 1))
    check(version.is_newer(tag, older),
          "an older copy (%s) is offered this release" % older)

    picked = version.pick_asset(assets)
    if check(picked is not None, "the updater finds an asset to install"):
        name = picked["name"]
        size = int(picked.get("size") or 0)
        check(name == small, "it picks the update payload, not the bundle", name)
        check(size < 50_000_000, "the download is small",
              "%.1f KB" % (size / 1024))

    print("\nThe two pre-releases DOWNLOAD_MODELS.bat depends on")
    try:
        every = updater._get_json(RELEASES_API)
        tags = {str(r.get("tag_name")): r for r in every}
        for name in LOAD_BEARING:
            r = tags.get(name)
            if check(r is not None, "%s still exists" % name):
                check(bool(r.get("prerelease")), "%s is still a pre-release" % name,
                      "if it is not, it can steal /releases/latest")
    except Exception as exc:  # noqa: BLE001
        check(False, "could not list releases", str(exc))

    print()
    if failures:
        print("%d CHECK(S) FAILED:" % len(failures))
        for f in failures:
            print("  - %s" % f)
        for n in notes:
            print("\n%s" % n)
        print("\nSee RELEASE_GUIDE.md, 'When something goes wrong'.")
        return 1

    print("ALL CHECKS PASSED -- %s is live and installable." % tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
