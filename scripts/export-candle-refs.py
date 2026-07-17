#!/usr/bin/env python3
"""Export candle workspace dependency repositories into candle_refs."""

from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "candle_refs"
CARGO_TOML = ROOT / "Cargo.toml"
CRATES_API = "https://crates.io/api/v1/crates"


def run(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT)


def normalize_repo(url: str) -> str | None:
    url = url.strip().rstrip("/")
    if not url:
        return None
    if url.endswith(".git"):
        url = url[:-4]
    if "github.com" not in url and "gitlab.com" not in url:
        return None
    return url


def workspace_dependencies() -> dict[str, str]:
    text = CARGO_TOML.read_text(encoding="utf-8")
    section = re.search(r"\[workspace\.dependencies\](.*?)(?:\n\[|\Z)", text, re.S)
    if not section:
        raise RuntimeError("workspace.dependencies section not found")

    deps: dict[str, str] = {}
    for line in section.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([A-Za-z0-9_-]+)\s*=", line)
        if not match:
            continue
        crate = match.group(1)
        if "path" in line:
            continue
        version_match = re.search(r'version\s*=\s*"([^"]+)"', line)
        if version_match:
            deps[crate] = version_match.group(1)
            continue
        simple_match = re.match(rf"{re.escape(crate)}\s*=\s*\"([^\"]+)\"", line)
        if simple_match:
            deps[crate] = simple_match.group(1)
    return deps


def version_key(value: str) -> tuple:
    parts: list[tuple[int, str]] = []
    for segment in value.split("."):
        digits = ""
        suffix = ""
        for char in segment:
            if char.isdigit():
                if suffix:
                    suffix += char
                else:
                    digits += char
            else:
                suffix += char
        parts.append((int(digits) if digits else 0, suffix))
    return tuple(parts)


def resolve_crate_version(name: str, declared: str, resolved: dict[str, str]) -> str:
    if name in resolved:
        return resolved[name]

    if not declared:
        raise RuntimeError("no declared version in Cargo.toml and not in cargo metadata")

    if re.fullmatch(r"\d+", declared) or re.fullmatch(r"\d+\.\d+", declared):
        with urllib.request.urlopen(f"{CRATES_API}/{name}", timeout=60) as response:
            payload = json.load(response)
        versions = [version["num"] for version in payload["versions"]]
        matched = [
            version
            for version in versions
            if version == declared or version.startswith(f"{declared}.")
        ]
        if not matched:
            raise RuntimeError(f"no crates.io version for {name} matching {declared}")
        return sorted(matched, key=version_key)[-1]

    return declared


def resolved_versions() -> dict[str, str]:
    meta = json.loads(run(["cargo", "metadata", "--format-version", "1"], ROOT))
    versions: dict[str, str] = {}
    for pkg in meta["packages"]:
        name = pkg["name"]
        version = pkg["version"]
        current = versions.get(name)
        if current is None or version > current:
            versions[name] = version
    return versions


def crates_io_info(name: str, version: str) -> dict[str, str]:
    url = f"{CRATES_API}/{name}/{version}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    crate = payload["version"]
    repo = normalize_repo(crate.get("repository") or "") or ""
    return {
        "version": crate["num"],
        "source": repo,
        "download": crate["dl_path"],
    }


def tag_candidates(name: str, version: str) -> list[str]:
    return [
        f"v{version}",
        version,
        f"{name}-{version}",
        f"{name}-v{version}",
        f"{name}_{version}",
    ]


def clone_repo(name: str, source: str, version: str, dest: Path) -> str | None:
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    for tag in tag_candidates(name, version):
        try:
            run(
                ["git", "clone", "--depth", "1", "--branch", tag, source, str(dest)],
                ROOT,
            )
            return tag
        except subprocess.CalledProcessError:
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)

    try:
        run(["git", "clone", "--depth", "1", source, str(dest)], ROOT)
        for tag in tag_candidates(name, version):
            try:
                run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", f"refs/tags/{tag}:refs/tags/{tag}"], ROOT)
                run(["git", "-C", str(dest), "checkout", tag], ROOT)
                return tag
            except subprocess.CalledProcessError:
                continue
    except subprocess.CalledProcessError:
        pass
    finally:
        if dest.exists() and not (dest / ".git").exists():
            shutil.rmtree(dest, ignore_errors=True)

    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    return None


def download_crate_source(name: str, version: str, dest: Path) -> str:
    info = crates_io_info(name, version)
    download_url = f"https://crates.io{info['download']}"
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(download_url, timeout=120) as response:
        data = response.read()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise RuntimeError(f"empty crate archive for {name}")
        top = members[0].name.split("/")[0]
        extract_root = OUT / f".tmp-{name}"
        if extract_root.exists():
            shutil.rmtree(extract_root, ignore_errors=True)
        archive.extractall(extract_root, filter="data")
        extracted = extract_root / top
        shutil.move(str(extracted), str(dest))
        shutil.rmtree(extract_root, ignore_errors=True)

    return f"crates-io-{version}"


def export_dependency(name: str, version: str) -> tuple[str, str, str, str]:
    info = crates_io_info(name, version)
    source = info["source"]
    dest = OUT / name

    if source:
        ref = clone_repo(name, source, version, dest)
        if ref is not None:
            return version, source, ref, "git"

    ref = download_crate_source(name, version, dest)
    return version, source or f"https://crates.io/crates/{name}", ref, "crate"


def main() -> int:
    wanted = workspace_dependencies()
    resolved = resolved_versions()

    OUT.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "# candle_refs — workspace.dependencies из candle/Cargo.toml",
        "# Версии — resolved через cargo metadata (ветка wgpu/vulkan).",
        "# Сгенерировано scripts/export-candle-refs.py",
        "",
        "[policy]",
        "allow_prerelease = true",
        "allow_dev_branches = false",
        "",
    ]

    cloned = 0
    failed = 0

    for name, declared in wanted.items():
        try:
            version = resolve_crate_version(name, declared, resolved)
        except Exception as exc:  # noqa: BLE001
            manifest_lines.extend(
                [
                    f"[repos.{name}]",
                    f'declared = "{declared}"',
                    f'note = "version resolve failed: {exc}"',
                    "",
                ]
            )
            failed += 1
            print(f"FAILED {name}: {exc}")
            continue

        try:
            resolved_version, source, ref, kind = export_dependency(name, version)
            manifest_lines.extend(
                [
                    f"[repos.{name}]",
                    f'declared = "{declared}"',
                    f'version = "{resolved_version}"',
                    f'ref = "{ref}"',
                    f'source = "{source}"',
                    f'kind = "{kind}"',
                    "",
                ]
            )
            cloned += 1
            print(f"exported {name} @ {ref} ({kind})")
        except Exception as exc:  # noqa: BLE001 - export script reports all failures
            manifest_lines.extend(
                [
                    f"[repos.{name}]",
                    f'version = "{version}"',
                    f'note = "export failed: {exc}"',
                    "",
                ]
            )
            failed += 1
            print(f"FAILED {name} {version}: {exc}")

    manifest_path = OUT / "manifest.toml"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

    print(f"\nDone: {cloned} exported, {failed} failed, {len(wanted)} workspace deps")
    print(f"manifest: {manifest_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
