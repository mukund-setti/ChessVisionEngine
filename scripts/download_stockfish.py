#!/usr/bin/env python3
"""Download and install Stockfish chess engine."""

import os
import sys
import platform
import tarfile
import zipfile
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm


STOCKFISH_RELEASES = {
    "linux-x64": "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar",
    "linux-x64-modern": "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-modern.tar",
    "macos-x64": "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-macos-x86-64-avx2.tar",
    "macos-arm64": "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-macos-m1-apple-silicon.tar",
    "windows-x64": "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.zip",
}


def get_platform_key() -> str:
    """Determine the platform-specific download key."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        return "linux-x64"
    elif system == "darwin":
        if machine == "arm64":
            return "macos-arm64"
        return "macos-x64"
    elif system == "windows":
        return "windows-x64"
    else:
        raise RuntimeError(f"Unsupported platform: {system} {machine}")


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_stockfish(archive_path: Path, dest_dir: Path) -> Path:
    """Extract Stockfish from archive and return executable path."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".tar":
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(dest_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path.suffix}")

    for root, dirs, files in os.walk(dest_dir):
        for file in files:
            if file.startswith("stockfish") and not file.endswith((".tar", ".zip")):
                return Path(root) / file

    raise FileNotFoundError("Stockfish executable not found in archive")


def main():
    """Download and install Stockfish."""
    print("Chess Vision Engine - Stockfish Installer")
    print("=" * 50)

    try:
        platform_key = get_platform_key()
        url = STOCKFISH_RELEASES[platform_key]
        print(f"Platform: {platform_key}")
        print(f"Download URL: {url}")
    except (RuntimeError, KeyError) as e:
        print(f"Error: {e}")
        print("Please download Stockfish manually from: https://stockfishchess.org/download/")
        sys.exit(1)

    if platform.system() == "Windows":
        install_dir = Path.cwd()
        exe_name = "stockfish.exe"
    else:
        install_dir = Path("/usr/local/bin")
        exe_name = "stockfish"

        if not os.access(install_dir, os.W_OK):
            install_dir = Path.cwd()
            print(f"Note: Installing to current directory (no write access to /usr/local/bin)")

    final_path = install_dir / exe_name

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        archive_name = url.split("/")[-1]
        archive_path = tmpdir / archive_name

        print(f"\nDownloading Stockfish...")
        download_file(url, archive_path)

        print(f"Extracting...")
        stockfish_exe = extract_stockfish(archive_path, tmpdir)

        print(f"Installing to {final_path}...")

        import shutil
        shutil.copy2(stockfish_exe, final_path)

        if platform.system() != "Windows":
            os.chmod(final_path, 0o755)

    print(f"\n✓ Stockfish installed successfully!")
    print(f"  Path: {final_path}")

    env_file = Path.cwd() / ".env"
    if env_file.exists():
        print(f"\nTip: Update STOCKFISH_PATH in .env to: {final_path}")
    else:
        env_example = Path.cwd() / ".env.example"
        if env_example.exists():
            print(f"\nTip: Copy .env.example to .env and set STOCKFISH_PATH={final_path}")

    print("\nTesting Stockfish...")
    import subprocess
    try:
        result = subprocess.run(
            [str(final_path), "uci"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "uciok" in result.stdout:
            print("✓ Stockfish is working correctly!")
        else:
            print("⚠ Stockfish may not be working correctly")
    except Exception as e:
        print(f"⚠ Could not test Stockfish: {e}")


if __name__ == "__main__":
    main()