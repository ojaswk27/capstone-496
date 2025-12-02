#!/usr/bin/env python3
"""
Real-data downloader for Aerospace Design Assistant
- Retries + checksums
- 2 s polite pause for NTRS
- Rich progress bar (optional)
"""

import os
import time
import hashlib
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple

# Optional pretty progress
try:
    from rich.progress import track
    RICH = True
except ImportError:
    RICH = False

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATA_DIR = Path("data/papers")
CATEGORIES = ["drones", "fixed_wing", "helicopters", "rockets", "satellites", "gliders"]

# Same URLs you already curated
REAL_SOURCES: Dict[str, List[Tuple[str, str]]] = {
    "drones": [
        ("https://ntrs.nasa.gov/api/citations/20110015820/downloads/20110015820.pdf",
         "NASA_Quadrocopter_Control_Design.pdf"),
        ("https://ntrs.nasa.gov/api/citations/20180001326/downloads/20180001326.pdf",
         "NASA_High_Fidelity_Multirotor_Aerodynamics.pdf"),
        ("https://ntrs.nasa.gov/api/citations/20220004968/downloads/1548_Malpica%20_Withrow_041422.pdf",
         "NASA_Multicopter_Handling_Qualities.pdf")
    ],
    "fixed_wing": [
        ("https://ocw.mit.edu/courses/16-885j-aircraft-systems-engineering-fall-2004/10ad50aedfa52b48e527fbec49da636e_aero_primer.pdf",
         "MIT_Aerodynamics_Primer.pdf"),
        ("https://ocw.mit.edu/courses/16-01-unified-engineering-i-ii-iii-iv-fall-2005-spring-2006/880678ca3fb1307aa73de7a6d073d730_spring_06_l1.pdf",
         "MIT_Flight_Power_Relations.pdf"),
        ("https://ocw.mit.edu/courses/16-001-unified-engineering-materials-and-structures-fall-2021/mit16_001_f21_lec_driver_art.pdf",
         "MIT_Breguet_Range_Equation.pdf")
    ],
    "rockets": [
        ("https://ntrs.nasa.gov/api/citations/19750009792/downloads/19750009792.pdf",
         "NASA_Sounding_Rocket_Handbook.pdf"),
        ("https://ntrs.nasa.gov/api/citations/19680016252/downloads/19680016252.pdf",
         "NASA_Rocket_Dynamic_Stability.pdf"),
        ("https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/pages/lecture-notes/MIT16_07F09_Lec14.pdf",
         "MIT_Rocket_Equation_Dynamics.pdf")
    ],
    "satellites": [
        ("https://ntrs.nasa.gov/api/citations/20210000201/downloads/TP-20210000201.pdf",
         "NASA_CubeSat_Technology_State_of_Art.pdf"),
        ("https://ntrs.nasa.gov/api/citations/20240016467/downloads/2025_IEEE_Aerospace_R5_avionics_final.pdf",
         "NASA_Avionics_Design_Architecture.pdf")
    ],
    "helicopters": [
        ("https://ntrs.nasa.gov/api/citations/20205004075/downloads/1428_Withrow_070720.pdf",
         "NASA_Multirotor_Configuration_Trades.pdf")
    ],
    "gliders": [
        ("https://ntrs.nasa.gov/api/citations/20160003578/downloads/20160003578.pdf",
         "NASA_Glider_Flight_Testing.pdf")
    ]
}

# ------------------------------------------------------------------
# HTTP Session with retry & polite headers
# ------------------------------------------------------------------
SESSION = requests.Session()
retry = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
SESSION.mount("https://", HTTPAdapter(max_retries=retry))
SESSION.headers.update({
    "User-Agent": "AerospaceDesignBot/1.0 (academic use, polite 2-s delay)"
})

# ------------------------------------------------------------------
# Download helper
# ------------------------------------------------------------------
def download_file(url: str, dest: Path, min_kb: int = 20) -> str:
    """Download with progress bar, return SHA-256 hex."""
    resp = SESSION.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    sha = hashlib.sha256()
    downloaded = 0
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            fh.write(chunk)
            sha.update(chunk)
            downloaded += len(chunk)
    if downloaded < min_kb * 1024:
        dest.unlink()
        raise ValueError(f"File too small ({downloaded} bytes)")
    return sha.hexdigest()

# ------------------------------------------------------------------
# Main downloader
# ------------------------------------------------------------------
def download_all() -> None:
    print("🚀 Starting real-data download...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(cat, url, name) for cat, pairs in REAL_SOURCES.items() for url, name in pairs]
    if not tasks:
        print("⚠️  No URLs configured")
        return

    iterator = track(tasks, description="Downloading") if RICH else tasks
    for cat, url, filename in iterator:
        folder = DATA_DIR / cat
        folder.mkdir(exist_ok=True)
        dest = folder / filename

        if dest.exists() and dest.stat().st_size > 0:
            print(f"⏭️  {cat}/{filename}  (exists)")
            continue

        try:
            checksum = download_file(url, dest)
            print(f"✅ {cat}/{filename}  (sha256:{checksum[:7]})")
            time.sleep(2)          # NTRS polite pause
        except Exception as e:
            print(f"❌ {cat}/{filename}  → {e}")

    print("\n🎉 Download complete.")

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    download_all()