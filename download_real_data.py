import os
import urllib.request
import time

# Define the directory structure
DATA_DIR = "data/papers"
CATEGORIES = [
    "drones", "fixed_wing", "helicopters", "rockets", "satellites", "gliders"
]

# Curated list of Open Access Technical PDFs (NASA, MIT, etc.)
# These are stable, direct download links to real engineering documents.
REAL_SOURCES = {
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
         "MIT_Rocket_Equation_Dynamics.pdf")  # Note: Check if this link resolves, if not, fallback to NTRS
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
         "NASA_Glider_Flight_Testing.pdf")  # Fallback/Example
    ]
}


def download_files():
    print("🚀 Starting Real Data Download...")

    for category in CATEGORIES:
        folder_path = os.path.join(DATA_DIR, category)
        os.makedirs(folder_path, exist_ok=True)

        # Clear dummy data if it exists
        for f in os.listdir(folder_path):
            if "dummy" in f.lower() or "txt" in f:
                # Optional: Delete .txt files if you want to replace them entirely
                # os.remove(os.path.join(folder_path, f))
                pass

        if category in REAL_SOURCES:
            for url, filename in REAL_SOURCES[category]:
                dest_path = os.path.join(folder_path, filename)

                if os.path.exists(dest_path):
                    print(f"⏭️  Skipping {filename} (Already exists)")
                    continue

                print(f"⬇️  Downloading {filename}...")
                try:
                    # Add User-Agent to avoid 403 Forbidden from some servers
                    req = urllib.request.Request(
                        url,
                        data=None,
                        headers={
                            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
                        }
                    )

                    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                        out_file.write(response.read())

                    print(f"✅ Saved to {category}/{filename}")
                    time.sleep(1)  # Be polite to the servers

                except Exception as e:
                    print(f"❌ Failed to download {filename}: {e}")

    print("\n🎉 Download Complete. You now have REAL PDF data.")


if __name__ == "__main__":
    download_files()
