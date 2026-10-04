"""
generate_data.py
----------------
Creates a simulated network traffic dataset and saves it to:
    data/traffic_data.csv

How the data is made (simple idea):
1. For each row we first pick a traffic condition: Normal, Medium, High
   or Congestion (with realistic proportions - normal traffic is the most common).
2. Then we generate the 7 traffic measurements for that condition using
   random numbers. The ranges of neighbouring classes OVERLAP on purpose,
   so the dataset is not "too perfect" and the model must really learn.
3. Values are linked in a realistic way:
       utilization  ->  bandwidth  ->  bytes/sec  ->  packets/sec
   (more utilization means more bandwidth, more bytes, more packets).

Why there is no data leakage:
- The label (congestion_level) is NOT calculated from a fixed formula
  using the feature columns. It is chosen first, and the features are
  random samples that only tend to look like that class.
- No helper/hidden columns are saved in the CSV.

Run:  python generate_data.py
"""

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Settings (you can change these)
# ------------------------------------------------------------------
NUM_ROWS = 5000          # total rows in the dataset
RANDOM_SEED = 42         # same seed -> same dataset every time
LINK_CAPACITY_MBPS = 100 # assumed network link speed (100 Mbps)

# Where to save the file (works no matter where you run the command from)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "traffic_data.csv")

# Class names and how often each one appears
CLASSES = ["Normal", "Medium", "High", "Congestion"]
CLASS_PROBABILITIES = [0.40, 0.30, 0.20, 0.10]

# For each class: (mean, standard deviation) of the main measurements
PROFILES = {
    #               utilization %   latency ms    loss % (avg)  connections
    "Normal":     {"util": (25, 10), "lat": (15, 6),   "loss": 0.15, "conn": (120, 50)},
    "Medium":     {"util": (50, 9),  "lat": (35, 12),  "loss": 0.60, "conn": (300, 80)},
    "High":       {"util": (73, 8),  "lat": (75, 25),  "loss": 1.80, "conn": (550, 120)},
    "Congestion": {"util": (91, 5),  "lat": (160, 60), "loss": 5.50, "conn": (800, 150)},
}


def make_row(label, rng):
    """Generate one row of traffic measurements for the given class."""
    p = PROFILES[label]

    # 1) Network utilization (% of link capacity used), kept between 1 and 100
    utilization = np.clip(rng.normal(*p["util"]), 1, 100)

    # 2) Bandwidth usage in Mbps = utilization share of the link (+ small noise)
    bandwidth = LINK_CAPACITY_MBPS * utilization / 100 * rng.normal(1.0, 0.05)
    bandwidth = float(np.clip(bandwidth, 0.1, LINK_CAPACITY_MBPS))  # cannot exceed the link

    # 3) Bytes per second = bandwidth (Mbps) converted to bytes (+ small noise)
    bytes_per_sec = bandwidth * 1_000_000 / 8 * rng.normal(1.0, 0.05)
    bytes_per_sec = min(bytes_per_sec, LINK_CAPACITY_MBPS * 1_000_000 / 8)  # link limit

    # 4) Packets per second = bytes / average packet size (200 - 1400 bytes)
    avg_packet_size = np.clip(rng.normal(700, 150), 200, 1400)
    packets_per_sec = bytes_per_sec / avg_packet_size

    # 5) Latency in milliseconds (never below 1 ms, capped at 500 ms)
    latency = np.clip(rng.normal(*p["lat"]), 1, 500)

    # 6) Packet loss in % (random, mostly small, capped at 30 %)
    packet_loss = np.clip(rng.exponential(p["loss"]), 0, 30)

    # 7) Number of active connections (whole number, at least 5)
    connections = int(np.clip(rng.normal(*p["conn"]), 5, 1500))

    return {
        "bandwidth_usage": round(bandwidth, 2),            # Mbps
        "packets_per_second": int(round(packets_per_sec)), # packets/s
        "bytes_per_second": int(round(bytes_per_sec)),     # bytes/s
        "latency": round(latency, 2),                      # ms
        "packet_loss": round(packet_loss, 2),              # %
        "network_utilization": round(utilization, 2),      # %
        "connection_count": connections,                   # count
        "congestion_level": label,                         # target column
    }


def generate_dataset():
    """Build the full dataset and return it as a Pandas DataFrame."""
    rng = np.random.default_rng(RANDOM_SEED)

    # Pick a class for every row using the probabilities above
    labels = rng.choice(CLASSES, size=NUM_ROWS, p=CLASS_PROBABILITIES)

    rows = [make_row(label, rng) for label in labels]
    df = pd.DataFrame(rows)

    # Remove accidental duplicate rows (very unlikely, but safe)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def main():
    df = generate_dataset()

    # Make sure the data/ folder exists, then save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Dataset created successfully!")
    print(f"Saved to : {OUTPUT_FILE}")
    print(f"Rows     : {len(df)}")
    print(f"Columns  : {list(df.columns)}")
    print("\nClass distribution:")
    print(df["congestion_level"].value_counts().reindex(CLASSES))
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()