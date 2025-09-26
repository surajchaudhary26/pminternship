import pandas as pd
from pathlib import Path

# Paths
processed_dir = Path("ml_engine/data/processed")
matches_file = processed_dir / "matches.csv"

# Load matches
df = pd.read_csv(matches_file)

# Show info
print(f"✅ Found {len(df)} student IDs in {matches_file}")
print("\n🔹 Sample Student IDs you can test:")

print(df["student_id"].head(10).tolist())  # first 10
