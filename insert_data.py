import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.getenv("SUPABASE_DB_URL"))

# Column mapping for pipeline compatibility
column_mapping = {
    "time_step": "operational_hour_timestamp",
    "459_0": "exhaust_gas_temperature_c",
    "291_0": "dpf_differential_pressure_kpa",
    "397_0": "engine_soot_mass_index",
}

# Read CSVs
df_readouts = pd.read_csv("data/train_operational_readouts.csv")
df_specs = pd.read_csv("data/train_specifications.csv")
df_tte = pd.read_csv("data/train_tte.csv")

print(f"Readouts: {len(df_readouts)} rows, {len(df_readouts.columns)} cols")
print(f"Specs:    {len(df_specs)} rows, {len(df_specs.columns)} cols")
print(f"TTE:      {len(df_tte)} rows, {len(df_tte.columns)} cols")

# Sample 5 rows per vehicle to stay under 500MB free tier
df_readouts = df_readouts.groupby("vehicle_id", group_keys=False).sample(n=5, random_state=42)
df_readouts = df_readouts.rename(columns=column_mapping)
print(f"Sampled:  {len(df_readouts)} rows")

unique_vehicles = sorted(df_readouts["vehicle_id"].unique())

with engine.connect() as conn:
    # 1. Insert accounts (1 account per vehicle)
    print("\nInserting accounts...")
    conn.execute(text("TRUNCATE crm_b2b_accounts CASCADE"))
    df_accounts = pd.DataFrame({
        "account_id": unique_vehicles,
        "company_name": [f"Company_{v}" for v in unique_vehicles],
        "operating_prefecture": ["Unknown"] * len(unique_vehicles)
    })
    df_accounts.to_sql("crm_b2b_accounts", conn, if_exists="append", index=False, method="multi", chunksize=100)

    # 2. Insert vehicles (1-to-1: vehicle_id = account_id)
    print("Inserting vehicles...")
    conn.execute(text("TRUNCATE crm_fleet_vehicles CASCADE"))
    df_vehicles = pd.DataFrame({
        "vehicle_id": unique_vehicles,
        "account_id": unique_vehicles
    })
    df_vehicles.to_sql("crm_fleet_vehicles", conn, if_exists="append", index=False, method="multi", chunksize=100)

    # 3. Drop and recreate telemetry table with correct column names
    print("Recreating telemetry table with mapped column names...")
    conn.execute(text("DROP TABLE IF EXISTS vehicle_sensor_telemetry CASCADE"))
    conn.execute(text(f'''
        CREATE TABLE vehicle_sensor_telemetry (
            vehicle_id INTEGER,
            operational_hour_timestamp NUMERIC,
            exhaust_gas_temperature_c NUMERIC,
            dpf_differential_pressure_kpa NUMERIC,
            engine_soot_mass_index NUMERIC
        )
    '''))

    # Add remaining anonymous sensor columns
    sensor_cols = [c for c in df_readouts.columns if c not in ("vehicle_id", "operational_hour_timestamp", "exhaust_gas_temperature_c", "dpf_differential_pressure_kpa", "engine_soot_mass_index")]
    for col in sensor_cols:
        conn.execute(text(f'ALTER TABLE vehicle_sensor_telemetry ADD COLUMN "{col}" NUMERIC'))
    conn.commit()

    # 4. Insert telemetry data
    print("Inserting telemetry data...")
    df_readouts.to_sql("vehicle_sensor_telemetry", conn, if_exists="append", index=False, method="multi", chunksize=500)

print("\nDone. Tables populated.")
