import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier

def main():
    # 1. Initialize environment and database connections
    load_dotenv()
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise ValueError("Database connection string missing from .env file.")
        
    print("Connecting to Supabase Cloud Instance...")
    engine = create_engine(db_url)
    
    # 2. Extract raw structured tables from Supabase
    print("Extracting relational tables from PostgreSQL tables...")
    df_accounts = pd.read_sql("SELECT * FROM crm_b2b_accounts", engine)
    df_vehicles = pd.read_sql("SELECT * FROM crm_fleet_vehicles", engine)
    df_telemetry = pd.read_sql("SELECT * FROM vehicle_sensor_telemetry", engine)
    
    # 3. Pull ground truth labels from your local dataset folder
    print("Loading historical ground truth failure tags...")
    tte_path = os.path.join("data", "train_tte.csv")
    df_labels = pd.read_csv(tte_path)[['vehicle_id', 'in_study_repair']]
    df_labels.rename(columns={'in_study_repair': 'true_failure_state'}, inplace=True)
    
    # 4. Feature Engineering & Applying Descriptive Semantic Labels
    print("Transforming abstract properties into descriptive variables...")
    df_features = df_telemetry.groupby('vehicle_id').agg({
        'operational_hour_timestamp': 'max',
        'exhaust_gas_temperature_c': ['mean', 'max'],
        'dpf_differential_pressure_kpa': ['mean', 'max'],
        'engine_soot_mass_index': ['mean', 'max']
    })
    
    # Flatten MultiIndex columns into intuitive descriptive names
    df_features.columns = [
        'total_operating_hours',
        'avg_exhaust_temperature_c', 'peak_exhaust_temperature_c',
        'avg_dpf_differential_pressure_kpa', 'peak_dpf_differential_pressure_kpa',
        'avg_engine_soot_mass_index', 'peak_engine_soot_mass_index'
    ]
    df_features.reset_index(inplace=True)
    
    # 5. Build Unified Analytics Dataset
    df_master = pd.merge(df_features, df_vehicles, on='vehicle_id', how='inner')
    df_master = pd.merge(df_master, df_labels, on='vehicle_id', how='inner')
    df_master = pd.merge(df_master, df_accounts, on='account_id', how='inner')
    
    # 6. Execute Local Machine Learning Inference Pipeline
    print("Executing predictive maintenance scoring model...")
    X = pd.get_dummies(df_master.drop(columns=['vehicle_id', 'account_id', 'company_name', 'operating_prefecture', 'true_failure_state']))
    y = df_master['true_failure_state'].astype(int)
    
    # Train robust classification forest to generate continuous probability profiles
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    # Assign calculated probability outputs back into descriptive column features
    probabilities = model.predict_proba(X)[:, 1]
    df_master['downtime_risk_percentage'] = np.round(probabilities * 100, 2)
    
    # Establish operational action priorities
    df_master['maintenance_priority_action'] = np.where(df_master['downtime_risk_percentage'] >= 75.0, 'CRITICAL DISPATCH',
                                                np.where(df_master['downtime_risk_percentage'] >= 40.0, 'PREVENTATIVE MONITOR', 'STABLE OPERATION'))
    
    # 7. Add Simulated Financial and Environmental KPIs for Tsugi no Hi
    # Financial breakdown downtime cost penalty vs a cheap preventative flush solution
    df_master['expected_breakdown_cost_jpy'] = np.round(probabilities * 500000, 0)
    df_master['preventative_maintenance_cost_jpy'] = 30000
    
    # Environmental circularity metrics mapping
    df_master['recycled_materials_saved_kg'] = np.where(df_master['true_failure_state'] == 1, 45.0, 0.0)
    df_master['co2_emissions_prevented_kg'] = np.where(df_master['true_failure_state'] == 1, 350.0, 0.0)
    
    # 8. Output Flat Clean Extraction for Tableau Public Connection
    output_file_path = os.path.join("data", "tableau_fleet_analytics.csv")
    df_master.to_csv(output_file_path, index=False)
    print(f"Pipeline executed successfully. Target dashboard file generated at: {output_file_path}")

if __name__ == "__main__":
    main()