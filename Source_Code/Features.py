import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv("/Users/riachaudhry/THSTI_PREDICTIVE_MODEL/Data.csv")


numeric_cols = ["Age", "Height_cm", "Weight_kg", "Alcohol_drinks_per_week", 
                "Caffeine_intake_coffee", "Caffeine_intake_tea", 
                "Caffeine_intake_soda", "Water_intake_liters", 
                "First_headache_age", "Headache_severity_avg", 
                "Headache_severity_max"]
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(df[numeric_cols])
scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_cols)

categorical_cols = ["Sex", "Headache_pattern", "Trigger_event", "Pain_location",
                    "Pain_type", "Associated_symptoms", "Relief_method", "Trigger_type"]
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_array = ohe.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(categorical_cols))


le_features = ["Smoke", "Drug_use", "Physical_activity", "Multiple_headache_types",
               "Headache_pattern", "Headache_worsening", "Sleep_interruption",
               "Sleep_relief", "Daily_activity_interference", "Pain_onset",
               "Aura_experience", "Aura_type", "Trigger_identified", "Monthly_headache_count"]
le_encoded = {}
for col in le_features:
    le = LabelEncoder()
    le_encoded[col] = le.fit_transform(df[col])

le_df = pd.DataFrame(le_encoded)


Recent_headache_date = pd.to_datetime(df["Recent_headache_date"], errors="coerce", dayfirst=True)
today = pd.to_datetime("01-01-2027", dayfirst=True)
days_since_headache = (today - Recent_headache_date).dt.days.fillna(0)


def parse_time_range(time_range_str):
    try:
        start_str, end_str = [x.strip().lower() for x in time_range_str.split('-')]
        start = pd.to_datetime(start_str, format="%I %p")
        end = pd.to_datetime(end_str, format="%I %p")
        if end <= start:
            end += pd.Timedelta(days=1)
        return (end - start).total_seconds() / 3600
    except:
        return np.nan

df["Sleep_pattern_weekdays"] = df["Sleep_pattern_weekdays"].astype(str).str.strip().str.replace("–", "-", regex=False)
df["Sleep_pattern_weekends"] = df["Sleep_pattern_weekends"].astype(str).str.strip().str.replace("–", "-", regex=False)

sleep_weekdays_duration = df["Sleep_pattern_weekdays"].apply(parse_time_range)
sleep_weekends_duration = df["Sleep_pattern_weekends"].apply(parse_time_range)


X_final = pd.concat([
    scaled_numeric_df.reset_index(drop=True),
    encoded_df.reset_index(drop=True),
    le_df.reset_index(drop=True),
    pd.DataFrame({
        'days_since_headache': days_since_headache,
        'sleep_pattern_weekdays': sleep_weekdays_duration,
        'sleep_pattern_weekends': sleep_weekends_duration
    }).reset_index(drop=True)
], axis=1)


le_target = LabelEncoder()
y = le_target.fit_transform(df["Headache_type_label"])

# information analysis to find the most important features 
mi_scores = mutual_info_classif(X_final.fillna(0), y, discrete_features='auto')

mi_df = pd.DataFrame({
    'Feature': X_final.columns,
    'Mutual_Info_Score': mi_scores
}).sort_values(by='Mutual_Info_Score', ascending=False)

print(mi_df)


import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))
plt.bar(mi_df['Feature'][:20], mi_df['Mutual_Info_Score'][:20])
plt.xticks(rotation=90)
plt.title("Top 20 Feature Importances (Mutual Information)")
plt.ylabel("Mutual Info Score")
plt.tight_layout()
plt.show()