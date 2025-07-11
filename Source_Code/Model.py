import pandas as pd
import sklearn 
from sklearn.preprocessing import OneHotEncoder
from datetime import timedelta
from datetime import time 
import numpy as np 
from sklearn.utils import shuffle 
from sklearn import linear_model, preprocessing 
import datetime as datetime 
from sklearn import model_selection 
from sklearn import svm
from sklearn import metrics 
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

# reading the data: 
df = pd.read_csv("/Users/riachaudhry/THSTI_PREDICTIVE_MODEL/Data.csv")


# using standard scaler to scale the numeric values 

from sklearn.preprocessing import StandardScaler
data_numeric = df[[  "Caffeine_intake_tea", 
                   "Caffeine_intake_soda", 
                   "First_headache_age", "Headache_severity_avg", 
               ]]
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(data_numeric)

# used one hot encoder for the non ordinal, non-numeric  values 
categorical_cols = ["Headache_pattern","Pain_location","Pain_type","Relief_method","Trigger_type", "Associated_symptoms" ]
encoder = OneHotEncoder( handle_unknown='ignore', sparse_output=False)
encoded_data = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

# used label encoder for the ordinal non numerical values 
numerical = preprocessing.LabelEncoder()
Headache_type_label = numerical.fit_transform(list(df["Headache_type_label"]))

rug_use = numerical.fit_transform(list(df["Drug_use"]))

Headache_pattern = numerical.fit_transform(list(df["Headache_pattern"]))
Pain_onset = numerical.fit_transform(list(df["Pain_onset"]))
Aura_experience = numerical.fit_transform(list(df["Aura_experience"]))
Aura_type = numerical.fit_transform(list(df["Aura_type"]))
Trigger_identified = numerical.fit_transform(list(df["Trigger_identified"]))
Monthly_headache_count = numerical.fit_transform(list(df["Monthly_headache_count"]))

# manually encoded the time and time ranges 
list_recent_headache = list(df["Recent_headache_date"])
Recent_headache_date = pd.to_datetime(list_recent_headache, errors= "coerce", dayfirst= True)
print(Recent_headache_date)

'''
today = pd.to_datetime(datetime.datetime.today().date(), dayfirst= True)
'''
today = pd.to_datetime("01-01-2027", dayfirst= True )
def days_since_headaches(x):
    for i in range(len(x)):     
             y = (today - x)
    return y
days_since_headache = (days_since_headaches(Recent_headache_date))

days_since_headache = pd.Series((today - Recent_headache_date).days)

def parse_time_range(time_range_str):

    try:
      
      start_str, end_str = [x.strip().lower() for x in time_range_str.split('-')]
      start = pd.to_datetime(start_str, format = "%I %p")
      end = pd.to_datetime(end_str, format = "%I %p")
      if end <= start:
            end += pd.Timedelta(days=1)

      return start, end
    except:
      return pd.NaT, pd.NaT
    
df["Sleep_pattern_weekdays"] = df["Sleep_pattern_weekdays"].astype(str).str.strip()
df["Sleep_pattern_weekdays"] = df["Sleep_pattern_weekdays"].str.replace("–", "-", regex=False)
list_sleep_weekdays = [] 
for i in range(len(df["Sleep_pattern_weekdays"])):
    time_range_str = df.loc[i, "Sleep_pattern_weekdays"]
    start, end = parse_time_range(time_range_str)
    if pd.notna(start) and pd.notna(end):
        duration = (end - start).total_seconds() / 3600
        list_sleep_weekdays.append(duration)
    else:
        print(f"  Skipped row {i} due to missing values.")
    

df["Sleep_pattern_weekends"] = df["Sleep_pattern_weekends"].astype(str).str.strip()
df["Sleep_pattern_weekends"] = df["Sleep_pattern_weekends"].str.replace("–", "-", regex=False)
list_sleep_weekends = []
for i in range(len(df["Sleep_pattern_weekends"])):

    time_range_str = df.loc[i, "Sleep_pattern_weekends"]
    start, end = parse_time_range(time_range_str)
    if pd.notna(start) and pd.notna(end): 
        duration = (end - start).total_seconds() / 3600
        list_sleep_weekends.append(duration)
    else:
        duration = None 
   
# used xgboost for the model 
data2 = encoded_df # the data that uses onehotencoder 
data3 = pd.DataFrame ({


    "Headache_pattern": Headache_pattern,
   
    "Pain_onset": Pain_onset,

    "Aura_experience": Aura_experience,
    "Aura_type": Aura_type,

    "Trigger_identified": Trigger_identified,

    "Monthly_headache_count": Monthly_headache_count,
    "days_since_headache": days_since_headache,
    "list_sleep_weekdays": list_sleep_weekdays,
    "list_sleep_weekends": list_sleep_weekends
})


# data that was manually encoded 
data1 = pd.DataFrame(scaled_numeric)
final_data = pd.concat([data1,data3,data2], axis = 1 )

pred = Headache_type_label

x = np.array(final_data)
y = np.array(pred)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

#it takes one parameter being the amount of neighbors, so the value of k 
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(x_train, y_train)
pred = xgb.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)
# analysing how the model behaves 

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))



# visualizing the results and analysis of the model

accuracy = accuracy_score(y_test, pred)

# Create bar plot
plt.figure(figsize=(4, 6))
plt.bar(['Accuracy'], [accuracy], color='cornflowerblue')
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.text(0, accuracy + 0.03, f"{accuracy:.2%}", ha='center', fontsize=12)
plt.tight_layout()
plt.show()

class_labels = ['Migraine', 'Tension', 'Cluster', 'Other']
report_dict = classification_report(y_test, pred, target_names=class_labels, output_dict=True)

# Convert to DataFrame
df_report = pd.DataFrame(report_dict).transpose()
df_report = df_report.loc[class_labels, ['precision', 'recall', 'f1-score']]

# Plot
df_report.plot(kind='bar', figsize=(10, 6))
plt.title('Classification Report Metrics per Class')
plt.ylabel('Score')
plt.ylim(0, 1.0)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


class_labels = ['Migraine', 'Tension', 'Cluster', 'Other']

cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


