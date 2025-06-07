# 📌 1. Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# 📌 2. Load all datasets (they must be in the same folder as this script)
# Each dataset includes HIV-related indicators for different regions and years
deaths = pd.read_excel("Aids related deaths by region.xlsx")
art = pd.read_excel("% of people living with HIV and receiving ART.xlsx")
pmct = pd.read_excel("Coverage of pregnant women receiving ARV for PMTCT.xlsx")
mortality = pd.read_excel("mortality rate by region.xlsx")
incidence_prev = pd.read_excel("Incidence prevalence ratio by region.xlsx")

# 📌 3. Define a function to clean and reshape the data
# Converts wide-format (years as columns) to long-format and extracts only numeric values
def clean_and_melt_safe(df, value_name):
    df_long = df.melt(id_vars=["Region"], var_name="Year", value_name=value_name)
    df_long[value_name] = df_long[value_name].astype(str).str.replace(",", "")
    df_long[value_name] = df_long[value_name].str.extract(r"(\d+\.?\d*)")
    df_long.dropna(subset=[value_name], inplace=True)
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce')
    df_long.dropna(subset=[value_name], inplace=True)
    df_long["Year"] = df_long["Year"].astype(int)
    return df_long

# 📌 4. Clean and reshape all five datasets
deaths_clean = clean_and_melt_safe(deaths, "Deaths")
art_clean = clean_and_melt_safe(art, "ART_Coverage")
pmct_clean = clean_and_melt_safe(pmct, "PMTCT_Coverage")
mortality_clean = clean_and_melt_safe(mortality, "Mortality_Rate")
incidence_clean = clean_and_melt_safe(incidence_prev, "Incidence_Prevalence_Ratio")

# 📌 5. Merge all datasets on Region and Year
# This creates one complete dataset with all indicators side by side
merged = deaths_clean.merge(art_clean, on=["Region", "Year"], how="inner")
merged = merged.merge(pmct_clean, on=["Region", "Year"], how="inner")
merged = merged.merge(mortality_clean, on=["Region", "Year"], how="inner")
merged = merged.merge(incidence_clean, on=["Region", "Year"], how="inner")

# 📌 6. Export the merged dataset for backup or Tableau use
merged.to_excel("Merged_HIV_Dataset.xlsx", index=False)

# 📌 7. Prepare the data for machine learning
# Convert the 'Region' column to numeric form (one-hot encoding)
merged_encoded = pd.get_dummies(merged, columns=["Region"], drop_first=True)

# Separate input features (X) and the target variable (y = Deaths)
X = merged_encoded.drop(columns=["Deaths"])
y = merged_encoded["Deaths"]

# 📌 8. Split the dataset into training and testing sets (80/20)
# Training data is used to train the model, testing data checks its accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 9. Train the Random Forest Regressor model
# This model uses multiple decision trees to predict the number of deaths
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 10. Test the model by predicting on unseen (test) data
# Then evaluate how accurate the predictions are using R² and RMSE
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print performance results
print("R^2 Score:", round(r2, 4))  # How well predictions match actual values (1 = perfect)
print("RMSE:", round(rmse, 2))    # Average error in predictions

# 📌 11. Plot Actual vs Predicted deaths
# This graph helps visualize how close the predictions are to the real values
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # perfect prediction line
plt.xlabel("Actual Deaths")
plt.ylabel("Predicted Deaths")
plt.title("Actual vs Predicted AIDS-Related Deaths")
plt.grid(True)
plt.tight_layout()
plt.show()

# 📌 12. Feature importance analysis
# Shows which features were most important in predicting deaths
importances = model.feature_importances_
features = X.columns
feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=True)

# 📌 12. Plot: Feature Importance (cleaned and sorted)
importances = model.feature_importances_
features = X.columns

# Create DataFrame and sort from most to least important
feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Optional: Clean one-hot encoded region names for readability
feat_df["Feature"] = feat_df["Feature"].str.replace("Region_", "", regex=False)

# Plot the importance scores
plt.figure(figsize=(10, 6))
plt.barh(feat_df["Feature"], feat_df["Importance"], color="skyblue")
plt.xlabel("Importance Score")
plt.title("Feature Importance in Predicting AIDS-Related Deaths")
plt.gca().invert_yaxis()  # Put most important feature at the top
plt.tight_layout()
plt.show()


# 📌 13. Export predictions to Excel for visualization in Tableau
# Add actual and predicted deaths back into the test data
output_df = X_test.copy()
output_df["Actual_Deaths"] = y_test.values
output_df["Predicted_Deaths"] = y_pred

# If 'Year' column is missing (due to encoding), reattach it
if "Year" not in output_df.columns:
    output_df["Year"] = merged_encoded.loc[X_test.index, "Year"]
    
# Reattach original Region name if needed
if "Region" not in output_df.columns and "Region" in merged.columns:
    output_df["Region"] = merged.loc[X_test.index, "Region"]

# Sort the prediction results by Region and Year for easy viewing in Excel/Tableau
sort_cols = ["Region", "Year"] if "Region" in output_df.columns else ["Year", "Actual_Deaths"]
output_df_sorted = output_df.sort_values(by=sort_cols)

# Export prediction results
output_df_sorted.to_excel("RF_Predictions_HIV.xlsx", index=False)

# 📌 14. Re-export the merged dataset in sorted order
# Sorting makes it easier to understand and use in visual dashboards
merged_sorted = merged.sort_values(by=["Region", "Year"])
merged_sorted.to_excel("Merged_HIV_Dataset.xlsx", index=False)
