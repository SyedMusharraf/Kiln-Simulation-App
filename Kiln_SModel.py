import pandas as pd
import numpy as np
import joblib 


#MACHINe learing imports 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import  StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



#visuualization import
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("sci-kit/kiln_simulation_results_copy copy.xlsx")
# df.drop(columns=["CaO_input",'SiO2_input', 'Al2O3_input', 'Fe2O3_input'],inplace=True)
target = ['T_solid_output','T_gas_output']

features = [col for col in df.columns if col not in target]
# X = df[features]
# print(features)
# exit()

joblib.dump(features, "kiln_sm_features.pkl")
X = df[features]
            
y = df[target]
            

X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

            # print(X_train)
            # exit()

def train_model(X_train, y_train):

    model = RandomForestRegressor(
                #n_estimators=150,        # Reduced from GridSearchCV options
                # max_depth=25,           # Set a reasonable max depth
                # min_samples_split=5,    # Moderate split size
                # max_features='sqrt',    # Efficient feature sampling
    random_state=42,        # Reproducibility
    n_jobs=-1               # Use all CPU cores
    )
    param_grid = {'n_estimators': [100,150,200],
    'max_depth': [8,10,12],
    'min_samples_split': [2,4] 
    }
            #gridsearchCv 
    grid = GridSearchCV(
    model,param_grid,
    cv=5,
    n_jobs=-1
    )

    grid.fit(X_train, y_train)

                

    return grid.best_estimator_, grid.cv_results_

best_model, cv_results= train_model(X_train,y_train)
# print(best_model)
cv_df = pd.DataFrame(cv_results)
            # cv_df.to_excel(f"Model_{t}.xlsx")

# joblib.dump(best_model,'Kiln_sm_stream.pkl')
# joblib.dump(scaler,'kiln_sm_scaler.pkl')
            

y_pred = best_model.predict(X_test)

        # Get feature importances from the model
        #this return the most variable containing values 
feature_importances = best_model.feature_importances_

        #get the columns 
feature_names = X.columns

        # Create a DataFrame for better handling
        #creating DF and sort it by most value 
importance_df = pd.DataFrame({
'Feature': feature_names,
'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

        # Plot feature importances
        # creating the graph and bar to look 
        
        #palette is for color 
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title(f'Feature Importance for {target}')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
# plt.savefig(f'Feature_{target}.png')

r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nthe metrics for: {target}")
print(f"R2 value : {r2:.4f}")
print(f"Mean_squared_error: {mse:.4f}")
print(f"Root_Mean_sqrt_Error:{rmse:.4f}")
print(f"Mean_absolute_Error:{mae:.4f}")

plt.figure(figsize=[8,8])
plt.scatter(y_test,y_pred,alpha=0.6,color="purple")
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"k--")
plt.xlabel("actual values")
plt.ylabel("prediction values ")
plt.title(f"{target} for actual v/s prediction values")
plt.grid(True)
# plt.savefig(f'Single_Model_{target}.png')

exit()

y_pred = np.array(y_pred)
for i , col in enumerate(target):
    actual = y_test.iloc[:,i]
    predicted = y_pred[:,i]

    r2 = r2_score(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)

    print(f"\nthe matrics of : {col}:")
    print(f"R² Score: {r2:.4f}")  #  -->0.98
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")



    plt.figure(figsize=(8,8))
    plt.scatter(actual,predicted,color = 'purple',alpha=0.6,)
    plt.plot([actual.min(),actual.max()],[actual.min(),actual.max()],'k--',lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel("Predicted Values")
    plt.title(f"Acutal Vs Predicted for {col}")
    plt.grid(True)
                # plt.show()
    plt.savefig(f'Single_Model_{col}.png')











