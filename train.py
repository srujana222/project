import pandas as pd
df=pd.read_csv("github.csv")
df
df.isnull().sum()

df.info()

df.dtypes

df.shape

df["primary_language"] = df["primary_language"].fillna(df["primary_language"].mode()[0])

# Drop unnecessary columns
df = df.drop(["repo_name", "owner"], axis=1)
df.isnull().sum()

cat_col=df.select_dtypes(include="object").columns.tolist()
cat_col

num_col=df.select_dtypes(exclude="object").columns.tolist()
num_col

from sklearn.preprocessing import LabelEncoder
le_lang = LabelEncoder()
df["primary_language"] = le_lang.fit_transform(df["primary_language"])

df

x=df.drop("stars",axis=1)
x

y = df["stars"]

y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from xgboost import XGBRegressor

model=XGBRegressor()
model.fit(x_train,y_train)


y_pred_GBR=model.predict(x_test)
y_pred_GBR

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred_GBR)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_GBR)
print("mse",mse,"rmse",rmse,"r_sques",r2)

import joblib
joblib.dump(model,"git.pkl")
joblib.dump(le_lang, "encoder.pkl")
print("✅ Model & encoder saved successfully")
