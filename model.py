import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import catboost as cb
from sklearn.pipeline import make_pipeline
import pickle

df = pd.read_csv('data/final.csv')
target = 'rating'
y = df[target]
X = df.drop(target, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
     (['jer_sent'], StandardScaler()),
     (['ela_sent'], [SimpleImputer(),StandardScaler()]),
     (['kra_sent'], [SimpleImputer(),StandardScaler()]),
     (['geo_sent'],  [SimpleImputer(),StandardScaler()]),
     (['jer_lines'], StandardScaler()),
     (['ela_lines'], StandardScaler()),
     (['kra_lines'], StandardScaler()),
     (['geo_lines'],  StandardScaler()),
     ('location', MultiLabelBinarizer()),
     # ('frank', None),
     # ('newman', None),
     # ('peterman', None),
     # ('puddy', None),
     # ('Writers',  LabelBinarizer()),
     # ('Director',  LabelBinarizer()),
     ], df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)





# import xgboost as xgb
# from sklearn.model_selection import cross_val_score
#
#
# xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.5, max_depth = 7, alpha = 1, n_estimators = 200)
#
#
# xg_reg.fit(Z_train,y_train)
# xg_reg.score(Z_train, y_train)
# xg_reg.score(Z_test, y_test)





model = cb.CatBoostRegressor(
    iterations=10,
    learning_rate=0.5,
)

model.fit(
    Z_train, y_train,
    eval_set=(Z_test, y_test),
    verbose=False,
    plot=False,
)



model.score(Z_train, y_train)
model.score(Z_test, y_test)







pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open('pipe.pkl', 'wb'))

# from sklearn.metrics import mean_squared_error
#
# # A:
# mean_squared_error(y_test, d)
# mean_squared_error(y_test, model.predict(Z_test))
#
#
# model.get_feature_importance(data=None,
#                        prettified=False,
#                        thread_count=-1,
#                        verbose=False)
#

## Below is sample predicting
X_train.sample().to_dict(orient='list')

new = pd.DataFrame({
    'id': ['S05E15'],
    'jer_sent': [0.06],
    'ela_sent': [0.06],
    'kra_sent': [0.062],
    'geo_sent': [0.061],
    'jer_lines': [100],
    'ela_lines': [45],
    'kra_lines': [36],
    'geo_lines': [83],
    'location': ["['The Improv', 'Jerry’s Apartment', 'Monk’s Café', 'Pendant Publishing']"]
})

type(pipe.predict(new)[0])

prediction = float(pipe.predict(new)[0])
