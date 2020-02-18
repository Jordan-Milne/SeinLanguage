import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import catboost as cb
from sklearn.pipeline import make_pipeline
import pickle
import numpy as np

df = pd.read_csv('data/final3.csv')

df.describe()
# df = pd.read_csv('data/final.csv')
# df['rating'] = np.exp(df['rating'])


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


model = cb.CatBoostRegressor(
    iterations=1000,
    learning_rate=0.5,
    # depth=10,
    # random_seed=5,
    # early_stopping_rounds=200,
)

model.fit(
    Z_train, y_train,
    eval_set=(Z_test, y_test),
    verbose=False,
    plot=False,
)

# # GridSearch
# grid = {'learning_rate': [0.03, 0.1, 0.5],
#         'depth': [4, 6, 10],
#         'l2_leaf_reg': [1, 5, 9],
#         'early_stopping_rounds': [200]}
#
#
#
# grid_search_result = model.grid_search(grid,
#                                        X=Z_train,
#                                        y=y_train,
#                                        plot=False)


model.score(Z_train, y_train)
model.score(Z_test, y_test)
































# pipe = make_pipeline(mapper, model)
# pipe.fit(X_train, y_train)
# pipe.score(X_test, y_test)
# pickle.dump(pipe, open('pipe.pkl', 'wb'))

# from sklearn.metrics import mean_squared_error
# #
# # # A:
# mean_squared_error(np.log(y_test), np.log(model.predict(Z_test)))
# #0.1765933837242009
# #
# #
# # model.get_feature_importance(data=None,
# #                        prettified=False,
# #                        thread_count=-1,
# #                        verbose=False)
# #
#
# ## Below is sample predicting
# X_train.sample().to_dict(orient='list')
#
# new = pd.DataFrame({
#     'id': ['S07E15'],
#      'jer_sent': [0.119],
#      'ela_sent': [0.051],
#      'kra_sent': [0.1866],
#      'geo_sent': [0.1001],
#      'jer_lines': [27],
#      'ela_lines': [64],
#      'kra_lines': [4],
#      'geo_lines': [44],
#     'location': ["['The Improv', 'Jerry’s Apartment', 'Monk’s Café', 'Pendant Publishing']"]
# })
#
# type(pipe.predict(new)[0])
#
# prediction = np.log(float(pipe.predict(new)[0]))
# prediction
# prediction = round(np.log(float(pipe.predict(new)[0]),1))
