import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import category_encoders as ce
from sklearn.metrics import f1_score
from pickle import dump

df = pd.read_csv('Job_Placement_Data.csv')

#Taking deep copy
df1 = df.copy(deep = True)

#No duplicates
df[df.duplicated(keep = 'last')]


def target_encode_split(dataframe):
  le = LabelEncoder()
  dataframe['status']= le.fit_transform(dataframe['status'])
  
  x = dataframe.drop(['status'], axis = 1)
  y = dataframe['status']
  trainx, testx, trainy, testy = train_test_split(x, y, test_size = 0.1, random_state = 69, stratify = y)

  df_list = [trainx, testx, trainy, testy]

  for i in df_list:
    i.reset_index(drop = True, inplace = True)
  
  return trainx, testx, trainy, testy, le

train_x, test_x, train_y, test_y, le = target_encode_split(df1)

#########################################BASE MODEL#########################################
# lin_model = LogisticRegression(random_state = 69)

# print(train_x.isnull().sum(), test_x.isnull().sum())

# train_x.fillna(method = 'ffill', inplace = True)
# test_x.fillna(method = 'bfill', inplace = True)
# print(train_x.isnull().sum(), test_x.isnull().sum())
# train_x.reset_index(drop = True)
# test_x.reset_index(drop = True)

# one = OneHotEncoder(sparse = False, handle_unknown = 'ignore')

# one.fit(train_x)

# train_x = pd.DataFrame(one.transform(train_x), columns = one.get_feature_names_out())

# test_x = pd.DataFrame(one.transform(test_x), columns = one.get_feature_names_out())

# lin_model.fit(train_x, train_y)

# pred = lin_model.predict(test_x)

# from sklearn.metrics import f1_score
# f1_score(test_y, pred)
####################################### BASE SCORE IS 0.82 ####################################################


def separate_dtypes(trx, tex):
  train_num = trx.select_dtypes(include = 'number')
  train_cat = trx.select_dtypes(include = 'object')
  train_cat = train_cat.reset_index(drop = True)

  test_num = tex.select_dtypes(include = 'number')
  test_cat = tex.select_dtypes(include = 'object')
  test_cat = test_cat.reset_index(drop = True)

  return train_num, train_cat, test_num, test_cat

train_num, train_cat, test_num, test_cat = separate_dtypes(train_x, test_x)

def impute(trcat, tecat, trnum, tenum):

  for column in trcat.columns:
      trcat[column].fillna(trcat[column].mode()[0], inplace=True)

  for column in tecat.columns:
      tecat[column].fillna(trcat[column].mode()[0], inplace=True)

  for i in trnum.columns:
    trnum[i].fillna(trnum[i].median(), inplace = True)

  for i in tenum.columns:
    tenum[i].fillna(trnum[i].median(), inplace = True)

  train_x1 = pd.concat([trnum, trcat], axis = 1)
  test_x1 = pd.concat([tenum, tecat], axis = 1)

  return train_x1, test_x1

train_x1, test_x1 = impute(train_cat, test_cat, train_num, test_num)

def encode(trx1, tray, tex1):
  enc = ce.CatBoostEncoder()
  enc.fit(trx1, tray)
  train_x3 = pd.DataFrame(enc.transform(trx1), columns = trx1.columns)
  test_x3 = pd.DataFrame(enc.transform(tex1), columns = tex1.columns)

  return train_x3, test_x3, enc

train_x3, test_x3, enc = encode(train_x1, train_y, test_x1)

def scale(trx3, tex3):
  scaler = StandardScaler()
  scaler.fit(trx3)
  trx3 = pd.DataFrame(scaler.transform(trx3), columns = trx3.columns)
  tex3 = pd.DataFrame(scaler.transform(tex3), columns = tex3.columns)

  return trx3, tex3, scaler

train_x3, test_x3, scaler = scale(train_x3, test_x3)

knn = KNeighborsClassifier()
lore = LogisticRegression(random_state = 69, solver = 'liblinear')
dtree = DecisionTreeClassifier(random_state = 69)
rforest = RandomForestClassifier(random_state = 69)

knn_param = [{'n_neighbors' : [2, 3, 4]}]
lore_param = [{'solver' : ["lbfgs", "libinear","saga"]}]
dtree_param = [{'max_depth' : [3, 4, 5], 'max_features' : ['auto', 'sqrt', 'log2']}]
rforest_param = [{'n_estimators' : [5, 10, 20], 'max_depth' : [3, 4, 5]}]

inner_cv = KFold(n_splits = 3)
gridcvs = {}

# estimate performance of hyperparameter tuning and model algorithm pipeline
for params, model, name in zip((knn_param, lore_param, dtree_param, rforest_param), (knn, lore, dtree, rforest), ('LogicticRegression', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier')):
    
    # perform hyperparameter tuning
    gcv = GridSearchCV(estimator = model, param_grid = params, cv = inner_cv, 
                       scoring = 'f1',
                       refit = True)
    
    gridcvs[name] = gcv


outer_cv = KFold(n_splits = 5)

# outer loop cv
for name, gs_model in sorted(gridcvs.items()):
      nested_score = cross_val_score(gs_model, train_x3, train_y, 
                                     cv = outer_cv, n_jobs = -1, 
                                     scoring = 'f1')
      print(name, nested_score.mean(), nested_score.std())


# select HP for the best model (model2) based on regular k-fold on whole training set    
final_cv = KFold(n_splits = 5)

gcv_final_HP = GridSearchCV(estimator = knn,
                            param_grid = knn_param,
                            cv = final_cv, scoring = 'f1'
                            )
    
gcv_final_HP.fit(train_x3, train_y)



# check the score of normal gridsearchcv and compare that to the best model selected in nestedcv
# if the score are very different then bias has been introduced. In this case, the scores are exactly similar.
gcv_final_HP.best_score_

# get the best params from the gcv_final_HP
gcv_final_HP.best_params_


final_model = KNeighborsClassifier(n_neighbors = 3)
# fit the model to whole "training" dataset
final_model.fit(train_x3, train_y)
pred = final_model.predict(test_x3)

from sklearn.metrics import confusion_matrix, f1_score

c_matrix = confusion_matrix(test_y, pred)

f1score = f1_score(test_y, pred)

# save the model
dump(final_model, open('model.pkl', 'wb'))
# save the scaler
dump(scaler, open('scaler.pkl', 'wb'))
# save the encoder
dump(enc, open('encoder.pkl', 'wb'))