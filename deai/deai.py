import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import lightgbm as lgb

#データの読み込み
data = pd.read_csv('my_data_1026.csv')
data.head()

#null削除
data = data.dropna()

#変数の抽出
x = data.iloc[:,0:1231]
y1 = data['PC1_class']
y2 = data['PC2_class']
y3 = data['PC3_class']

#lightgbmがカラム名にカッコが入っているとダメなのでカラム名からカッコを消す
pd.set_option('display.max_rows', 1100)
x_column=[]
x_columns=[]

[x_column.append(i.replace('[','')) for i in x.columns]
[x_columns.append(i.replace(']','')) for i in x_column]
x.columns=x_columns
x.columns

#実際使う変数の抽出（使わない変数をdelete）
pd.set_option('display.max_rows', 1100)

del_names = pd.read_csv('remain.csv')
del_names = del_names.iloc[0:1092]
del_names = del_names.drop(del_names.index[[1,2,3,946,1015]])
for i in del_names['remain']:
    del x[i]
del x['CKAH_01']
del x['CKAI_01']

# 学習データとテストデータのインデックスを作成
ss = ShuffleSplit(n_splits=1,
                  train_size=0.8,
                  test_size=0.2,
                  random_state=123)

train_index, test_index = next(ss.split(x))

x_train, x_test = x.iloc[train_index], x.iloc[test_index] # 学習データ，


y_train_pc1, y_test_pc1 = y1.iloc[train_index], y1.iloc[test_index] # PC1の学習データのラベル，テストデータのラベル

y_train_pc2, y_test_pc2 = y2.iloc[train_index], y2.iloc[test_index] # PC2の学習データのラベル，テストデータのラベル

y_train_pc3, y_test_pc3 = y3.iloc[train_index], y3.iloc[test_index] # PC3の学習データのラベル，テストデータのラベル

#PC1 クロスバリデーション後

lgb_train = lgb.Dataset(x_train, y_train_pc1)
lgb_eval = lgb.Dataset(x_test, y_test_pc1)


params = {'metric': 'binary_error',
  'objective': 'binary',
  'verbosity': -1,
  'boosting_type': 'gbdt',
  'feature_pre_filter': 'false',
  'lambda_l1': 0.0,
  'lambda_l2': 0.0,
  'num_leaves': 3,
  'feature_fraction': 0.7799999999999999,
  'bagging_fraction': 0.6373682966693643,
  'bagging_freq': 6,
  'min_child_samples': 50}

gbm1 = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=10000,
                early_stopping_rounds=100,
                verbose_eval=50)

preds = np.round(gbm1.predict(x_test))
print('Accuracy score = \t {}'.format(accuracy_score(y_test_pc1, preds)))
print('Precision score = \t {}'.format(precision_score(y_test_pc1, preds)))
print('Recall score =   \t {}'.format(recall_score(y_test_pc1, preds)))
print('F1 score =      \t {}'.format(f1_score(y_test_pc1, preds)))

#PC2 クロスバリデーション後
lgb_train = lgb.Dataset(x_train, y_train_pc2)
lgb_eval = lgb.Dataset(x_test, y_test_pc2)


params = {'metric': 'binary_error',
  'objective': 'binary',
  'verbosity': -1,
  'boosting_type': 'gbdt',
  'feature_pre_filter': 'false',
  'lambda_l1': 0.0,
  'lambda_l2': 0.0,
  'num_leaves': 3,
  'feature_fraction': 0.4,
  'bagging_fraction': 0.4925666524340133,
  'bagging_freq': 3,
  'min_child_samples': 20}

gbm2 = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=10000,
                early_stopping_rounds=100,
                verbose_eval=50)

preds = np.round(gbm2.predict(x_test))
print('Accuracy score = \t {}'.format(accuracy_score(y_test_pc2, preds)))
print('Precision score = \t {}'.format(precision_score(y_test_pc2, preds)))
print('Recall score =   \t {}'.format(recall_score(y_test_pc2, preds)))
print('F1 score =      \t {}'.format(f1_score(y_test_pc2, preds)))