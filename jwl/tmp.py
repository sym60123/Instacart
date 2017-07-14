import numpy as np
import pandas as pd
import lightgbm as lgb
import gc

products = pd.read_pickle('products')
priors = pd.read_pickle('priors')
users = pd.read_pickle('users')
userXproduct = pd.read_pickle('userXproduct')
df_temp = pd.read_pickle('df_temp')
df_train = pd.read_pickle('df_train')

def eval_fun(labels, preds):
    labels = labels.split(' ')
    preds = preds.split(' ')
    rr = (np.intersect1d(labels, preds))
    precision = np.float(len(rr)) / len(preds)
    recall = np.float(len(rr)) / len(labels)
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)


#把预测结果 通过阈值 挑选出来 放入一个字符串中 ‘product1 2 3 4···’
def get_pred_results(df_test,thrshold=0.22):
    TRESHOLD = thrshold  # guess, should be tuned with crossval on a subset of train data

    d = dict()
    for row in df_test.itertuples():
        if row.pred > TRESHOLD:
            try:
                d[row.order_id] += ' ' + str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)

    for order in df_test.order_id:
        if order not in d:
            d[order] = 'None'

    sub = pd.DataFrame.from_dict(d, orient='index')
    sub.reset_index(inplace=True)
    sub.columns = ['order_id', 'products']
    return sub

def fscore_nn(data, pred, model, alpha):
    data['pred'] = pred
    data_pred = get_pred_results(data, thrshold=alpha)
    # 合表
    data_pred1 = pd.merge(data_pred, df_temp, on=['order_id'])
    # 求F1结果表
    res = list()
    for entry in data_pred1.itertuples():
        res.append(eval_fun(entry[2], entry[3]))
    res = pd.DataFrame(np.array(res), columns=['precision', 'recall', 'f1'])
    return res["precision"].mean(), res['recall'].mean(), res['f1'].mean()

labels = np.array(df_train['labels'],dtype=pd.Series)
df_train.drop(['labels'],axis=1,inplace=True)

f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
       'user_average_days_between_orders', 'user_average_basket',
       'order_hour_of_day', 'days_since_prior_order', #'days_since_ratio',
       'aisle_id', 'department_id', 'product_orders', 'product_reorders',
       'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
       'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
       'UP_delta_hour_vs_last'] # 'dow', 'UP_same_dow_as_last_order'


from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import cross_validation
from sklearn import preprocessing

train, test, train_labels, test_labels = cross_validation.train_test_split(df_train, labels, test_size=0.3, random_state=0)

scalerX = preprocessing.MinMaxScaler(feature_range=(0, 1))
train_x = scalerX.fit_transform(train[f_to_use])
test_x = scalerX.transform(test[f_to_use])


model = Sequential()
model.add(Dense(128, input_dim=18, activation='relu'))
model.add(Dense(256, input_dim=128, activation='relu'))
model.add(Dense(64, input_dim=256, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

from theano.tensor import basic as tensor
def weight_crossentropy(output, target):
    return -(target * tensor.log(output)+ (1.0 - target) * tensor.log(1.0 - output))

# model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=["accuracy"])
model.compile(loss=weight_crossentropy, optimizer="rmsprop", metrics=["accuracy"])
pred_prob = model.predict_proba(test_x, batch_size=5000)

print(fscore_nn(test, pred_prob, model, 0.12))