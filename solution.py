import pandas as pd
import numpy as np
from os.path import join
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# 1. read data

ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
train = pd.read_csv(join('data', "train.csv"))


# 2. blocking
def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR

#blocking on attr category 
def block_by_cat(ltable, rtable):
    # ensure category is str
    ltable['category'] = ltable['category'].astype(str)
    rtable['category'] = rtable['category'].astype(str)

    # get all categories
    cats_l = set(ltable["category"].values)
    cats_r = set(rtable["category"].values)
    cats = cats_l.union(cats_r)

    # map each category to left ids and right ids
    cat2ids_l = {c.lower(): [] for c in cats}
    cat2ids_r = {c.lower(): [] for c in cats}
    for i, x in ltable.iterrows():
        cat2ids_l[x["category"].lower()].append(x["id"])
    for i, x in rtable.iterrows():
        cat2ids_r[x["category"].lower()].append(x["id"])

    # put id pairs that share the same category in candidate set
    candset = []
    for brd in cats:
        l_ids = cat2ids_l[brd]
        r_ids = cat2ids_r[brd]
        for i in range(len(l_ids)):
            for j in range(len(r_ids)):
                candset.append([l_ids[i], r_ids[j]])
    return candset

# blocking to reduce the number of pairs to be compared
candset = block_by_cat(ltable, rtable)
print("number of pairs originally", ltable.shape[0] * rtable.shape[0])
print("number of pairs after blocking",len(candset))
candset_df = pairs2LR(ltable, rtable, candset)



# 3. Feature engineering
import Levenshtein as lev

def jaccard_similarity(row, attr):
    x = set(row[attr + "_l"].lower().split())
    y = set(row[attr + "_r"].lower().split())
    return len(x.intersection(y)) / max(len(x), len(y))


def levenshtein_distance(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    return lev.distance(x, y)

def feature_engineering(LR):
    LR = LR.astype(str)
    attrs = ["title", "category", "brand", "modelno", "price"]
    features = []
    for attr in attrs:
        j_sim = LR.apply(jaccard_similarity, attr=attr, axis=1)
        l_dist = LR.apply(levenshtein_distance, attr=attr, axis=1)
        features.append(j_sim)
        features.append(l_dist)
    features = np.array(features).T
    return features
candset_features = feature_engineering(candset_df)

# also perform feature engineering to the training set
training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_features = feature_engineering(training_df)
training_label = train.label.values

# 4. Model training and prediction
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight="balanced",n_jobs=-1, random_state=4)
rf_gscv = GridSearchCV(rf,scoring="f1",param_grid={ "n_estimators":[50,100,200,300,400,500], "max_depth":[5,10,15,20,25]
},cv=5,n_jobs=-1)
rf_gscv.fit(training_features, training_label)
best_rf2 = rf_gscv.best_estimator_

kF = StratifiedKFold(n_splits=4)

f1 = []
recall = []
precision = []
for train_index, test_index in kF.split(training_features, training_label):
    reduced_data_train = training_features[train_index]
    reduced_labels_train = training_label[train_index]
    
    reduced_data_test = training_features[test_index]
    reduced_labels_test = training_label[test_index]
    
    rf2 = RandomForestClassifier(n_estimators=300,max_depth=10,class_weight="balanced")
    rf2.fit(reduced_data_train, reduced_labels_train)
    
    y_pred = rf2.predict(reduced_data_test)
    f1.append(f1_score(reduced_labels_test,y_pred))
    recall.append(recall_score(reduced_labels_test,y_pred))
    precision.append(precision_score(reduced_labels_test,y_pred))

f1 = np.array(f1)
recall = np.array(recall)
precision = np.array(precision)

print("F1:{}".format(f1.mean()))
print("Recall:{}".format(recall.mean()))
print("Precision:{}".format(precision.mean()))

cand_y_pred = best_rf2.predict(candset_features)

# 5. output

matching_pairs = candset_df.loc[cand_y_pred == 1, ["id_l", "id_r"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs_in_training = training_df.loc[training_label == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)