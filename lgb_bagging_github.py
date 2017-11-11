
# coding: utf-8

# In[1]:

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
get_ipython().magic(u'matplotlib inline')
import lightgbm as lgbm


# In[10]:

#Data Available at following link
#https://datahack.analyticsvidhya.com/contest/data-science-hackathon-churn-prediction/
train = pd.read_csv('./train.csv',low_memory=False)
test = pd.read_csv('./test.csv',low_memory=False)
all_data=pd.concat([train,test])
del train,test
gc.collect()


# In[11]:

#Three ways to identify columns that can be dropped
# Run lgbm and indentify feature importance, remove all features with value equal to 0
# Features with high autocorrealtion, Features with high correlation against a given feature is dropped
# Categorical features wiht high levels such as zip, brn_code
colsToDrop ={'AGRI_Closed_PrevQ1','AGRI_DATE','AGRI_PREM_CLOSED_PREVQ1',
 'AGRI_TAG_LIVE','AL_CNC_Closed_PrevQ1','AL_CNC_PREM_CLOSED_PREVQ1','AL_Closed_PrevQ1',
 'AL_PREM_CLOSED_PREVQ1','ATM_C_prev1','ATM_C_prev2','ATM_C_prev3','ATM_C_prev4',
 'ATM_C_prev5','ATM_C_prev6','BL_Closed_PrevQ1','BL_PREM_CLOSED_PREVQ1','BRN_CW_Amt_prev1','BRN_CW_Amt_prev2',
 'BRN_CW_Amt_prev3','BRN_CW_Amt_prev4','BRN_CW_Amt_prev5','BRN_CW_Amt_prev6','BRN_CW_Cnt_prev1',
 'BRN_CW_Cnt_prev2','BRN_CW_Cnt_prev3','BRN_CW_Cnt_prev4','BRN_CW_Cnt_prev5','BRN_CW_Cnt_prev6',
 'CC_PREM_CLOSED_PREVQ1','CE_Closed_PrevQ1','CE_DATE','CE_PREM_CLOSED_PREVQ1','CE_TAG_LIVE',
 'COUNT_ATM_C_prev1','COUNT_ATM_C_prev2','COUNT_ATM_C_prev3','COUNT_ATM_C_prev4','COUNT_ATM_C_prev5','COUNT_ATM_C_prev6',
 'COUNT_MB_C_prev1','COUNT_MB_C_prev2','COUNT_MB_C_prev3','COUNT_MB_C_prev4','COUNT_MB_C_prev5',
 'COUNT_MB_C_prev6','COUNT_POS_C_prev1','COUNT_POS_C_prev2','COUNT_POS_C_prev3','COUNT_POS_C_prev4','COUNT_POS_C_prev5',
 'COUNT_POS_C_prev6','CV_Closed_PrevQ1','CV_PREM_CLOSED_PREVQ1','Complaint_Logged_PrevQ1','Complaint_Resolved_PrevQ1','EDU_Closed_PrevQ1',
 'EDU_DATE','EDU_PREM_CLOSED_PREVQ1','EDU_TAG_LIVE','FRX_PrevQ1',
 'FRX_PrevQ1_N','GL_Closed_PrevQ1','LAP_DATE','LAS_DATE','LAS_TAG_LIVE','MB_C_prev1','MB_C_prev2','MB_C_prev3',
 'MB_C_prev4','MB_C_prev5','MB_C_prev6','MB_D_prev1','MB_D_prev2','MB_D_prev3','MB_D_prev4','MB_D_prev5','MB_D_prev6',
 'MF_TAG_LIVE','OTHER_LOANS_Closed_PrevQ1','OTHER_LOANS_PREM_CLOSED_PREVQ1','OTHER_LOANS_TAG_LIVE','PL_Closed_PrevQ1',
 'PL_PREM_CLOSED_PREVQ1','POS_C_prev1','POS_C_prev2','POS_C_prev3','POS_C_prev4','POS_C_prev5','POS_C_prev6',
 'RD_TAG_LIVE','Req_Logged_PrevQ1','TL_Closed_PrevQ1','TL_DATE','TL_PREM_CLOSED_PREVQ1','TL_TAG_LIVE','TWL_Closed_PrevQ1',
 'TWL_PREM_CLOSED_PREVQ1','TWL_TAG_LIVE','brn_code','city','lap_tag_live','zip'}


# In[12]:

all_data.drop(colsToDrop,inplace=True,axis=1)
gc.collect()


# In[13]:

strcols = []
for z in zip(all_data.columns,all_data.dtypes):
    if z[1]=='object':
        print z
        strcols.append(z[0])
for col in strcols:
    all_data[col].fillna('NA',inplace=True)
    le = LabelEncoder()
    all_data.loc[:,col] = le.fit_transform(all_data[col].values)
train = all_data[~all_data.Responders.isnull()]
test = all_data[all_data.Responders.isnull()]
del all_data
gc.collect()


# In[14]:

print train.Responders.value_counts()


# In[15]:

kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=23, shuffle=True)
ids_kfold = train[['UCIC_ID']].copy()
for i, (train_index, test_index) in enumerate(skf.split(train.UCIC_ID.values, train.Responders.values)):
    ids_kfold.loc[test_index,'kfold_val_num']=i
ids_kfold.to_csv('kfold_ids_2.csv',index=False)


# In[16]:

lgb_params = {}
lgb_params['learning_rate'] = 0.03
lgb_params['max_depth'] = 7
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 100
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 150
lgb_params['objective'] = 'binary',
lgb_params['metric'] = {'binary_logloss', 'auc'}
lgb_params['verbose'] = 1
lgb_params['scale_pos_weight'] = 1.


# In[17]:

eop_d = train.filter(regex='^EOP.*')
train.loc[:,'ef_EOP'] = train.EOP_prev1/train.EOP_prev2
#train.loc[:,'ef_EOP_mean'] = eop_d.T.mean()
train.loc[:,'ef_EOP_skew'] = eop_d.T.skew()


# In[18]:

eop_d = train.filter(regex='^C_prev.*')
train.loc[:,'ef_C'] = train.C_prev1/train.C_prev2

eop_d = train.filter(regex='^D_prev.*')

train.loc[:,'ef_D'] = train.D_prev1/train.D_prev2

eop_d = train.filter(regex='^BAL_prev.*')
train.loc[:,'ef_BAL'] = train.BAL_prev1/train.BAL_prev2

eop_d = train.filter(regex='^BRANCH_D_prev.*')
train.loc[:,'ef_BR_D'] = train.BRANCH_D_prev1/train.BRANCH_D_prev2


# In[19]:

train['cd_diff'] = (train.C_prev1-train.D_prev1)
train['EOP_AMB_diff'] = (train.EOP_prev1 - train.CR_AMB_Prev1)
train['EOP_AMB_ratio'] = (train.EOP_prev1 / train.CR_AMB_Prev1)
gc.collect()


# In[20]:

vintage = pd.read_csv('./train.csv',usecols=['vintage']).vintage.values
train['vintage'] = np.floor(vintage/180)
train['vintage_yr'] = 0 #np.floor(vintage/365)                            


# In[21]:

print train.columns


# In[23]:

def two_decile(preds, dtrain):
    labels = dtrain.get_label()
    srt_preds=np.sort(preds)
    tmp = pd.DataFrame()
    tmp['label']=labels
    tmp['pred']=preds
    tmp=tmp.sort_values('pred',ascending=False).head(2*(tmp.shape[0]/10))
    score=(1.*tmp.label.sum())/np.sum(labels)
    return [('2dec', score, True)]


# In[24]:

lgb_seeds = [23437,4654787,3453467,23477,8967678]


# In[26]:

ids_kfold = pd.read_csv('./kfold_ids_2.csv')
kfold = 5
nbag = 4
lgb_models = []
for i in range(kfold):
    #if i!= 3:
    #    continue
    bag_mdls = []
    print('[Fold %d/%d]' % (i + 1, kfold))
    val_id = ids_kfold[ids_kfold.kfold_val_num==i].UCIC_ID
    train_valid = train[train.UCIC_ID.isin(val_id)]
    train_tr = train[~train.UCIC_ID.isin(val_id)]
    train.Responders
    print(train_valid.shape,train_tr.shape)
    
    y_train = train_tr.Responders.values
    y_valid = train_valid.Responders.values
    X_train = train_tr.drop(['UCIC_ID', 'Responders'], axis=1)
    X_valid = train_valid.drop(['UCIC_ID', 'Responders'], axis=1)
     
    # Convert our data into XGBoost format
    lgb_train = lgbm.Dataset(X_train, y_train)
    lgb_val = lgbm.Dataset(X_valid, y_valid)

    for bag in range(nbag):
        print('[Bag %d/%d]' % (bag + 1, nbag))
        lgb_params['seed'] = lgb_seeds[bag]
        mdl=lgbm.train(lgb_params,lgb_train,valid_sets=lgb_val,num_boost_round=1401,verbose_eval=100,feval=two_decile)
        bag_mdls.append(mdl)
    
    
    lgb_models.append(bag_mdls)
    
    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
    
    del train_valid,train_tr,lgb_train,lgb_val
    gc.collect()
    


# In[27]:

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


# In[29]:

for t in range(1200, 1401, 25):
    target_df = train[['Responders']]
    for i in range(kfold):
        val_id = ids_kfold[ids_kfold.kfold_val_num==i].UCIC_ID
        train_valid = train[train.UCIC_ID.isin(val_id)]
        train_valid_idx = train_valid.index

        X_valid = train_valid.drop(['UCIC_ID', 'Responders'], axis=1)

        # Train the model! We pass in a max of 700 rounds (with early stopping after 70)
        # and the custom metric (maximize=True tells xgb that higher metric is better)
        target_df.loc[train_valid_idx,'pred']=0
        for bag in range(nbag):
            mdl = lgb_models[i][bag]
            target_df.loc[train_valid_idx,'pred'] += mdl.predict(X_valid,num_iteration=t)
        target_df.loc[train_valid_idx,'pred']/=nbag
        del train_valid
        gc.collect()
    print 'tree depth:',t,roc_auc_score(target_df.Responders.values,target_df.pred.values)
    target_sort = target_df.sort_values('pred',ascending=False)
    top_20_target = target_sort.head(2*30000)
    print '2decal',(1.*top_20_target.Responders.sum())/target_sort.Responders.sum()


# In[30]:

for t in range(1325, 1326, 25):
    target_df = train[['Responders']]
    for i in range(kfold):
        val_id = ids_kfold[ids_kfold.kfold_val_num==i].UCIC_ID
        train_valid = train[train.UCIC_ID.isin(val_id)]
        train_valid_idx = train_valid.index

        X_valid = train_valid.drop(['UCIC_ID', 'Responders'], axis=1)
        target_df.loc[train_valid_idx,'pred']=0
        for bag in range(nbag):
            mdl = lgb_models[i][bag]
            target_df.loc[train_valid_idx,'pred'] += mdl.predict(X_valid,num_iteration=t)
        target_df.loc[train_valid_idx,'pred']/=nbag
        del train_valid
        gc.collect()
    print 'tree depth:',t,log_loss(target_df.Responders.values,target_df.pred.values)
    print 'tree depth:',t,roc_auc_score(target_df.Responders.values,target_df.pred.values)


# In[31]:

target_sort = target_df.sort_values('pred',ascending=False)
top_20_target = target_sort.head(2*30000)
(1.*top_20_target.Responders.sum())/target_sort.Responders.sum()


# In[32]:

target_df.to_csv('lgb_bag_5_680793.csv',index=False)


# In[33]:

ucic_id = test.UCIC_ID.values


# In[34]:

subm = pd.DataFrame()
subm['UCIC_ID'] = ucic_id
subm['Responders'] = 0


# In[35]:

set(test.columns).difference(set(train.columns))


# In[37]:

print 'tree:',t
t=801
for i in range(kfold):
    for bag in range(nbag):
        print i,bag
        mdl = lgb_models[i][bag]
        subm.loc[:,'Responders'] += mdl.predict(test.drop(['UCIC_ID', 'Responders'],axis=1),num_iteration=t)

    gc.collect()
subm['Responders'] /= kfold


# In[38]:

subm['Responders'] /= nbag


# In[39]:

subm.to_csv('subm/lgb_bag_4_680793.csv',index=False)


# In[44]:

colset = set(train.columns)
for i in range(kfold):
    mdl  = lgb_models[i][0]
    imp=pd.DataFrame()
    imp.loc[:,'name'] = mdl.feature_name()
    imp.loc[:,'val'] = mdl.feature_importance()
    nvcols = set(imp[imp.val<=0].name.values)
    colset = colset.intersection(nvcols)


# In[45]:

print colset

