import warnings
warnings.filterwarnings('ignore')
import os
import gc
import time
import feather
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas()

sns.set_style('white')
gc.enable()

DATA_DIR = '../input/home-credit-default-risk/'

import glob
def get_path(str, first=True, parent_dir='../input/**/'):
    res_li = glob.glob(parent_dir+str)
    return res_li[0] if first else res_li

def load_folds_lables():
    path = '../input/hcdr-prepare-kfold/'
    eval_sets = np.load(path+'eval_sets.npy')
    y = np.load(path+'target.npy')
    return eval_sets, y
folds, labels = load_folds_lables()
nfolds = 5
train_num = len(labels)
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def nest_print(dict_item, inline=True, indent=True):
    s = []
    s_ind = '\t' if indent else ''
    for k, v in dict_item.items():
        s += [': '.join([str(k), str(round(v, 6))])]
    if inline:
        print(s_ind+' '.join(s))
    else:
        print(s_ind+'\n'.join(s))

def get_imp_plot(lgb_res, savefig=True):
    lgb_feat_imps = lgb_res['feat_imps']
    name = lgb_res['name']
    lgb_imps = pd.DataFrame(
        np.vstack(lgb_feat_imps).T, 
        columns=['fold_{}'.format(i) for i in range(nfolds)],
        index=feature_name,
    )
    lgb_imps['fold_mean'] = lgb_imps.mean(1)
    lgb_imps = lgb_imps.loc[
        lgb_imps['fold_mean'].sort_values(ascending=False).index
    ]
    lgb_imps.reset_index().to_csv(f'{name}_lgb_imps.csv', index=False)
    del lgb_imps['fold_mean']; gc.collect();

    max_num_features = min(len(feature_name), 300)
    f, ax = plt.subplots(figsize=[8, max_num_features//4])
    data = lgb_imps.iloc[:max_num_features].copy()
    data_mean = data.mean(1).sort_values(ascending=False)
    data = data.loc[data_mean.index]
    data_index = data.index.copy()
    data = [data[c].values for c in data.columns]
    data = np.hstack(data)
    data = pd.DataFrame(data, index=data_index.tolist()*nfolds, columns=['igb_imp'])
    data = data.reset_index()
    data.columns = ['feature_name', 'igb_imp']
    sns.barplot(x='igb_imp', y='feature_name', data=data, orient='h', ax=ax)
    plt.grid()
    if savefig:
        plt.savefig(f'{name}_lgb_imp.png')

def lgb_cv_train(
    name, params, X, y, X_test, feature_name,
    num_boost_round, early_stopping_rounds, verbose_eval,
    cv_folds, metric=roc_auc_score,
    verbose_cv=True, nfolds=nfolds, msgs={}
):
    pred_test = np.zeros((X_test.shape[0],))
    pred_val = np.zeros((X.shape[0],))
    cv_scores = []
    feat_imps = []
    models = []
    for valid_fold in range(nfolds):
        mask_te = cv_folds==valid_fold
        mask_tr = ~mask_te
        print('[level 1] processing fold %d...'%(valid_fold+1))
        t0 = time.time()
        dtrain = lgb.Dataset(
            X[mask_tr], y[mask_tr],
            feature_name=feature_name,
            free_raw_data=False
        )
        dvalid = lgb.Dataset(
            X[mask_te], y[mask_te],
            feature_name=feature_name,
            free_raw_data=False
        )
        evals_result = {}
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dvalid],
            valid_names=['train','valid'],
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval
        )
        pred_val[mask_te] = model.predict(X[mask_te])
        pred_test += model.predict(X_test)/nfolds
        scr = metric(y[mask_te], pred_val[mask_te])
        feat_imps.append(model.feature_importance()/model.best_iteration)
        if verbose_cv:
            print(f'{name} auc:', scr, 
                  f'fold {valid_fold+1} done in {time.time() - t0:.2f} s')
        cv_scores.append(scr)
        models.append(model)
    scr = metric(y, pred_val)
    msgs = dict(
        msgs, 
        cv_score_mean=np.mean(cv_scores), 
        cv_score_std=np.std(cv_scores),
        cv_score_min=np.min(cv_scores), 
        cv_score_max=np.max(cv_scores),
        oof_score=scr
    )
    nest_print(msgs)
    result = dict(
        name=name,
        pred_val=pred_val,
        pred_test=pred_test,
        cv_scores=cv_scores,
        models=models,
        feat_imps=feat_imps
    )
    return result
train_ids = pd.read_csv(get_path('*application_train.csv'), usecols=['SK_ID_CURR'])['SK_ID_CURR'].values
test_ids = pd.read_csv(get_path('*application_test.csv'), usecols=['SK_ID_CURR'])['SK_ID_CURR'].values
sk_id_curr = np.load(get_path('sk_id_curr*'))
train_ids.shape, test_ids.shape, sk_id_curr.shape
def label_encoding(df):
    obj_cols = [c for c in df.columns if df[c].dtype=='O']
    for c in obj_cols:
        df[c] = pd.factorize(df[c], na_sentinel=-1)[0]
    df[obj_cols].replace(-1, np.nan, inplace=True)
    return df
id_labels = pd.DataFrame()
id_labels['SK_ID_CURR'] = sk_id_curr
id_labels['TARGET'] = -1
id_labels['TARGET'][:train_num] = labels
id_labels['fold'] = -1
id_labels['fold'][:train_num] = folds
results = {}

lgb_params =  {
    'boosting_type': 'gbdt', 
    'objective': 'binary', 
    'metric': 'auc', 
    'num_threads': 4, 
    'min_data_in_leaf': 80,
    'max_depth': -1,
    'num_leaves': 28,
    'seed': 233,
    'lambda_l1': 0.04,
    'lambda_l2': 0.4,
    'feature_fraction': 0.84,
    'bagging_fraction': 0.77,
    'bagging_freq': 1,
    'learning_rate': 0.02,
    #'min_split_gain': 0.0222415,
    #'min_child_weight': 40,
    'verbose': -1
}

round_params = dict(
    num_boost_round = 20000,
    early_stopping_rounds = 100,
    verbose_eval = 50,
)
csvname_li = [
    'bureau',
    #'bureau_balance',
    'previous_application',
    'installments_payments',
    'credit_card_balance',
    'POS_CASH_balance',
]

csvname = 'credit_card_balance'

print(f'Current: {csvname}...')

df = pd.read_csv(DATA_DIR+f'{csvname}.csv')
df = df.loc[np.isin(df['SK_ID_CURR'], sk_id_curr)]
df = df.merge(id_labels, how='left', on='SK_ID_CURR')
df = label_encoding(df)
df.head().T
tmp = df[df['SK_ID_PREV']==2582071.000].sort_values('MONTHS_BALANCE')
tmp.T
df['amt_bal_lmt_diff'] = df['AMT_BALANCE'] - df['AMT_CREDIT_LIMIT_ACTUAL']
df['amt_pcurr_lmt_diff'] = df['AMT_PAYMENT_CURRENT'] - df['AMT_CREDIT_LIMIT_ACTUAL']

df['amt_bal_tot_diff'] = df['AMT_BALANCE'] - df['AMT_TOTAL_RECEIVABLE']
df['amt_rcv_tot_diff'] = df['AMT_RECIVABLE'] - df['AMT_TOTAL_RECEIVABLE']
df['amt_pcurr_ptotc_diff'] = df['AMT_PAYMENT_CURRENT'] - df['AMT_PAYMENT_TOTAL_CURRENT']

df['amt_prcp_rcv_ratio'] = df['AMT_RECEIVABLE_PRINCIPAL']/(1+df['AMT_RECIVABLE'])
df['amt_insmin_bal_ratio'] = df['AMT_INST_MIN_REGULARITY']/(1+df['AMT_BALANCE'])
df['amt_insmin_ptotc_ratio'] = df['AMT_INST_MIN_REGULARITY']/(1+df['AMT_PAYMENT_TOTAL_CURRENT'])

df['amt_dcurr_per'] = df['AMT_DRAWINGS_CURRENT']/(1+df['CNT_DRAWINGS_CURRENT'])
df['amt_dac_per'] = df['AMT_DRAWINGS_ATM_CURRENT']/(1+df['CNT_DRAWINGS_ATM_CURRENT'])
df['amt_doc_per'] = df['AMT_DRAWINGS_OTHER_CURRENT']/(1+df['CNT_DRAWINGS_OTHER_CURRENT'])
df['amt_dpc_per'] = df['AMT_DRAWINGS_POS_CURRENT']/(1+df['CNT_DRAWINGS_POS_CURRENT'])
df = df.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
df['CNT_INSTALMENT_MATURE_CUM'].fillna(0, inplace=True)
df['flag_ins_mature'] = df.groupby('SK_ID_PREV', sort=False)['CNT_INSTALMENT_MATURE_CUM'].diff()
from joblib import Parallel, delayed
def func(index, group, main_key, prefix=''):
    res = pd.Series()
    
    res['id_size'] = len(group)
    res['dpd_times'] = (group['SK_DPD']>0).sum()
    res['dpd_sum'] = group['SK_DPD'].sum()
    res['dpd_mean'] = group['SK_DPD'].mean()
    res['dpd_max'] = group['SK_DPD'].max()
    res['dpd_std'] = group['SK_DPD'].std()
    
    res['dpd_def_times'] = (group['SK_DPD_DEF']>0).sum()
    res['dpd_def_sum'] = group['SK_DPD_DEF'].sum()
    res['dpd_def_mean'] = group['SK_DPD_DEF'].mean()
    res['dpd_def_max'] = group['SK_DPD_DEF'].max()
    res['dpd_def_std'] = group['SK_DPD_DEF'].std()
    
    res['flag_ins_mature_mean'] = group['flag_ins_mature'].dropna().mean()
    res['flag_ins_mature_sum'] = group['flag_ins_mature'].dropna().sum()
    x = group['AMT_CREDIT_LIMIT_ACTUAL'].values
    res['amt_lmt_ht_diff'] = x[-1] - x[0] if len(x)>1 else np.nan
    res['amt_lmt_l1p_diff_mean']  = np.diff(np.log1p(x)).mean()
    res['amt_lmt_min'] = x.min()
    res['amt_lmt_max'] = x.max()
    res['amt_lmt_std'] = x.std()
    res['amt_lmt_mean'] = x.mean()
    res['amt_lmt_argmin_nrm'] = np.argmin(x) / res['id_size']
    res['amt_lmt_argmax_nrm'] = np.argmax(x) / res['id_size']
    res['amt_lmt_maxmindiff'] = x.max() - x.min()
    if res['amt_lmt_argmin_nrm'] > res['amt_lmt_argmax_nrm']:
        res['amt_lmt_maxmindiff'] = -res['amt_lmt_maxmindiff']
    
    x = group['AMT_PAYMENT_CURRENT'].values
    res['amt_pcurr_min'] = x.min()
    res['amt_pcurr_max'] = x.max()
    res['amt_pcurr_std'] = x.std()
    res['amt_pcurr_mean'] = x.mean()
    
    if prefix!='':
        res.index = [prefix+c for c in res.index]
    
    res[main_key] = index
    return res
cache_dirs = ['./', '../input/hcdr-mr-feat-cc/']
feat_prev = None
main_key = 'SK_ID_PREV'
for d in cache_dirs:
    if os.path.isdir(d) and 'sk_id_prev_feat.csv' in os.listdir(d):
        print('found cache')
        feat_prev = pd.read_csv(os.path.join(d, 'sk_id_prev_feat.csv'))
        break
if feat_prev is None:
    feat_prev = Parallel(n_jobs=8)(
        delayed(func)(key, group, main_key, 'prev_') for key,group in tqdm(
            df.groupby([main_key], sort=False)
        )
    )
    feat_prev = pd.DataFrame(feat_prev)
feat_prev.to_csv('sk_id_prev_feat.csv', index=False)
df = df.merge(feat_prev, how='left', on=main_key)
del feat_prev; gc.collect();
def func(index, group, main_key, prefix=''):
    res = pd.Series()
    
    res['dpd_times'] = (group['SK_DPD']>0).sum()
    res['dpd_sum'] = group['SK_DPD'].sum()
    res['dpd_mean'] = group['SK_DPD'].mean()
    res['dpd_max'] = group['SK_DPD'].max()
    res['dpd_std'] = group['SK_DPD'].std()
    
    res['dpd_def_times'] = (group['SK_DPD_DEF']>0).sum()
    res['dpd_def_sum'] = group['SK_DPD_DEF'].sum()
    res['dpd_def_mean'] = group['SK_DPD_DEF'].mean()
    res['dpd_def_max'] = group['SK_DPD_DEF'].max()
    res['dpd_def_std'] = group['SK_DPD_DEF'].std()
    
    if prefix!='':
        res.index = [prefix+c for c in res.index]
    
    res[main_key] = index
    return res
main_key = 'prev_id_size'
feat = Parallel(n_jobs=8)(
    delayed(func)(key, group, main_key, 'prevsize_') for key,group in tqdm(
        df.groupby([main_key], sort=False)
    )
)
feat = pd.DataFrame(feat)
df = df.merge(feat, on=main_key, how='left')
# %%time
# main_key = 'NAME_CONTRACT_STATUS'
# feat = Parallel(n_jobs=8)(
#     delayed(func)(key, group, main_key, 'stats_') for key,group in tqdm(
#         df.groupby([main_key], sort=False)
#     )
# )
# feat = pd.DataFrame(feat)
# df = df.merge(feat, on=main_key, how='left')
eval_cols = ['SK_ID_CURR', 'fold', 'TARGET']
eval_df = df[eval_cols].copy()
df.drop(eval_cols+['SK_ID_PREV'], axis=1, inplace=True)

feature_name = df.columns.tolist()
y = eval_df['TARGET'].values.copy()
X = df.loc[y!=-1].values
X_test = df.loc[y==-1].values
cv_folds = eval_df.loc[y!=-1, 'fold'].values
y = y[y!=-1]

del df; gc.collect();
print('shapes', X.shape, y.shape, cv_folds.shape, X_test.shape)
res = lgb_cv_train(
    f'{csvname}', lgb_params,
    X, y, X_test, feature_name, 
    cv_folds=cv_folds,
    **round_params
)

eval_df['pred'] = -1
eval_df.loc[eval_df['fold']!=-1, 'pred'] = res['pred_val']
eval_df.loc[eval_df['fold']==-1, 'pred'] = res['pred_test']
eval_df.to_csv('eval_res.csv', index=False)
get_imp_plot(res, False)