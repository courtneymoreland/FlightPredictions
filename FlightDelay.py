import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.pipeline import  Pipeline
from sklearn.compose import make_column_transformer
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve
from sklearn.decomposition import TruncatedSVD


metrics_dict = dict()

def metrics(grid, X_test, y_test, label):

    global metrics_dict
    y_test_scores = grid.predict(X_test)
    lr_probs = grid.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, lr_probs)
    lr_precision, lr_recall, threshold_rec = precision_recall_curve(y_test, lr_probs)
    fpr, tpr, thresholds = roc_curve(y_test, lr_probs)
    
    print(f'ROC AUC SCORE: {roc_auc:.2f}')
    print(classification_report(y_test, y_test_scores))
    
    cm = confusion_matrix(y_test, y_test_scores)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)
    disp.plot(cmap='viridis', values_format='.0f')
    plt.title('Confusion Matrix')
    plt.show()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    lr_fpr, lr_tpr, threshold_roc = roc_curve(y_test, lr_probs)
    ax1.plot(lr_fpr, lr_tpr, label=label, linewidth=2)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    
    ax2.plot(lr_recall, lr_precision, label='Logistic', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    
    plt.legend()
    plt.show()
    
    metrics_dict[label] = {
        'precision': lr_precision,
        'recall': lr_recall,
        'false_positives': lr_fpr,
        'true_positives': lr_tpr,
        'threshold_precision': threshold_rec,
        'threshold_roc': threshold_roc
    }
    
    return roc_auc, lr_precision, lr_recall
def X_y(df, balance=False):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    if balance:
        smote_tomek = SMOTETomek(random_state=0)
        cat = X_train.apply(lambda column: column.dtype.name=='category')
        smote_nc = SMOTENC(categorical_features=cat, random_state=0)
        X_train, y_train = smote_nc.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test, X_val, y_val
def transform(df):
    all_cats = []
    for column in df.iloc[:,2:9].columns:
        all_cats.append(sorted(df[column].unique()))

    cat_columns = list(df.iloc[:,2:9].columns.values)

    dep_timeslots = sorted(data.DEP_TIME_BLK.unique().tolist()) 
    arr_timeslots = sorted(data.ARR_TIME_BLK.unique().tolist())

    transformer = make_column_transformer((OrdinalEncoder(categories=[dep_timeslots]), ['DEP_TIME_BLK']), (OrdinalEncoder(categories=[arr_timeslots]), ['ARR_TIME_BLK']), (OneHotEncoder(categories=all_cats), cat_columns), (StandardScaler(), ['DAY_OF_MONTH', 'DAY_OF_WEEK', 'DISTANCE']), remainder='passthrough')
    print('Tranform done!')
    return transformer
def statistics(df, column_label):
    new_df = df[[column_label,'y']].groupby(column_label).sum()
    new_df['data'] = df[column_label].value_counts().sort_index()
    new_df['PERCENTUAL'] = new_df['y']/new_df['data']*100
    new_df['MEAN'] = sum(new_df['y'])/sum(new_df['data'])*100
    new_df.reset_index(inplace=True)
    max_day = new_df['PERCENTUAL'].max()
    min_day = new_df['PERCENTUAL'].min()
    
    return new_df,max_day,min_day
def time_blocks(df):
    arr_time_blk = []
    for i, row in df.iterrows():
        if row['CANCELLED']==1 or row['DIVERTED']==1:
            arr_time_blk.append("MIA")
        elif np.isnan(row['ARR_TIME']):
            arr_time_blk.append(None)
        elif row['ARR_TIME'] == 2400:
            arr_time_blk.append('22-24')
        elif int(row['ARR_TIME']/100)%2==1:
            arr_time_blk.append(f"{str(int(row['ARR_TIME']/100)-1).zfill(2)}-{str(int(row['ARR_TIME']/100)+1).zfill(2)}")
        else:
            arr_time_blk.append(f"{str(int(row['ARR_TIME']/100)).zfill(2)}-{str(int(row['ARR_TIME']/100)+2).zfill(2)}")
        
    return arr_time_blk

data20 = pd.read_csv('Jan_2020_ontime.csv')
data19 = pd.read_csv('Jan_2019_ontime.csv')

#Empty columns
data20.drop(['Unnamed: 21'], axis=1, inplace=True) 
data19.drop(['Unnamed: 21'], axis=1, inplace=True)

#concatenate data
data = pd.concat([data19, data20])

#Dept_Time have same information as DEP_TIME_BLK
data.drop(['DEP_TIME'], axis=1, inplace=True)

#Drop unecessary columns
data.drop(['OP_CARRIER_AIRLINE_ID', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'TAIL_NUM', 'OP_UNIQUE_CARRIER'], axis=1, inplace=True)

#check missing values in data
nans_lst = []
for column in data.columns.values:  
    nans_lst.append(data[column].isna().sum())
nans = pd.DataFrame({'Columns': data.columns, 'Nan_count': nans_lst})
nans['Pct_of_dataset'] = nans['Nan_count']/data.shape[0]*100
print(nans)


data2 = data.loc[:,'ARR_DEL15':'DIVERTED']
data['y'] = np.where(np.sum(data2.T)!=0, 1, 0)
data['DEP_DEL15'] = data.apply(lambda row: 1 if row['CANCELLED']==1 else row['DEP_DEL15'], axis=1)
data['DEP_DEL15'] = data.apply(lambda row: 0 if ((np.isnan(row['DEP_DEL15'])) & (row['DIVERTED']==1)) else row['DEP_DEL15'], axis=1)

#convert our "ARR_TIME" to categorical/ordinal values
data['ARR_TIME_BLK'] = time_blocks(data)
data.drop(['ARR_TIME', 'CANCELLED', 'DIVERTED', 'ARR_DEL15'], axis=1, inplace=True)

#convert types to category
cols = data.columns.tolist()
cols = cols[:-2] + cols[-1:] + cols[-2:-1]
data = data[cols]
category_columns = data.columns[2:-1]  
data[category_columns] = data[category_columns].apply(lambda x: x.astype('category'))
print(data.info())


#Column "y" store all flights that were more than 15 min. delayed, cancelled og diverted flights.
balance = data['y'].sum()/data.shape[0]
print(f"THERE ARE {balance*100:.2f} % IN THE POSITIVE CLASS AND {(1-balance)*100:.2f} % IN THE NEGATIVE CLASS")
hist_data = [balance, 1-balance]
labels = 'True', 'False'
explode = (0, 0.1)

#check imbalance
fig1, ax1 = plt.subplots()
ax1.pie(hist_data, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
plt.savefig('img/imbalance.png')
plt.show()

#calculate delays related to weekdays
week = data[['DAY_OF_WEEK','y']].groupby('DAY_OF_WEEK').sum().sort_values(by='y',ascending=False)
week['PERCENTUAL'] = week['y']/(week['y'].sum())*100
weekdays = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
labels = []
for day in week.index.values.astype(int):
    labels.append(weekdays[day])
explode = (0.1, 0, 0, 0, 0, 0, 0)

fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.set_title('DELAYS BASED ON WEEKDAY')
ax1.pie(week['PERCENTUAL'], explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
plt.savefig('img/weekday_delays.png')
plt.show()

#Delays in relation to day of month
df,max_day, min_day = statistics(data, 'DAY_OF_MONTH')
print(f"DATE WITH MOST DELAYS IS THE {df.loc[df['PERCENTUAL']==max_day].iloc[0,0]}th WITH {max_day:.0f} % OF IT'S FLIGHTS DELAYED")
print(f"DATE WITH LEAST DELAYS IS THE {df.loc[df['PERCENTUAL']==min_day].iloc[0,0]}th,  WITH {min_day:.0f} % OF IT'S FLIGHTS DELAYED")
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.set_title('DELAYS IN PERCENT IN RELATION TO TOTAL NUMBER OF FLIGHTS')
ax2.bar(df['DAY_OF_MONTH'], df['PERCENTUAL'], label='DAY OF MONTH')
ax2.plot(df['DAY_OF_MONTH'], df['MEAN'], color='r', linestyle='--', label='MEAN')
ax2.set_xlabel('Day of month', fontsize = 14)
ax2.set_ylabel('Flight delays in %', fontsize = 14)
plt.legend()

plt.savefig('img/day_of_month_delays.png')
plt.show()

#Delays in relation to carrier (company)
df,max_day, min_day = statistics(data, 'OP_CARRIER')
print(f"DATE WITH MOST DELAYS IS THE {df.loc[df['PERCENTUAL']==max_day].iloc[0,0]}th WITH {max_day:.0f} % OF IT'S FLIGHTS DELAYED")
print(f"DATE WITH LEAST DELAYS IS THE {df.loc[df['PERCENTUAL']==min_day].iloc[0,0]}th,  WITH {min_day:.0f} % OF IT'S FLIGHTS DELAYED")

df.rename(columns={"data": "TOTAL_FLIGHTS"}, inplace=True)
print("PERCENT OF FLIGHTS DELAYED PR. CARRIER")
print(df[['OP_CARRIER','TOTAL_FLIGHTS', 'PERCENTUAL']].sort_values(by='PERCENTUAL', ascending=False))
print(f"MEAN: {df['MEAN'][0]}")
fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.set_title('DELAYS BASED ON CARRIERS IN PERCENT IN RELATION TO TOTAL NUMBER OF FLIGHTS')
plt.bar(np.arange(len(df)), df['PERCENTUAL'], label='CARRIER ID')
ax1.plot(df.index.values.astype(int), df['MEAN'], color='r', linestyle='--', label='MEAN')
ax1.set_xlabel('Carrier ID', fontsize = 14)
ax1.set_ylabel('Flight delays by Carrier in %', fontsize = 14)
plt.xticks(np.arange(len(df)), rotation=45)
ax1.set_xticklabels(df['OP_CARRIER'])
plt.legend()
plt.savefig('img/dcarrier_delays.png')
plt.show()

#Delays in relation to origin
df,max_day, min_day = statistics(data, 'ORIGIN')
df.rename(columns={"data": "TOTAL_FLIGHTS"}, inplace=True)
print(df[['ORIGIN', 'TOTAL_FLIGHTS', 'PERCENTUAL']].sort_values(by='PERCENTUAL', ascending=False).head(20))
fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.set_title('DELAYS BASED ON ORIGIN IN PERCENT IN RELATION TO TOTAL NUMBER OF FLIGHTS')
plt.scatter(df['TOTAL_FLIGHTS'], df['PERCENTUAL'], label='ORIGIN')
ax1.set_xlabel('Airport size (Total number of flights operated in dataset period)', fontsize = 14)
ax1.set_ylabel('Flight delays by origin airport in %', fontsize = 14)
ax1.set_xscale('log')
plt.legend()
plt.savefig('img/origin_delays.png')
plt.show()

#Delays in relation to destination
df,max_day, min_day= statistics(data, 'DEST')
df.rename(columns={"data": "TOTAL_FLIGHTS"}, inplace=True)
print(df[['DEST', 'TOTAL_FLIGHTS', 'PERCENTUAL']].sort_values(by='PERCENTUAL', ascending=False).head(20))
fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.set_title('DELAYS BASED ON DESTINATION IN PERCENT IN RELATION TO TOTAL NUMBER OF FLIGHTS')
plt.scatter(df['TOTAL_FLIGHTS'], df['PERCENTUAL'], label='DESTINATION')
ax1.set_xlabel('Airport size (Total number of flights operated in dataset period)', fontsize = 14)
ax1.set_ylabel('Flight delays by destination airport in %', fontsize = 14)
ax1.set_xscale('log')
plt.legend()
plt.savefig('img/destination_delays.png')
plt.show()


#Model training
sn.set(style='white')
data2 = data.copy()
balance = False
transformer = transform(data2)
X_train, X_test, y_train, y_test, X_val, y_val = X_y(data2, balance)

svd = TruncatedSVD()
logistic = LogisticRegression(max_iter=10000, tol=0.1)
pipe = Pipeline(steps=[('transformer', transformer), ('svd', svd), ('logistic', logistic)])
param_grid = {
    'svd__n_components': (50, 300, 50),
    'logistic__C': [.1, 1, .01],
}

grid = GridSearchCV(pipe, param_grid, cv=5, verbose=2, n_jobs=8)
grid.fit(X_train, y_train)

log_roc_auc, log_precision, log_recall = metrics(grid, X_val, y_val, 'LOGISTIC')