import pandas as pd
import numpy as np
import datetime 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from tqdm import tqdm
import json
import joblib

#for plotting functions
import seaborn           as sns
import matplotlib.pyplot as plt
import pylab             as pl 
sns.set(rc={'figure.figsize':(14,4)})

get_col                    = lambda df,s: [i for i in df.columns if s.lower() in i.lower()]

def clean_id(df,col):
    
    values =  df[col]
    values =  [str(i) for i in values]
    values =  [s.split(".")[0].replace(",",'').strip() for s in values]
    
    return values
    
def sorted_importance(feature_names, feature_importances,n_features=20):
    
    if n_features == -1:
        n_features = len(feature_names)
        
    #feature importance plot
    pairs = zip(feature_names, feature_importances)
    pairs = sorted(pairs, key= lambda x: x[1])[::-1]
    x = [i for i,_ in pairs[:n_features]]
    y = [i for _,i in pairs[:n_features]]
    return x,y

def purity_plot(actual,pred_proba, model_type = ''):
    plt.title("Purity Plot over Predicted Probability Range " + model_type)
    sns.countplot(x=[int(i*10) for i in pred_proba], hue= list(actual))
    plt.show()
    
def plot_feature_importance(feature_names,
                            feature_importances,
                            n_features = 30,
                            model_type = ''):

    #feature importance plot
    x,y = sorted_importance(list(feature_names) ,list(feature_importances),n_features)

    plt.title(f"Top {n_features} Raw Features " + model_type)
    g = sns.barplot(x, y)
    _ = g.set_xticklabels(g.get_xticklabels(),rotation=90)
    plt.show()

def plot_original_column_feature_importance(feature_names, 
                                           feature_importances,
                                           sub_cols,
                                           start = 1,
                                           n_features= -1,
                                           n_gb_feat = 50,
                                           special_cols = ['WWAPC','APC'],
                                           model_type = ''):
    
    """Used only for random forest"""
    importance_df = pd.DataFrame({"Derived_Column" : feature_names,
                                  "Importance" :   feature_importances})

    mapper = {x: sub_cols[start:][int(x.split("_")[0][1:])] for x in feature_names if x.startswith('x')}
    
    for i in special_cols:
        mapper.update({x: i for x in feature_names if x.startswith(i)})
    
    mapper.update({x:x for x in feature_names if x not in mapper.keys()})
    
    importance_df["Original_Column"] = importance_df.Derived_Column.map(mapper)

    importance_df_gb = importance_df.groupby('Original_Column').agg({'Importance':'sum'})

    x,y = sorted_importance(importance_df_gb.index,importance_df_gb.Importance,n_features)
    
    
    plt.title(f"Top {n_gb_feat} Grouped Features " + model_type)
    g = sns.barplot(x[:n_gb_feat], y[:n_gb_feat])
    _ = g.set_xticklabels(g.get_xticklabels(),rotation=90)
    plt.show()
    
def get_reasons(forceplot,
                prediction,
                n = 4):
    
    if type(forceplot) == str:
        
        #extract dict
        s = forceplot
        assert len(s) > 0, SHAP html string has zero length"

        start = s.find("{")
        end   = (len(s) - s[::-1].find("}"))

        assert start < end, Error in SHAP html parsing"

        s = s[start:end]    
        force_plot_dict = json.loads(s)
    
    elif type(forceplot) == dict:
        assert all([i in forceplot.keys() for i in ['featureNames','features'] ]),  "dict does not have required keys"
        force_plot_dict = forceplot
    else:
        assert False, "input shap type is not str or dict"
        
    #get feature names and importance
    featureNames = force_plot_dict['featureNames']
    features     = force_plot_dict['features']

    #create data frame
    importance             = pd.DataFrame(features).T
    importance['features'] = importance.index.map(lambda x: featureNames[int(x)])
    
    #return top 4 reasons
    return importance.sort_values( by = 'effect',  ascending=  (not prediction)).head(n)
    
#use in conjunction with reason_print_format
format_map_df = pd.read_csv('format_mapping.csv')
format_map_df.set_index('Feature Name', inplace = True)

def reason_print_format(row):
    
    feature  =  row['features'] 
    value    =  row['value']    
    
    #if not(feature in format_map_df.index):
    #    return str(feature) + " = " + str(value)
        
    try:
        reason_type = format_map_df.loc[feature,'Type']
        if  reason_type  == 'dict':
            return json.loads(format_map_df.loc[feature]['Text'])[str(value).lower()]

        elif reason_type == 'string':
            return format_map_df.loc[feature]['Text'].format(str(value))

        elif reason_type == 'perc':
            return format_map_df.loc[feature]['Text'].format(int(100*value))

        elif reason_type == 'float':
            return format_map_df.loc[feature]['Text'].format(np.round(value,2))

        elif reason_type == 'int':
            return format_map_df.loc[feature]['Text'].format(int(value))
        
    except:        
        if 'float' in str(type(value)):
            value = np.round(value,2)
        return str(feature) + " value is " + str(value)
