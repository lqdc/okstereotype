#!/usr/bin/env python
'''
@file pick_best_model.py
@date Mon 04 Mar 2013 12:16:53 AM EST
@author Roman Sinayev
@email roman.sinayev@gmail.com
@detail
Here we attempt to pick the best model for classification of okc data.
The model is later used in results_small.py to predict some personal attributes.
'''

import lxml
import requests
from pandas import read_csv
import numpy as np
from collections import defaultdict
from sklearn.linear_model import SGDClassifier,Perceptron,LogisticRegression,LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import cPickle as pickle
from time import time
import sys
from pprint import pprint

CSVLOCATION="~/" #pandas dataframe with scraped data from okcupid

class OkProfile:
    def __init__(self, username):
        self.essays = ''
        self.username = username
    def scrape(self):
        r = requests.get('http://www.okcupid.com/profile/%s' % self.username)
        root = lxml.html.fromstring(r.content)
        essays = []
        for i in range(1, 9):
            try:
                essays.append(root.cssselect("div[id=essay_%d]" % i)[0].cssselect("div[class=text]")[0].text_content())
            except IndexError, ie:
                pass
        self.essays = '\n'.join(essays)
        return self.essays

class Arrays:
    def __init__(
        self, 
        fields = [
                 'age',
                 'gender', 
                 'bodytype', 
                 'religion', 
                 'smoking', 
                 'ethnicities',
                 ], 
        csv_location = CSVLOCATION
        ):
        '''
        min_entries = lowest number of items allowed per field
        fields = list with fields to target
        csv_location = location of the csv file
        '''
        self.fields = fields
        self.csv_location = csv_location
        self.min_entries = 2000

    def _get_classification_targets(self,df, index_dict):
        '''
        creates target arrays that represent all unique values as integers
        creates the mappings between values and the integers
        
        parameters:
        df = the data frame with all the data
        index_dict = the dictionary with correct index values for a given field key
        min_entries = lowest number of items allowed per field

        returns:
        target_dicts = target array dictioinary with ints corresponding to unique values
        field_dicts = mapping between values and target arrays' ints
        '''
        field_dicts = {} #mapping between fields and field_dict dictionaries
        target_dicts = {}
        for field in self.fields:
            if field == "age":
                target = df.ix[index_dict[field]][field]
                target_dicts[field] = np.array(target)
                continue
            field_dict = {}
            target = np.empty(len(df), dtype=int)            
            good_items_grouped = df.ix[index_dict[field]].groupby(field)
            # for group in good_items_grouped.groups:
                # print group, len(good_items_grouped.groups[group])
            for i, key in enumerate(good_items_grouped.indices.keys()):
                #create the mapping between good keys in a field and target
                #also populate the target with values
                field_dict[key] = i
                target[good_items_grouped.groups[key]] = i
            field_dicts[field] = field_dict
            target_dicts[field] = target
        return target_dicts, field_dicts

    def _lower_values(self, df, index_dict):
        '''
        lowers df items to the lowest entry in the field where the entry is larger than min_entries

        this is necessary because models should be trained on data with equal representation
        from all classes.
        
        parameters:
        df = the data frame with all the data
        index_dict = the dictionary with correct index values for a given field key

        returns:
        index_dict = modified index dict that only has good index values for each field key
        '''
        for field in self.fields:
            if field == "age":
                continue
            value_counts = df[field].value_counts()
            while min(value_counts) < self.min_entries: #delete lowest index if it is too low
                index_dict[field] = np.setdiff1d(
                                                index_dict[field],
                                                df[df[field] == value_counts.idxmin()].index,
                                                assume_unique=True
                                                ) 
                value_counts = value_counts.drop(value_counts.idxmin())
            min_value = min(value_counts)
            good_items_grouped = df.ix[index_dict[field]].groupby(field)
            for value in value_counts.index: #lower all values to the lowest field count
                index_dict[field] = np.setdiff1d(
                                            index_dict[field], 
                                            good_items_grouped.groups[value][min_value:], 
                                            assume_unique=True
                                            )
        return index_dict

    def _prune_smoking(self, x):
        '''prune smoking field'''
        x = str(x)
        if x == 'nan': return np.nan
        elif "No" not in x: return "Yes"
        else: return "No"

    def _prune_ethnicities(self,x):
        '''prune ethnicities field'''
        x = str(x)
        if x == 'nan': return np.nan
        elif "," in x or "/" in x or "Undeclared" in x or "Other" in x:
            return np.nan
        else:
            return x
    def _prune_reply(self,x):
        '''prune reply rate field'''
        x = str(x)
        if x == 'nan': return np.nan
        elif "often" in x or "Last" in x:
            return np.nan
        else:
            return x
    def _prune_bodytype(self,x):
        '''prune body type field'''
        x = str(x)
        if x == 'nan': return np.nan
        elif x.startswith("A little") or x.startswith("Over") or x.startswith("Full"):
            return "Overweight"
        else: return "Not Overweight"

    def _prune_religion(self,x):
        '''prune religion field'''
        x = str(x)
        if x == 'nan' or x == "Agnosticism" or x == "Other": 
            return np.nan
        elif not x.startswith("Atheism"):
            return "Religious"
        else:
            return "Atheist"

    def _delete_extra_fields(self, df):
        '''prune unacceptable values and convert them to acceptable and returns index of correct values'''
        index_dict = {}
        for field in self.fields:
            if field == "smoking":
                df["smoking"] = df["smoking"].apply(self._prune_smoking)
            elif field =="ethnicities":
                df["ethnicities"] = df["ethnicities"].apply(self._prune_ethnicities)
            elif field == "replies":
                df["replies"] = df["replies"].apply(self._prune_reply)
            elif field == "bodytype":
                df["bodytype"] = df["bodytype"].apply(self._prune_bodytype)
            elif field == "religion":
                df[field] = df[field].apply(self._prune_religion)
            index_dict[field] = df[df[field].notnull()].index
        return index_dict

    def _delete_stuff_because_no_memory(self, df):
        df = df[df["gender"].notnull()]
        number_f = len(df[df["gender"] == "F"])
        number_m = len(df[df["gender"] == "M"])
        df = df.drop(df[df["gender"] == "M"][number_f:].index)
        df.index = np.arange(len(df))
        return df
    def _check_mapping(self,index_dict, targets_dict, field_dicts, df):
        for field in self.fields:
            if field == "age": continue
            gender_index = index_dict[field]
            gender_target = targets_dict[field]
            for i in gender_index:
                assert gender_target[i] == field_dicts[field][df[field].ix[i]]
        print "passed"

    def get_train_and_target(self):
        '''
        create train arrays, targets, mapping between them and an index of good values in the
        dataframe for each field

        returns 
        train array = the training sparse matrix 
        targets_dict = the target arrays with known values corresponding to the fields of interest 
        field_dicts = mapping between field values and target values
        index_dict = mapping between field values and good indices for each field
        '''
        df = read_csv(self.csv_location, sep="\t")
        # df.essays = df["essays"].apply(lambda x: x.lower()) #lower case essays
        p = np.random.permutation(len(df))
        df = df.ix[p]
        # df = self._delete_stuff_because_no_memory(df)
        p = np.random.permutation(len(df))
        df = df.ix[p]
        test_df = df[len(df)/2:]
        test_df.index = np.arange(len(test_df))
        df = df[:len(df)/2]
        df = df[df["essays"].apply(lambda x: len(x.split()) > 270)]
        df.index = np.arange(len(df))
        print len(df)
        index_dict = self._delete_extra_fields(df)
        test_index_dict = self._delete_extra_fields(test_df)
        index_dict = self._lower_values(df, index_dict)#lower values to lowest acceptable in a field
        test_index_dict = self._lower_values(test_df, test_index_dict)
        targets_dict, field_dicts = self._get_classification_targets(df, index_dict)
        test_targets_dict, test_field_dicts = self._get_classification_targets(test_df, test_index_dict)
        # self._check_mapping(index_dict, targets_dict, field_dicts, df)
        assert test_field_dicts == field_dicts
        print field_dicts
        train_essays = df["essays"]
        test_essays = test_df["essays"]
        return train_essays, targets_dict, field_dicts, index_dict, test_essays, test_targets_dict, test_index_dict

class TrainAndModel:
    def __init__(self):
        self.models_dict = {
                        'age' : [LinearRegression],
                        'gender' : [LinearSVC, LogisticRegression], 
                        'bodytype': [LinearSVC, LogisticRegression], 
                        'religion': [LinearSVC, LogisticRegression], 
                        'smoking' : [LinearSVC, LogisticRegression],  
                        'ethnicities': [LinearSVC, LogisticRegression],
                        }
        self.train_dump_loc = "/home/roman/pickled_train.obj"
        self.models_dump_loc = "/home/roman/pickled_models.obj"
        self.vectorizer_dump_loc = "/home/roman/pickled_vectorizer.obj"

    def vectorize_essays(self):
        self.arrays = Arrays()
        self.train_essays, \
        self.targets_dict, \
        self.field_dicts, \
        self.index_dict ,\
        self.test_essays, \
        self.test_targets,\
        self.test_index_dict = self.arrays.get_train_and_target()
        count_vect = TfidfVectorizer(analyzer="word", smooth_idf=True,token_pattern=u'\\b\\w+\\b',ngram_range=(1,2))
        self.train_counts = count_vect.fit_transform(self.train_essays)
        self.test_counts = count_vect.transform(self.test_essays)

    def dump_vectorizer(self,vectorizer):
        f = open(self.vectorizer_dump_loc, "w")
        pickle.dump(vectorizer, f)
        f.close()

    def load_vectorizer(self):
        f = open(self.vectorizer_dump_loc)
        pickle.load(f)
        f.close()

    def dump_essay_vector(self):
        f = open(self.train_dump_loc, "w")
        pickle.dump(self.train_counts, f)
        f.close()

    def load_essay_vector(self):
        f = open(self.train_dump_loc)
        self.train_counts = pickle.load(f)
        f.close()

    def make_models(self):
        best_vals = {}
        for field in self.arrays.fields:
            if field == "age":
                continue
            best_vals[field] = {}
            field_models = self.models_dict[field]
            field_index = self.index_dict[field]
            field_target = self.targets_dict[field][field_index]
            field_dict = self.field_dicts[field]
            field_counts = self.train_counts[field_index,:]

            test_field_index = self.test_index_dict[field]
            test_field_target = self.test_targets[field][test_field_index]
            test_field_counts = self.test_counts[test_field_index,:]

            print "----------"
            print field
            print "----------"
            for model in field_models:
                best_vals[field][model] = {"accuracy": 0, "c" : 0} #first is accuracy, second is c value
                for c in np.arange(0.1,5.5,0.5):
                    if model == LogisticRegression:
                        this_model = model(C=c, fit_intercept=False, dual=True)
                    else:
                        if c > 2.0:
                            continue
                        this_model = model(C=c)
                    start_time = time()
                    clf = this_model.fit(field_counts, field_target)
                    pred = clf.predict(test_field_counts)
                    accuracy = np.mean(pred == test_field_target)
                    if accuracy > best_vals[field][model]["accuracy"]:
                        best_vals[field][model]["accuracy"] = accuracy
                        best_vals[field][model]["c"] = c
                        print "best so far", best_vals[field][model]["accuracy"]
                        print this_model
                    print "accuracy", accuracy
                    print "c", c
                    print "time", time() - start_time
                    # self.clfs[field].append(this_model)
        for key in best_vals.keys():
            print key
            print best_vals[key]
    def dump_models(self):
        f = open(self.models_dump_loc, "w")
        pickle.dump(self.clfs, f)
        f.close()        

    def load_models(self):
        f = open(self.models_dump_loc)
        self.clfs = pickle.load(f)
        f.close()
if __name__ == '__main__':
    t_m = TrainAndModel()
    t_m.vectorize_essays()
    t_m.make_models()
