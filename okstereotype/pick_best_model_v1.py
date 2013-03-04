#!/usr/bin/env python
'''
@file classify_okc.py
@date Sun 11 Nov 2012 06:09:21 PM EST
@author Roman Sinayev
@email roman.sinayev@gmail.com
@detail
The program reads in a csv file in the format:
#############
UserID,age,gender,ethnicity, essay
user1,18,F,white,hello....
#############
where the csv file is tab delimited instead of commas as above and then attempts to 
predict age, gender, etc from the person's essays.
Age is predicted using regressions, and other fields are predicted
using a number of classifiers, including SVM, Logistic Regression, Perceptron, etc
'''
import sys
import numpy as np
import pandas
import re
import pylab
from pandas import DataFrame, read_csv
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.linear_model import SGDClassifier,Perceptron,LogisticRegression,LinearRegression, Ridge, Lasso, LassoCV, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier,OutputCodeClassifier,OneVsRestClassifier
from time import time

class Regress_Ages:
    '''used to test regression strategies on ages field in the OKC dataframe'''
    predictions = []
    def __init__(self, ok_loc):
        '''read in data frame and vectorize essays'''
        self.df = read_csv(ok_loc, sep="\t")
        self.train_essays, self.test_essays, self.train_target, self.test_target = self.regress_prepare("age", self.df)
        self.count_vect = TfidfVectorizer(analyzer="word", smooth_idf=True,token_pattern=u'\\b\\w+\\b',ngram_range=(1,2))
        self.train_counts_tf = self.count_vect.fit_transform(self.train_essays)
        self.test_counts_tf = self.count_vect.transform(self.test_essays)

    def regress_prepare(self, field, df, max_entries=140000):
        '''
        split data into train and test sets equally.
        Also split the target data in the provided field into
        train and test sets.
        field        -- field we are trying to determine
        df           -- data frame with profiles
        max_entries  -- maximum number of entries from all subsets of the field combined
        '''
        df = df[df[field].notnull()]
        total_len = len(df[field])
        p = np.random.permutation(total_len)
        target_array = df[field][p][:max_entries]
        train_array = df["essays"][p][:max_entries]
        if max_entries < total_len:
            total_len = max_entries
        return (train_array[:total_len/2],\
                train_array[total_len/2:],\
                target_array[:total_len/2],\
                target_array[total_len/2:],\
                )

    def add_prediction(self, prediction):
        ''' 
        add a new prediction
        prediction -- tuple consists of percent correct for each year and prediction name
        '''
        self.predictions.append(prediction)

    def calculate_baseline(self):
        '''
        use a baseline of explicit age mentions in the essays
        and assess the performance of the baseline on the test target
        '''
        pred = []
        for essay in self.test_essays:
            for i in range(18,34):
                if str(i) in essay:
                    pred.append(i)
                    break
            else:
                pred.append(np.random.randint(18,34))
        b_pred = self.get_p_correct(pred, self.test_target)
        self.add_prediction((b_pred, "Baseline"))

    def calculate_random(self):
        '''
        calculate baseline of randomly estimating ages and assess the performance
        of the baseline
        '''
        pred = np.random.random_integers(18,33,len(self.test_target))
        p_correct_rand = self.get_p_correct(pred,self.test_target)
        self.add_prediction((p_correct_rand, "Random"))

    def get_p_correct(self, pred, test_target):
        '''
        pred        -- numpy array of predicted ages
        test_target -- numpy array of real ages in the test group
        returns the percentage of correct guesses for each age
        '''
        p_correct = []
        for i in range(16):
            p_correct.append(100 * np.sum(abs(pred-test_target) <= i)/float(len(pred)))
        return p_correct

    def add_regression(self, model, name):
        '''adds a model that fits the vectorized data with the provided name
        model -- scikit-learn model
        name  -- string represenation of the name
        '''
        pred = model.fit(self.train_counts_tf,self.train_target).predict(self.test_counts_tf)
        p_correct_model = self.get_p_correct(pred, self.test_target)
        self.add_prediction((p_correct_model, name))

    def plot_predictions(self):
        '''plot data kept in the class's predictions list'''
        ind = np.arange(len(self.predictions[0][0]))
        plot_list = []
        for (r, n) in self.predictions:
            plot_list.append(ind)
            plot_list.append(r)
        ax = pylab.subplot(111)
        ax.set_ylim(0,100)
        ax.set_ylabel("% Correct")
        ax.set_xlabel("Within Range")
        lines = pylab.plot(*plot_list)
        pylab.setp(lines, linewidth=2)
        pylab.legend(map(lambda x: x[1], self.predictions),
               'upper right', shadow=True, fancybox=True)
        pylab.xticks(ind, ind)
        pylab.title("Regression Performance in Age Classification",fontsize="large")
        pylab.show()


def prune_smoking(x):
    '''prune smoking field'''
    x = str(x)
    if "No" not in x:
        return "Yes"
    else:
        return "No"

def prune_ethnicities(x):
    '''prune ethnicities field'''
    x = str(x)
    if "," in x or "/" in x or "Undeclared" in x or "Other" in x:
        return "NA"
    else:
        return x
def prune_reply(x):
    '''prune reply rate field'''
    x = str(x)
    if "often" in x or "Last" in x:
        return "NA"
    else:
        return x
def prune_bodytype(x):
    '''prune body type field'''
    x = str(x)
    if x.startswith("A little") or x.startswith("Over") or x.startswith("Full"):
        return "Overweight"
    else: return "Not Overweight"

def prune_religion(x):
    '''prune religion field'''
    x = str(x)
    if not x.startswith("Atheism"):
        return "Religious"
    else:
        return "Atheist"



def use_lowest_field_count(field, df, max_entries=1000000, min_entries=1000):
    '''
    splits the data into subsets by field and uses the smallest
    subset as the count to take randomly from each subset.

    field        -- field we are trying to determine
    df           -- data frame with profiles
    max_entries  -- maximum number of entries from all subsets combined

    returns 
    training essays - numpy array
    test essays - numpy array
    training target values - numpy array
    test target values - numpy array
    '''
    ######DELETE EXTRA FIELDS#########
    df = df[df[field].notnull()]
    if field == "smoking":
        df["smoking"] = df["smoking"].apply(prune_smoking)
    elif field =="ethnicities":
        df["ethnicities"] = df["ethnicities"].apply(prune_ethnicities)
        df = df[df[field] != "NA"]
    elif field == "replies":
        df["replies"] = df["replies"].apply(prune_reply)
        df = df[df[field] != "NA"]
    elif field == "bodytype":
        df["bodytype"] = df["bodytype"].apply(prune_bodytype)
        df = df[df[field] != "NA"]
    elif field == "religion":
        df = df[df[field] != "Agnosticism"]
        df = df[df[field] != "NA"]
        df = df[df["religion"] != "Other"]
        df[field] = df[field].apply(prune_religion)
    ##################################
    field_dict = defaultdict(int)
    for i in df[field]: field_dict[i] += 1
    for key in field_dict.keys():
        if field_dict[key] < min_entries:
        # get rid of a subset if it has less than 1000 elements
            del field_dict[key]
    possible_values = field_dict.keys()
    min_count = min(field_dict.values())
    print "field %s min value is %d" % (field, min_count), possible_values
    if min_count * len(field_dict.keys()) > max_entries:
        min_count = max_entries/len(field_dict.keys())
        print "lowering min value to %d" % (min_count)
    g_e = df.groupby(field).essays
    target_array = np.empty(0,dtype=int)
    train_array = np.empty(0)
    for i, p_val in enumerate(possible_values):
        essays = np.array(g_e.get_group(p_val))
        np.random.shuffle(essays)
        essays = essays[:min_count]
        for ind,j in enumerate(essays): essays[ind]=j.lower()#make lower case
        train_array = np.append(train_array,essays)
        this_target = np.empty(min_count,dtype=int)
        this_target.fill(i)
        target_array = np.append(target_array,this_target)
    total_len = len(target_array)
    p = np.random.permutation(total_len)
    target_array = target_array[p]
    train_array = train_array[p]
    print "done fragmenting essays"
    return (train_array[:(total_len/4)*3],\
            train_array[(total_len/4)*3:],\
            target_array[:(total_len/4)*3],\
            target_array[(total_len/4)*3:],\
            possible_values)

def get_best_params(grid_scores):
    '''return the best parameters from grid scores'''
    max_score = float(0)
    best_parameters = {}
    for i in grid_scores:
        if i[1] > max_score:
            max_score = i[1]
            best_parameters = i[0]
    return best_parameters, max_score

def show_most_informative_features(vectorizer, clf,possible_values,n=20):
    '''print the highest and lowest weighted features using coef_ inner field in the classifier'''
    if hasattr(clf, 'coef_'):
        for i in range(len(clf.coef_)):
            c_f = sorted(zip(clf.coef_[i], vectorizer.get_feature_names()))
            top = zip(c_f[:n], c_f[:-(n+1):-1])
            print "\t%s\t\t\t%s" % ("Everything Else",possible_values[i])
            print "="*30
            for (c1,f1),(c2,f2) in top:
                print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1,f1,c2,f2)
            print "="*30
            print

def get_params_for_SVM(train_set, train_set_labels):
    '''
    Perform grid search to determine the best parameters for SVM Classifier
    train_set -- the training set
    train_set_labels -- labels that correspond to the training set

    returns the best determined parameters and the mean score 
    for the classifier obtained with those parameters
    '''
    pipeline = Pipeline([
                        ('vect', TfidfVectorizer(token_pattern=u'\\b\\w+\\b')),
                        ('clf', LinearSVC())
                        ])
    parameters = {"vect__ngram_range" : ((1,2),), \
                'clf__C': (0.1, 0.3, 0.5, 0.7, 0.9, 1.0 , 1.3, 10), \
                }
    gs_clf = GridSearchCV(pipeline, parameters, n_jobs=2, pre_dispatch=2,verbose=2 )
    # score_func = metrics.f1_score
    gs_clf = gs_clf.fit(train_set, train_set_labels)
    best_parameters, mean_score = get_best_params(gs_clf.grid_scores_)
    for param_name in sorted(parameters.keys()):
        print "%s: %r" % (param_name, best_parameters[param_name])
    print "best parameters", gs_clf.best_params_
    print "best score", gs_clf.best_score_
    return best_parameters, mean_score

def get_params_for_NB(text_clf, train_set, train_set_labels):
    '''
    Perform grid search to determine the best parameters for Multinomial Naive Bayes Classifier
    text_clf -- classifier
    train_set -- the training set
    train_set_labels -- labels that correspond to the training set

    returns the best determined parameters and the mean score 
    for the classifier obtained with those parameters
    '''
    parameters = {"clf__alpha": (0.1, 0.15, 0.2), \
                "clf__fit_prior": (True, ),\
                "vect__token_pattern": (u'\\b\\w+\\b',)\
                }
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=2, cv=4,pre_dispatch=2)
    gs_clf = gs_clf.fit(train_set, train_set_labels)
    best_parameters, mean_score = get_best_params(gs_clf.grid_scores_)
    for param_name in sorted(parameters.keys()):
        print "%s: %r" % (param_name, best_parameters[param_name])
    return best_parameters, mean_score

def dumb_classify(test_essays, possible_values, field):
    '''
    use a naive baseline classification strategy to determine values for test essays
    the strategy tries to find a pattern in an essay and if the pattern is not found,
    assigns a random value to that field

    test_essays     -- essays we want to evaluate
    possible_values -- list of string answers for a field
    field           -- string field we want to classify

    returns the guessed target array and possible values key/value pairs
    '''
    bl_words = {}
    bl_words["m"] = [" man", "guy", " male"]#male
    bl_words["f"] = ["woman", "girl", "female"]#female
    bl_words["yes"] = ["[^t] smok", "cigaret"]#smoking
    bl_words["no"] = ["t smoke"]#not smoking
    bl_words["religious"] = ["god", "religious", "bible", "quran"]
    target = []
    possible_values_dict = {}
    correct = 0
    total_marked = 0
    for i,v in enumerate(possible_values):
        possible_values_dict[v] = i
    for essay in test_essays:
        modified = False
        for v in possible_values:
            if modified:
                break
            if field not in ["city", "state", "marital", "ethnicities"]:
                #use key classification for these
                try:
                    for word in bl_words[v.lower()]:
                        re_search = re.compile(word)
                        if re_search.search(essay): # word found
                            target.append(possible_values_dict[v])
                            modified = True
                            break
                except:
                    pass
            else:
                re_search = re.compile(v.lower())
                if re_search.search(essay):
                    target.append(possible_values_dict[v])
                    modified = True
                    break
        if not modified:
            target.append(np.random.randint(len(possible_values)))
    return np.array(target, dtype=int),possible_values_dict

def print_baseline_statistics(predicted, real, possible_values_dict):
    '''
    Print baseline statistics using predicted and real sets of values
    
    predicted              -- Values predicted using a baseline classifier (numpy array)
    real                   -- Real values (numpy array)
    possible_values_dict   -- Dictionary with keys corresponding to possible string answer for this field
                                and values corresponding to correct value for that key
    '''
    print '{:>30}{:>10}{:>10}'.format("precision","recall", "f1-score")
    total_values = len(possible_values_dict)
    total_precision = 0
    total_recall = 0
    total_f_score = 0
    for k in possible_values_dict:
        v = possible_values_dict[k]
        e1 = real == v
        e2 = predicted == v
        tp = np.sum(e1 & e2)
        recall = float(tp)/np.sum(e1)
        precision = float(tp)/np.sum(predicted == v)
        f_score = 2.0* (precision*recall)/(precision + recall)
        total_precision += precision
        total_recall += recall
        total_f_score += f_score
        print '{:>20}{:>10.2f}{:>10.2f}{:>10.2f}'.format(k,precision,recall,f_score)
    print '{:>20}{:>10.2f}{:>10.2f}{:>10.2f}'.format(\
            "Average",\
            total_precision/total_values,\
            total_recall/total_values,\
            total_f_score/total_values)

def binary_search(n,error,train_counts_tf,target_vals):
    '''
    get the number of features close to n within error by evaluating the SVM function
    with variable values of C and L1 distance measure to decrease/increase number of features.

    n               --  number of final features
    error           --  error within which to get the number of features
    train_counts_tf --  tf-idf transformed training counts
    target_vals     --  target values in the training set

    returns decreased/transformed train counts and Lin. SVM classifier
    '''
    c = 0.1
    lsvm = LinearSVC(C=c,penalty="l1",dual=False) 
    tc = lsvm.fit_transform(train_counts_tf, target_vals)
    features = tc.shape[1]
    if abs(features - n) < error: return tc, lsvm
    i=0
    new_c = c
    if features < n:
        while features < n:
            c = new_c
            new_c = new_c*2
            print "c %f, new_c %f, iteration %d, features %d" % (c,new_c,i, features)
            lsvm = LinearSVC(C=new_c,penalty="l1",dual=False) 
            tc = lsvm.fit_transform(train_counts_tf, target_vals)
            features = tc.shape[1]
            i+=1
    else:
        while features > n:
            c = new_c
            new_c = new_c/2
            print "c %f, new_c %f, iteration %d, features %d" % (c,new_c,i, features)
            lsvm = LinearSVC(C=new_c,penalty="l1",dual=False) 
            tc = lsvm.fit_transform(train_counts_tf, target_vals)
            features = tc.shape[1]
            i+=1
    if new_c > c:
        upper = new_c
        lower = c
    else:
        upper = c
        lower = new_c
    while abs(n - features) > error:
        middle = (upper+lower)/2
        lsvm = LinearSVC(C=middle,penalty="l1",dual=False) 
        tc = lsvm.fit_transform(train_counts_tf, target_vals)
        features = tc.shape[1]
        if features > n:
            upper = middle
        else:
            lower = middle
        print "lower %f, upper %f, iteration %d, features %d" % (lower,upper,i, features)
        i+=1
    return tc,lsvm
def classify_stuff(df, stuff):
    '''
    classify a profile field using several classifiers and print statistics
    
    df    -- data frame with profiles
    stuff -- field name as a string that needs to be classified
    '''
    # df2 = df[df["essays"].apply(lambda x: len(x.split()) > 180)]
    train_essays, test_essays, target_vals, target_test_vals, target_names = use_lowest_field_count(stuff,df,max_entries=1000000)
    count_vect = TfidfVectorizer(analyzer="word", smooth_idf=True,token_pattern=u'\\b\\w+\\b',ngram_range=(1,2))
    train_counts_tf = count_vect.fit_transform(train_essays)
    test_counts_tf = count_vect.transform(test_essays)
    ###################################
    #CLASSIFY USING ALL FEATURES
    for c in np.arange(2.5,6.0,0.5):
        start_time = time()
        clf_all = LinearSVC(C=c).fit(train_counts_tf, target_vals)
        pred = clf_all.predict(test_counts_tf)
        print "Linear SVM accuracy is", np.mean(pred == target_test_vals), "c = %f" % c
        print "time is %0.3f" % (time()-start_time)
        for fit_intercept in [True, False]:
            for intercept_scaling in [0.1,1.0, 2.0, 10.0]:
                if fit_intercept == False and intercept_scaling > 0.1:
                    break
                start_time = time()
                clf_all = LogisticRegression(
                                            C=c, 
                                            fit_intercept=fit_intercept, 
                                            intercept_scaling=intercept_scaling,
                                            ).fit(train_counts_tf, target_vals)
                pred = clf_all.predict(test_counts_tf)
                print "Logistc Regression accuracy is", \
                        np.mean(pred == target_test_vals), \
                        "c = %f" % c, \
                        "fit_intercept", fit_intercept, \
                        "intercept_scaling", intercept_scaling
                print "time is %0.3f" % (time()-start_time)
    sys.exit()
    print metrics.classification_report(target_test_vals, pred, target_names=target_names)
    # print "Most Important Features for Linear SVM:"
    # show_most_informative_features(count_vect, clf_all,target_names)
    clf_all = LogisticRegression().fit(train_counts_tf, target_vals)
    pred = clf_all.predict(test_counts_tf)
    print "Logistc Regression accuracy is", np.mean(pred == target_test_vals)
    print metrics.classification_report(target_test_vals, pred, target_names=target_names)
    clf_all = MultinomialNB().fit(train_counts_tf, target_vals)
    pred = clf_all.predict(test_counts_tf)
    print "Multinomial NB accuracy is", np.mean(pred == target_test_vals)
    print metrics.classification_report(target_test_vals, pred, target_names=target_names)
    clf_all = Perceptron(n_iter=50).fit(train_counts_tf,target_vals)
    pred = clf_all.predict(test_counts_tf)
    print "Perceptron accuracy is", np.mean(pred == target_test_vals)
    print metrics.classification_report(target_test_vals, pred, target_names=target_names)
    pred, pred_dict = dumb_classify(test_essays, target_names, stuff)
    # print "Baseline accuracy is", np.mean(pred == target_test_vals)
    # print pred_dict
    # print np.sum(pred == 1)/float(len(pred))
    # print_baseline_statistics(pred, target_test_vals, pred_dict)
    #END CLASSIFY FULL FEATURES
    #################
    #PICK THE BEST FEATURES
    #############################
    print train_counts_tf.shape
    train_counts_tf,lsvm = binary_search(7000,500, train_counts_tf, target_vals)
    test_counts_tf = lsvm.transform(test_counts_tf)
    #ch2 = SelectKBest(chi2, k=1100)
    #train_counts_tf = ch2.fit_transform(train_counts_tf, target_vals)
    #test_counts_tf = ch2.transform(test_counts_tf)
    print train_counts_tf.shape
    # DONE PICKING FEATURES
    ##############################

    clfs = [(DecisionTreeClassifier(), \
            "Decision Tree Classifier"),\
            (RandomForestClassifier(n_estimators=10),\
            "Random Forest Classifier"),\
            (ExtraTreesClassifier(n_estimators=10, max_depth=None,\
                    min_samples_split=1, random_state=0),\
            "Extra Random Trees Classifier"),\
            (Perceptron(n_iter=50),\
            "Perceptron"),\
            (SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"),\
            "Stochastic Gradient Descent"),\
            (GradientBoostingClassifier(n_estimators=100,\
                            max_depth=1, random_state=0),\
            "Gradient Boosting Classifier"),\
            (MultinomialNB(alpha=0.15),"Naive Bayes"),\
            (LinearSVC(), "Linear Kernel SVM")]
    train_counts_tf = train_counts_tf.todense()
    test_counts_tf = test_counts_tf.todense()

    for clf in clfs:
        text_clf = clf[0]
        description = clf[1]
        text_clf = text_clf.fit(train_counts_tf,target_vals) 
        predicted = text_clf.predict(test_counts_tf)
        print "accuracy is", "%0.2f" % np.mean(predicted == target_test_vals), "for %s classifier" % description
        print metrics.classification_report(target_test_vals, predicted, target_names=target_names)


if __name__=="__main__":
    #this is a csv file with profile data in the format described at the beginning
    ok_loc = "/home/roman/Dropbox/okcupid_02.tsv"

    # r = Regress_Ages(ok_loc)
    # r.calculate_baseline()
    # r.calculate_random()
    # r.add_regression(LinearRegression(), "Linear Regression")
    # r.add_regression(Lasso(alpha=0.1), "Lasso")
    # r.add_regression(ElasticNet(alpha=0.1), "Elastic Net")
    # r.plot_predictions()

    df = read_csv(ok_loc, sep="\t")
    classify_stuff(df, "gender")
    # classify_stuff(df, "religion")
    # print "="*30
   # print "State"
   # print "="*30
   # classify_stuff(df, "state")
   # print "="*30
   # print "City"
   # print "="*30
   # classify_stuff(df, "city")
   # print "="*30
   # print "Body Type"
   # print "="*30
   # classify_stuff(df, "bodytype")
   # print "="*30
   # print "Reply Rate"
   # print "="*30
   # classify_stuff(df, "replies")
   # print "="*30
   # print "Smoking"
   # print "="*30
    # classify_stuff(df, "smoking")
   # print "="*30
   # print "Ethnicity"
   # print "="*30
   # classify_stuff(df, "ethnicities")
   # print "="*30
   # print "Diet"
   # print "="*30
   # classify_stuff(df, "diet")
   # print "="*30
   # print "Relationship Status"
   # print "="*30
   # classify_stuff(df, "marital")
