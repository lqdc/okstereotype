#!/usr/bin/env python
# -*- coding: utf8 -*- 

'''
@file results_small.py
@date Tue 15 Jan 2013 01:22:12 PM EST
@author Roman Sinayev
@email roman.sinayev@gmail.com
@detail
'''
import cPickle as pickle
from time import time
import os
import re
from pandas import DataFrame
from lxml import html
import requests
import string
import sys
from my_exceptions.essay_exceptions import ShortEssayException
from my_exceptions.essay_exceptions import PrivateProfileException
from my_exceptions.essay_exceptions import ProfileNotFoundException
from english_stop_words import ENGLISH_STOP_WORDS


class OkProfile:
    def __init__(self, username="Anonymous"):
        self.essays = ""
        self.username = username
        self.errors = []
        self.field_dicts = {'ethnicities': {'White': 0, 'Black': 1, 'Asian': 2}, 
                    'gender': {'M': 0, 'F': 1}, 
                    'bodytype': {'Not Overweight': 0, 'Overweight': 1}, 
                    'smoking': {'Yes': 0, 'No': 1}, 
                    'religion': {'Atheist': 0, 'Religious': 1}
                    }

    def scrape(self):
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:7.0.1) Gecko/20100101 Firefox/7.0.1'}
        r = requests.get('http://www.okcupid.com/profile/%s' % self.username, allow_redirects=True, headers = headers)
        js_url = u'http://www.okcupid.com/signup?cf=profile,mustlogin,redirect,meta_refresh&pass_login_redirect=/profile/%s' % self.username
        if js_url in r.content.decode("utf-8"):
            print "js_url found"
            r = requests.get(js_url, allow_redirects=True, headers=headers)
        root = html.fromstring(r.content)
        essays_accum = []
        try: 
            root.cssselect("p[id=must_login]")[0]
            #profile is private
            raise PrivateProfileException("Profile is private", profile=self)
        except IndexError:    
            for p_e in root.cssselect("p"):
                if u"have anyone by that name!Now would be a great time to look at" in p_e.text_content():
                    raise ProfileNotFoundException("Profile is not found", self)
        for i in range(0, 9):
            try:
                essays_accum.append(root.cssselect("div[id=essay_%d]" % i)[0].cssselect("div[class=text]")[0].text_content())
            except IndexError, ie:
                pass
        self.essays = u'\n '.join(essays_accum)
        self.essays = ''.join([char for char in self.essays if char in string.printable])
        if len(self.essays) == 0:
            print "BAM"
        self._check_essay()
        return self.essays

    def _check_essay(self):
        num_words = len(self.essays.split())
        num_chars = len(self.essays)
        min_words = 150
        min_chars = 600
        if num_chars < min_chars or num_words < min_words:
            raise ShortEssayException(message="Your essays are too short!", profile=self)
        self.num_words = num_words

    def feed_essay(self, essay):
        self.essays = essay
        self.num_words = len(essay.split())
        return self.essays

    def populate_profile(self, results):
        if self.num_words < 200:
            self.description = """Your essay was pretty short, 
                                so it may be difficult to draw conclusions.
                                The prediction probabilities are particularly
                                important in this case.
                                If you consider yourself to be fairly weird, 
                                or if you tend to use very uncommon words, 
                                we might be completely off.
                                """
        elif self.num_words < 500:
            self.description = """Your essay was of decent size. We 
                                should be able to tell at least a couple of 
                                things right about you.
                                """
        else:
            self.description = """Your essay was long. 
                                    You really poured your soul out or you 
                                    might just be a prolific writer. Excellent!
                                    We should do a good job predicting unless
                                    you mostly use uncommon words or are very
                                    unconventional.
                                """
        self.predictions = results[0]
        predictions_prob, matching_features = results[1], results[2]
        self.probabilities, self.words, self.scores = {},{},{}
        for key in self.field_dicts.keys():
            self.probabilities[key] = int(predictions_prob[key][0][self.field_dicts[key][self.predictions[key]]]*100)
            self.words[key] = matching_features[key]["tokens"][:30]
            self.scores[key] = matching_features[key]["scores"][:30]
        

class Predict:
    def __init__(self):
        pickled_dir = "/home/roman/Dropbox/django_practice/mysite/mysite/pickled_obj"
        self.train_dump_loc = os.path.join(pickled_dir ,"pickled_train.obj")
        self.models_dump_loc = os.path.join(pickled_dir ,"pickled_models.obj")
        self.vectorizer_dump_loc = os.path.join(pickled_dir ,"pickled_vectorizer.obj")
        self.features_loc = os.path.join(pickled_dir ,"pickled_features.obj")
        self.fields = ['gender', 'bodytype', 'religion', 'smoking', 'ethnicities', 'age']
        self.field_dicts = {'ethnicities': {'White': 0, 'Black': 1, 'Asian': 2}, 
                    'gender': {'M': 0, 'F': 1}, 
                    'bodytype': {'Not Overweight': 0, 'Overweight': 1}, 
                    'smoking': {'Yes': 0, 'No': 1}, 
                    'religion': {'Atheist': 0, 'Religious': 1}
                    }

    def load_vectorizer(self):
        f = open(self.vectorizer_dump_loc)
        self.count_vect = pickle.load(f)
        f.close()

    def load_essay_vector(self):
        f = open(self.train_dump_loc)
        self.train_counts = pickle.load(f)
        f.close()

    def load_models(self):
        f = open(self.models_dump_loc)
        self.trained_models = pickle.load(f)
        f.close()

    def load_features(self):
        f = open(self.features_loc)
        self.features_dicts = pickle.load(f)
        f.close()

    def predict_fields(self, essay):
        '''essay is a string.'''
        essay = ''.join([x if ord(x) < 128 else " " for x in essay])
        vectorized_essay = self.count_vect.transform([essay])
        predictions = {}
        predictions_probs = {}
        for field in self.fields:
            if field != "age":
                predictions_probs[field] = self.trained_models[field].predict_proba(vectorized_essay)
                pred = self.trained_models[field].predict(vectorized_essay)
                predictions[field] = [key for key, val in self.field_dicts[field].iteritems() if val == pred[0]][0]
            else:
                pred = self.trained_models[field].predict(vectorized_essay)
                predictions[field] = int(round(pred[0]))
        matching_features = self._get_matching_words(essay, predictions)
        return (predictions, predictions_probs, matching_features)

    def _find_tokens(self, essay):
        token_pattern = re.compile(u'\\b\\w+\\b')
        return token_pattern.findall(essay)

    def _bigrams(self, tokens):
        min_n, max_n = 1, 2
        new_tokens = set()
        n_tokens = len(tokens)
        for n in xrange(min_n, max_n + 1):
            for i in xrange(n_tokens - n + 1):
                new_tokens.add(u" ".join(tokens[i: i + n]))
        return list(new_tokens)

    def _get_matching_words(self, essay, predictions):
        matching_features = {}
        try:
            essay_tokens_df = DataFrame({"tokens" : self._bigrams(self._find_tokens(essay))})
            for field in self.features_dicts.keys():
                
                field_features_df = self.features_dicts[field][predictions[field]]
                matching_features[field] = field_features_df.merge(essay_tokens_df, on="tokens")
                matching_features[field]["scores"] = matching_features[field]["scores"].abs()
                matching_features[field] = matching_features[field][matching_features[field]["tokens"]
                                            .isin(ENGLISH_STOP_WORDS) == False]
                #get rid of short features
                try:
                    matching_features[field] = matching_features[field][matching_features[field]["tokens"]
                                            .apply(lambda x: len(x) > 2)]
                except IndexError as ie:
                    print ie
                    continue
        except AttributeError as a:
            print "features weren't loaded"
            print a
        return matching_features

def instantiate_predict():
    p = Predict()
    p.load_vectorizer()
    p.load_models()
    p.load_features()
    return p

if __name__ == '__main__':
    p = Predict()
    p.load_vectorizer()
    p.load_models()
    p.load_features()
    essay = u'''
            I like to take long walks to empty places.
            Getting better at browsing the internet. 
            Doing research in computational biology. love that stuff. 
            I like science because it's the only thing that makes sense to me.
            Telling a book by its cover.
            or its shoes.
            Show me your shoes, and I'll tell you who your friends are.
            Something like that.
            And books have shoes in this scenario.
            The first things people usually notice about me
            The book cover.
            books: depressing russian ones
            movies: Primer or the Coen brothers movies.
            I don't watch TV. I don't even have a TV. However, 
            Mad Men and Game of Thrones are pretty good.
            music: Mr. Gnome, Jack White
            I eat ice cream mostly.
            my Debian box, motorcycles, ice cream, wiki, M-301 
            mechanical pencil, gah... I think I need everything. 
            The sixth one would be more text boxes that ask this question.
            I'll have 5 things and a set of text boxes.
            Creating realistic models of improbable scenarios. 
            For example, how hard would an average Velociraptor 
            have to kick to break the door in my math class? 
            How many people would survive? Is sitting next to 
            the exit safer or more dangerous?
            On a typical Friday night I am
            throwing in more sub-plots into my story
            The most private thing I'm willing to admit
            I have no moral standards.
            You are not the Wicked Witch of the West.
            You are the Wicked Witch of the West!
            That would be dandy actually.
            Speak to me and don't speak softly. Talk to me and let me know.
            '''
    predictions, predictions_prob, matching_features = p.predict_fields(essay)
    # print predictions, predictions_prob, matching_features
    
