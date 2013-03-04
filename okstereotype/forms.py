#!/usr/bin/env python
# -*- coding: utf8 -*- 
'''
@file forms.py
@date Thu 10 Jan 2013 06:25:39 PM EST
@author Roman Sinayev
@email roman.sinayev@gmail.com
@detail
'''

from django import forms
import string
import re

class EssayForm(forms.Form):
    message = forms.CharField(widget=forms.Textarea(attrs={'placeholder': 'Paste essays here', 
                                                            "style": "width: 550px;", 
                                                            "id" : "essay_field1"}))
    def clean_message(self):
        message = self.cleaned_data["message"]
        num_words = len(message.split())
        min_words = 150
        max_chars = 30000
        min_chars = 600
        if num_words < min_words:
            raise forms.ValidationError("Not enough words!  Need at least %d" % min_words)
        if len(message) > max_chars:
            raise forms.ValidationError("Essay too long (over %d)" % max_chars)
        elif len(message) < min_chars:
            raise forms.ValidationError("Essay too short (under %d)" % min_chars)
        line_bools = map(
                lambda line_token: 
                    any([x not in set(string.printable) for x in line_token]), 
                    message.split("\n")
                        )
        message = u' '.join([y for x,y in zip(line_bools, message.split("\n")) if x is False])
        headlines = [u"Girls who like guys", 
            u"Bi girls only",
            u"Girls who like guys"
            u"Straight girls only",
            u"Guys who like girls",
            u"Bi guys only",
            u"Straight guys only",
            u"\\bAges \d{2}.\d{2}\\b",
            u"My self-summary",
            u"What I’m doing with my life",
            u"I’m really good at",
            u"The first things people usually notice about me",
            u"Favorite books, movies, shows, music, and food",
            u"The six things I could never do without",
            u"I spend a lot of time thinking about",
            u"On a typical Friday night I am",
            u"The most private thing I’m willing to admit",
            u"I’m looking for",
            u"You should message me if",
            u"For new friends",
            u"[Ss]hort-term dating",
            u"[Ll]ong-term dating",
            u"Who are single",
            ]
        for headline in headlines:
            searcher = re.compile(u"(.*)" + headline + u"(.*)")
            try:
                message = u' '.join(searcher.match(message).groups())
            except AttributeError as a:
                pass
        return message

class UsernameForm(forms.Form):
    username = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'placeholder': 'Username', 'style': 'font-size: 35px; height: 55px; width: 400px;', 'id':"username_field1"}))
    #  width: 380px;
    def clean_username(self):
        username = self.cleaned_data["username"]
        max_chars = 100
        num_chars = len(username)
        if num_chars > max_chars:
            raise forms.ValidationError("Username too long")
        elif num_chars < 1:
            raise forms.ValidationError("Please Enter Username")
        return username

class EssayForm2(forms.Form):
    message = forms.CharField(widget=forms.Textarea(attrs={"style": "width: 550px;", 
                                                            "id" : "essay_field2",
                                                            "readonly": "readonly",
                                                            }))
    def clean_message(self):
        message = self.cleaned_data["message"]
        num_words = len(message.split())
        min_words = 150
        max_chars = 30000
        min_chars = 600
        if num_words < min_words:
            raise forms.ValidationError("Not enough words!  Need at least %d" % min_words)
        if len(message) > max_chars:
            raise forms.ValidationError("Essay too long (over %d)" % max_chars)
        elif len(message) < min_chars:
            raise forms.ValidationError("Essay too short (under %d)" % min_chars)
        return message