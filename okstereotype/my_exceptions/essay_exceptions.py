#!/usr/bin/env python
# -*- coding: utf8 -*-

'''
@file essay_exceptions.py
@date Wed 16 Jan 2013 02:01:59 PM EST
@author Roman Sinayev
@email roman.sinayev@gmail.com
@detail
Exceptions classes are in this file
'''

class ShortEssayException(Exception):
    def __init__(self, message, profile):
        Exception.__init__(self, message)
        self.profile = profile
        profile.errors.append(message)

class PrivateProfileException(Exception):
    def __init__(self, message, profile):
        Exception.__init__(self, message)
        self.profile = profile
        profile.errors.append(message)

class ProfileNotFoundException(Exception):
    def __init__(self, message, profile):
        Exception.__init__(self, message)
        self.profile = profile
        profile.errors.append(message)
