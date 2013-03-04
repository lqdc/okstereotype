#!/usr/bin/env python
'''
@file generate_image.py
@date Sat 26 Jan 2013 01:41:38 PM EST
@author Roman Sinayev
@email roman.sinayev@gmail.com
@detail
'''
import os
import wordcloud
import numpy as np

def generate_image(profile, field):
    fonts_to_use = ["JosefinSansStd-Light.ttf", 
                    "Neucha.ttf", 
                    "Molengo-Regular.ttf", 
                    "ReenieBeanie.ttf", 
                    "Lobster.ttf"]
    assert len(fonts_to_use) == len(profile.field_dicts)
    fonts_dir = "/home/roman/Dropbox/django_practice/mysite/mysite/tag_cloud/fonts"
    # field_to_fonts = {}
    # for key, font in zip(profile.field_dicts.keys(), fonts_to_use):
    #     field_to_fonts[key] = font
    random_int = np.random.randint(len(fonts_to_use))
    # font_path = os.path.join(fonts_dir, field_to_fonts[field])
    font_path = os.path.join(fonts_dir, fonts_to_use[random_int])
    max_len = max(profile.words[field].apply(lambda x: len(x)))
    try: img = wordcloud.make_wordcloud(
                np.array(profile.words[field], dtype="S%d" % max_len), 
                np.array(profile.scores[field]), 
                font_path=font_path,
                width = 640,
                height = 480,
                )
    except Exception, e:
        print e
    return img
