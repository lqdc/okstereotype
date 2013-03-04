#!/usr/bin/env python
'''
@file views.py
@date Wed 09 Jan 2013 10:00:34 PM EST
@author Roman Sinayev
@email roman.sinayev@gmail.com
@detail
Views For StereotypeMe! App
'''

from django.http import Http404,  HttpResponseRedirect, HttpResponse
from django.template import RequestContext
from forms import EssayForm, UsernameForm
from django.shortcuts import render_to_response
from results_small import OkProfile
from my_exceptions.essay_exceptions import ShortEssayException
from my_exceptions.essay_exceptions import PrivateProfileException
from my_exceptions.essay_exceptions import ProfileNotFoundException
import my_exceptions
import okstereotype.urls
from time import sleep, time
from tag_cloud.generate_image import generate_image
from xkcd_graphs.generate_xkcd import plot_essay_len
from queue.rpc_client import EssayRpcClient
import cPickle as pickle

def get_results(request):
    if request.method != 'POST':
        essay_form = EssayForm()
        username_form = UsernameForm()
        return render_to_response('essay_form.html', 
                            {'username_form': username_form, "essay_form" : essay_form}, 
                            context_instance=RequestContext(request))
    else:
        return display_essays(request)

def display_essays(request):
    if "use_essay" in request.POST:
        essay_form = EssayForm(request.POST)
        username_form = UsernameForm()
        if essay_form.is_valid():
            message = essay_form.cleaned_data["message"]
            profile = OkProfile()
            profile.feed_essay(message)
            request.session["profile"] = profile
            return render_to_response("essay_form.html", 
                                    {'profile':profile, "show_essay": True},
                                    context_instance=RequestContext(request))
        else:
            return render_to_response('essay_form.html', 
                                {'username_form': username_form, "essay_form" : essay_form}, 
                                context_instance=RequestContext(request))
    elif "use_username" in request.POST:
        username_form = UsernameForm(request.POST)
        essay_form = EssayForm()
        if username_form.is_valid():
            username = username_form.cleaned_data["username"]
            profile = OkProfile(username)
            try:
                profile.scrape()
                request.session["profile"] = profile
            except PrivateProfileException:
                return HttpResponseRedirect('/errors/private/')
            except ShortEssayException:
                return HttpResponseRedirect('/errors/short/')
            except ProfileNotFoundException:
                return HttpResponseRedirect('/errors/notfound/')
            return render_to_response("essay_form.html", 
                                        {'profile':profile, "show_essay": True}, 
                                        context_instance=RequestContext(request))
        else:
            return render_to_response('essay_form.html', 
                            {'username_form': username_form, "essay_form" : essay_form}, 
                            context_instance=RequestContext(request))
    else:
        raise Http404

def display_results(request):
    if "submit_essay" in request.POST:
        start_time = time()
        try:
            profile = request.session["profile"]
        except KeyError as ke:
            raise Http404
        essay_client = EssayRpcClient()
        results = essay_client.call(profile.essays)
        profile.populate_profile(results)
        request.session["profile"] = profile
        print time() - start_time
        # with open("/home/roman/Dropbox/django_practice/mysite/mysite/pickled_obj/tmp_results.obj", "w") as f:
        #     pickle.dump(results, f, protocol=2)
        for key in profile.predictions.keys():
            if key == "smoking":
                if profile.predictions[key] == "Yes":
                    profile.predictions[key] = "Smoker"
                else:
                    profile.predictions[key] = "Nonsmoker"
            elif key == "bodytype":
                if profile.predictions[key] == "Overweight":
                    profile.predictions[key] = "Plump"
                else:
                    profile.predictions[key] = "Lean"
            elif key == "gender":
                if profile.predictions[key] == "M":
                    profile.predictions[key] = "Guy"
                else:
                    profile.predictions[key] = "Girl"
            elif key == "religion":
                if profile.predictions[key] == "Atheist":
                    profile.predictions[key] = "Freethinker"
                else:
                    profile.predictions[key] = "Spiritual"     
        return render_to_response("results.html", 
                    {"profile" : profile}, 
                    context_instance=RequestContext(request))

def temp_display_results(request):
    profile = request.session["profile"]
    profile.feed_essay("hi there hello")
    with open("/home/roman/Dropbox/django_practice/mysite/mysite/pickled_obj/tmp_results.obj", "r") as f:
        results = pickle.load(f)
    profile.populate_profile(results)
    request.session["profile"] = profile
    for key in profile.predictions.keys():
        if key == "smoking":
            if profile.predictions[key] == "Yes":
                profile.predictions[key] = "Smoker"
            else:
                profile.predictions[key] = "Nonsmoker"
        elif key == "bodytype":
            if profile.predictions[key] == "Overweight":
                profile.predictions[key] = "Plump"
            else:
                profile.predictions[key] = "Lean"
        elif key == "gender":
            if profile.predictions[key] == "M":
                profile.predictions[key] = "Guy"
            else:
                profile.predictions[key] = "Girl"
        elif key == "religion":
            if profile.predictions[key] == "Atheist":
                profile.predictions[key] = "Freethinker"
            else:
                profile.predictions[key] = "Spiritual"
    return render_to_response("results.html", 
                    {"profile" : profile}, 
                    context_instance=RequestContext(request))

def show_pic(request,field):
    profile = request.session["profile"]
    img = generate_image(profile, field)
    response = HttpResponse(mimetype="image/png")
    img.save(response, "PNG")
    return response

def show_xkcd(request):
    profile = request.session["profile"]
    fig = plot_essay_len(profile.num_words)
    response = HttpResponse(content_type='image/png')
    fig.savefig(response, format="png")
    return response


def show_error_page(request, reason):
    print "showing errors"
    if reason == "private":
        explanation = "The profile is private. Go make it public if you want this to work."
    elif reason == "short":
        explanation = "The profile is too short. Make it longer or go back and submit a separate essay"
    elif reason == "notfound":
        explanation = "The profile doesn't exist.  At least we don't think it does."
    return render_to_response("errors.html", {"explanation" : explanation}, context_instance = RequestContext(request))

def show_stats(request):
    return render_to_response("stats.html", context_instance=RequestContext(request))
