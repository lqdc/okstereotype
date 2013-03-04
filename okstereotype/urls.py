import sys
sys.path.append("/home/roman/Dropbox/django_practice/mysite/")
sys.path.append("/home/roman/Dropbox/django_practice/mysite/mysite")

from django.conf.urls import patterns, include, url
import okstereotype.views
import settings
# from multiprocessing import Process, cpu_count
import atexit
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
import os
from queue.rpc_server import initialize

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
        (r'^$', mysite.views.get_results),
        (r'^results/$', mysite.views.display_results),
        (r'^tempresults/$', mysite.views.temp_display_results),
        (r'^errors/(short)/$', mysite.views.show_error_page),
        (r'^errors/(private)/$', mysite.views.show_error_page),
        (r'^errors/(notfound)/$', mysite.views.show_error_page),
        (r'^images/(\w+)/$', mysite.views.show_pic),
        (r'^results/lengraph.png$', mysite.views.show_xkcd),
        (r'^stats/$', mysite.views.show_stats)


    # Examples:
    # url(r'^$', 'mysite.views.home', name='home'),
    # url(r'^mysite/', include('mysite.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # url(r'^admin/', include(admin.site.urls)),
        )
if settings.DEBUG:
    urlpatterns += staticfiles_urlpatterns()

# def start_procs():
#     for i in range(cpu_count()):
#         p = Process(target=initialize)
#         p.start()
#         all_procs.append(p.pid)

# def cleanup():
#     for pid in all_procs:
#         os.kill(pid,9)
#         print "killed", pid
#     print "cleaned up"

# atexit.register(cleanup)
# all_procs = []
# start_procs()



