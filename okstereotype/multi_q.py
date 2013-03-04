#!/usr/bin/env python
'''
@file multi_q.py
@date Sat 19 Jan 2013 10:33:37 AM EST
@author Roman Sinayev
@email roman.sinayev@gmail.com
@detail
OBSOLETE!!!!!! 
USE RPC server instead.  This does not work 
well with Scikit-learn's CountVectorizer because of memory allocation
for each new process
'''
from multiprocessing import Process, Queue, cpu_count, current_process, Array
from Queue import Empty
import os
from time import sleep,time
from results_small import instantiate_predict
import atexit

class MultiQ:
    def __init__(self):
        self.essay_q = Queue() #queue of incomping essays
        self.process_q = Queue(cpu_count()) #queue of processes
        self.out_q = Queue() #outgoing queue with results
        self.p_dict = {} # essay to processes mapping
        self.results = {} # essay to results mapping
        self.pred = instantiate_predict()
        self.all_procs = []
        atexit.register(self.cleanup)

    def enq_essay(self,essay):
        self.essay_q.put(essay)
        p = Process(target=self.compute_results, args=(essay,))
        p.start()
        self.p_dict[essay] = p
        self.all_procs.append(p.pid)
    

    def compute_results(self, essay):
        start_time = time()
        self.process_q.put(1)
        self.essay_q.get()
        predictions, predictions_prob, matching_features = self.pred.predict_fields(essay)
        result = [predictions, predictions_prob, matching_features]
        self.out_q.put((essay, result))
        self.process_q.get()
        print "done took %0.2f sec" % (time() - start_time)

    def check_size(self):
        return self.essay_q.qsize() 
        
    def check_status(self, essay):
        try:
            return self.p_dict[essay].is_alive()
        except KeyError:
            return False

    def get_results(self, essay):
        try:
            r = self.results[essay]
            del self.results[essay]
            return r
        except KeyError: #essay isn't done yet
            pass
        try: 
            e, r = self.out_q.get_nowait()
            self.p_dict[e].terminate()
            self.all_procs.remove(self.p_dict[e].pid)
            del self.p_dict[e]
        except Empty as em:
            return None
        if essay == e:
            return r
        else:
            self.results[e] = r

    def cleanup(self):
        timeout_sec = 5
        for pid in self.all_procs: # list of your processes
            os.kill(pid, 9) # supported from python 2.6
            print 'killed', pid
        print "cleaned up"
def main():
    m_q = MultiQ()
    for essay in [str(i) for i in range(1)]:
        m_q.enq_essay(essay)

if __name__ == '__main__':
    main()
