#!/usr/bin/env python
# RABBIT MQ-based RPC Server

import logging
import sys
import pika
from results_small import instantiate_predict
import cPickle as pickle
import atexit
import os

class RPCServer:
    def __init__(self):
        self.pred = instantiate_predict()
        logging.basicConfig(filename="./logs/output_%d.log" % os.getpid() , filemode="w", level=logging.INFO)
        #atexit.register(logging.shutdown())
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(self.on_request, queue='rpc_queue')
        print " [x] Awaiting RPC requests"
        self.channel.start_consuming()
        
    def on_request(self, ch, method, props, body):
        logging.info("Essay:\n\n %s" % body)
        predictions, predictions_prob, matching_features = self.pred.predict_fields(body)
        logging.info(predictions)
        response = pickle.dumps([predictions, predictions_prob, matching_features],protocol=2)
        ch.basic_publish(exchange='',
                 routing_key=props.reply_to,
                 properties=pika.BasicProperties(correlation_id = \
                                                props.correlation_id),
                 body=response)
        ch.basic_ack(delivery_tag = method.delivery_tag)

def initialize():
    RPCServer()
if __name__=="__main__":
    RPCServer()

