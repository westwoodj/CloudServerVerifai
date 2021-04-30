# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START app]
import json
import logging
import os
import pickle
from flask import Flask, request
from google.cloud import storage
#from google.appengine.api import app_identity
import pkldata
from pkldata import TriFNClassify
import numpy as np
import private
import tweepy
#import data_process as dp
from joblib import load
from data_process import myProcessing
#import gcsfs


MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL = None

app = Flask(__name__)

d = 10  # num features

#
alpha, beta, gamma, lmbda, eta = -5, 1e-4, 10, 0.1, 1 #lmbda because lambda functions in Python

'''
alpha and beta control social relationship and user-news engagements

gamma controls publisher-partisian contribution

eta controls the input of the linear classifier

'''

n, t = 74, 52 # news, terms
r = 73 #labeled-unlabeled boundary

D = np.random.uniform(0, 1, [n, d])  # news embedding
DL = D[:r, :]  # labeled
DU = D[r:, :]  # unlabeled
p = np.random.uniform(0, 1, [d, 1])  # mapper of labeled news embedding
q = np.random.uniform(0, 1, [d, 1])  # mapper for publisher embedding




auth = tweepy.OAuthHandler(private.CONSUMER_KEY, private.CONSUMER_SECRET)
auth.set_access_token(private.OAUTH_TOKEN, private.OAUTH_TOKEN_SECRET)
api = tweepy.API(auth)






def _load_model(topic):

    MODEL_FILENAME = os.environ['MODEL_FILENAME']
    FILENAME = topic+'_'+MODEL_FILENAME
    class CustomUnpickler(pickle.Unpickler):

        def find_class(self, module, name):
            if name == 'TriFNClassify':
                from pkldata import TriFNClassify
                return TriFNClassify
            return super().find_class(module, name)


    global MODEL
    global VOCAB
    client = storage.Client()
    BUCKET = client.get_bucket(MODEL_BUCKET)
    blob = BUCKET.blob(FILENAME)
    vblob = BUCKET.blob('{}_vocab.txt'.format(topic))
    blob.download_to_filename("{}_pickledmodel.pkl".format(topic))
    vblob.download_to_filename("{}_vocab.txt".format(topic))
    #fs = gcsfs.GCSFileSystem(project='verifi-5e841', token='google_default')
    #fs.ls(MODEL_BUCKET)
    #fs.invalidate_cache()
    #with fs.open(FILENAME, 'rb') as f:
    #    MODEL = pickle.load(f)

    # Note: Change the save/load mechanism according to the framework
    # used to build the model.
    #
    #f = blob.open(mode='rb')
    MODEL = CustomUnpickler(open("{}_pickledmodel.pkl".format(topic), 'rb')).load() #pickle.loads(s)


#verifi-5e841.appspot.com/TriFN.pkl

@app.route('/', methods=['GET'])
def index():
    return str(MODEL), 200


@app.route('/predict', methods=['POST'])
def predict():
    #print(request.get_json())
    topic = request.get_json()['topic']

    '''
    if topic == 'GENERAL':
        MODEL_FILENAME = os.environ['MODEL_FILENAME']
        FILENAME1 = 'BUZZFEED' + '_' + MODEL_FILENAME
        FILENAME2 = 'COVID' + '_' + MODEL_FILENAME
        class CustomUnpickler(pickle.Unpickler):

            def find_class(self, module, name):
                if name == 'TriFNClassify':
                    from pkldata import TriFNClassify
                    return TriFNClassify
                return super().find_class(module, name)

        client = storage.Client()
        BUCKET = client.get_bucket(MODEL_BUCKET)
        blob1 = BUCKET.blob(FILENAME1)
        blob2 = BUCKET.blob(FILENAME2)

        vblob1 = BUCKET.blob('BUZZFEED_vocab.txt')
        vblob2 = BUCKET.blob('COVID_vocab.txt'.format(topic))

        blob1.download_to_filename("BUZZFEED_pickledmodel.pkl")
        vblob1.download_to_filename("COVID_vocab.txt")

        blob1.download_to_filename("BUZZFEED_pickledmodel.pkl")
        vblob1.download_to_filename("BUZZFEED_vocab.txt")

        blob2.download_to_filename("COVID_pickledmodel.pkl")
        vblob2.download_to_filename("COVID_vocab.txt")
        # fs = gcsfs.GCSFileSystem(project='verifi-5e841', token='google_default')
        # fs.ls(MODEL_BUCKET)
        # fs.invalidate_cache()
        # with fs.open(FILENAME, 'rb') as f:
        #    MODEL = pickle.load(f)

        # Note: Change the save/load mechanism according to the framework
        # used to build the model.
        #
        # f = blob.open(mode='rb')
        MODEL1 = CustomUnpickler(open("BUZZFEED_pickledmodel.pkl", 'rb')).load()  # pickle.loads(s)
        MODEL2 = CustomUnpickler(open("COVID_pickledmodel.pkl", 'rb')).load()  # pickle.loads(s)


        VOCAB1 = MODEL1.getV()
        VOCAB2 = MODEL2.getV()
        tweet_id = request.get_json()['X']
        if (tweet_id.split("/")[0] == 'http:' or tweet_id.split("/")[0] == 'https:'):
            tweet_id = tweet_id.split('/')[-1].split('?')[0]
        try:
            tweet = api.get_status(tweet_id, tweet_mode='extended')
            X1 = myProcessing(tweet.full_text, "BUZZFEED_vocab.txt")
            X2 = myProcessing(tweet.full_text, "COVID_vocab.txt")
            # print(tweet_info)
            DU1 = X1.dot(VOCAB1)
            DU2 = X2.dot(VOCAB2)

            y1 = MODEL1.predict(DU1).tolist()

            y2 = MODEL2.predict(DU2).tolist()

            ynew = y1 + y2
            ynew = np.clip(ynew, -1, 1)
            return json.dumps({'y': np.sign(ynew).tolist()[0], 'confidence': abs(ynew.tolist()[0]), 'code': 200}), 200
        except Exception as e:
            return json.dumps({'message': 'Tweet could not be accessed, please try again.',
                               'code': 400}), 400  # using a 200 code as a response to not elicit problems in Adalo.



    else:
    '''
    _load_model(topic)
    tweet_id = request.get_json()['X']
    if (tweet_id.split("/")[0] == 'http:' or tweet_id.split("/")[0] == 'https:'):
        tweet_id = tweet_id.split('/')[-1].split('?')[0]
    try:
        tweet = api.get_status(tweet_id, tweet_mode='extended')
        X = myProcessing(tweet.full_text, "{}_vocab.txt".format(topic))
        # print(tweet_info)
        DU = X.dot(VOCAB)

        y = MODEL.predict(DU).tolist()
        ynew = np.clip(y, -1, 1)
        return json.dumps({'y': np.sign(y).tolist()[0], 'confidence': abs(ynew.tolist()[0]), 'code': 200}), 200
    except Exception as e:
        return json.dumps({'message': 'Tweet could not be accessed, please try again.', 'code': 400}), 400 # using a 200 code as a response to not elicit problems in Adalo.



@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return json.dumps({'message':
    "An internal error occurred: <pre>{}</pre> See logs for full stacktrace.".format(e), "code": 400})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END app]