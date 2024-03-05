#!/usr/bin/env python
# coding: utf-8

# # création projet flask + route api

# In[1]:
import sys
from flask import Flask, request, jsonify
from gensim.models import LdaModel
import os
import pandas as pd
import pickle

id2word = pd.read_pickle('id2word.pkl')
 


# In[2]:


# Charger le modèle LDA
lda_model_path = "lda_model.gensim"
lda_model_final = LdaModel.load(lda_model_path)


# In[3]:


lda_model_final


# In[5]:


model = lda_model_final
def predict_unsupervised_tags(text):
   """
   Predict tags of a preprocessed text
   
   Args:
       text(list): preprocessed text
       
   Returns:
       relevant_tags(list): list of tags
   """
   
   corpus_new = id2word.doc2bow({text})
   topics = lda_model_final.get_document_topics(corpus_new)
   
   #find most relevant topic according to probability
   relevant_topic = topics[0][0]
   relevant_topic_prob = topics[0][1]
   
   for i in range(len(topics)):
       if topics[i][1] > relevant_topic_prob:
           relevant_topic = topics[i][0]
           relevant_topic_prob = topics[i][1]
           
   #retrieve associated to topic tags present in submited text
   potential_tags = lda_model_final.get_topic_terms(topicid=relevant_topic, topn=20)
   
   relevant_tags = [id2word[tag[0]] for tag in potential_tags if id2word[tag[0]] in text]
   
   return relevant_tags


# # DATA TEST MODEL

# In[6]:


text_test = "Hello, I'm trying to retrieve the different extensions available on my processor using x86 assembly instructions. I've taken a look at intel's manual but I can't find any information. Some people have told me to use eax register but some others told me registers are empty by default. I'm kinda new to assembly, please help. I really want to know if my machine has some hypervisor capabilities."

predicted_tags = predict_unsupervised_tags(text_test)
print("Tags prédits non supervisés pour le texte test :", predicted_tags)


# In[ ]:




app = Flask(__name__)

@app.route('/predict_tags', methods=['POST'])
def predict_tags():
    # Récupérer les données du formulaire ou du corps de la requête
    data = request.json

    # Extraire le titre et le texte des données
    title = data.get('title', '')
    text = data.get('text', '')

    # Appeler la fonction predict_unsupervised_tags
    predicted_tags = predict_unsupervised_tags(title+ text)

    # Retourner les tags prédits au format JSON
    return jsonify({'predicted_tags': predicted_tags})

@app.errorhandler(500)
def internal_error(exception):
    print("500 error caught")
    etype, value, tb = sys.exc_info()
    print(traceback.print_exception(etype, value, tb))
    

@app.route('/', methods=['GET'])
def home():
    return "ok"
try:
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
except Exception as e:
    print(f"%tb", e)

    


# In[ ]:




