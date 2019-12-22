import requests
import json
import pandas as pd
from nltk.tokenize.casual import casual_tokenize
from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Word2Vec

word2vec_data = []

def tokenize_text(text):
	tokens = casual_tokenize(text)
	tokens = [word for word in tokens if word.isalpha() or word.isdigit()]
	return [word for word in tokens if word not in stopwords.words('english')]

def remove_stop_words(doc_list):
	#generator = tag_corpus(text, tag)
	tokens_list = doc_list[0][0][:-1]
	tag = doc_list[0][1]
	#print(tokens_list, tag)
	return (tokens_list, tag)

frame = pd.DataFrame(columns=['context', 'id', 'followup', 'yesno', 'answer'])

r = requests.get('https://s3.amazonaws.com/my89public/quac/train_v0.2.json')
dictionary = json.loads(r.text)
for each in dictionary['data']:
	post = {}
	post['context'] = tokenize_text(each['paragraphs'][0]['context'])[:-1]
	#post['context_tokenized'] = list(tag_corpus(post['context']))[0][0][:-1]
	post['answer'] = tokenize_text(each['paragraphs'][0]['qas'][0]['answers'][0]['text'])
	word2vec_data.append(post['context'])
	word2vec_data.append(post['answer'])
	post['id'] = each['paragraphs'][0]['id']
	post['followup'] = each['paragraphs'][0]['qas'][0]['followup']
	post['yesno'] = each['paragraphs'][0]['qas'][0]['yesno']
	post['answer'] = each['paragraphs'][0]['qas'][0]['answers'][0]['text']
	print(post)
	#break
	frame = frame.append(post, ignore_index=True)
print(word2vec_data)
word2vec_model = Word2Vec(iter=1)
word2vec_model.build_vocab(word2vec_data)
word2vec_model = Word2Vec(word2vec_data)

print(word2vec_model.wv.vocab)

word2vec_model.save('model_quac.model')
print(word2vec_model.most_similar(positive=['band']))
#print(frame)
#for 
#frame.to_csv('quac_set.csv')


# text: dictionary['data'][0]['paragraphs'][0]['context']
# id: dictionary['data'][1]['paragraphs'][0]['id']
# followup: dictionary['data'][1]['paragraphs'][0]['qas'][0]['followup']
# yesno: dictionary['data'][1]['paragraphs'][0]['qas'][0]['yesno']
# answer: dictionary['data'][1]['paragraphs'][0]['qas'][0]['answers'][0]['text']
#print(dictionary['data'][1]['paragraphs'][0]['id'])