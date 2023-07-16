import pandas as pd
import codecs
import wordcloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

chat_file = "chat.txt"

filecp = codecs.open(chat_file, encoding = "utf8")
df = pd.read_csv(chat_file, header = None, sep= ": ", on_bad_lines='warn')
df = df[df[1] != "<Media omitted>"]
df = df[df[1] != "This message was deleted"]

stopwords = pd.read_csv("stop_words_turkish.txt", header = None)

tf_idf_model  = TfidfVectorizer(stop_words=list(stopwords[0]),
                                lowercase=True,
                                ngram_range=(1,2),
                                token_pattern=r"(?u)\b\w\w\w\w+\b" #min 4 letter words
                                )
tf_idf_vector = tf_idf_model.fit_transform(df[1].values.astype('U'))
tfidf_weights = [(word, tf_idf_vector.getcol(idx).sum()) for word, idx in tf_idf_model.vocabulary_.items()]
print(tfidf_weights)

#without tf-idf (just tf)
"""
text = df[1].str.cat(sep=' ')
text = text.lower()
#text = text.split(" ")
#text = [x for x in text if (x not in list(stopwords[0]))]
#text = " ".join(x for x in text)

# Create and generate a word cloud image:
wc = wordcloud.WordCloud(width = 2400,
                         height = 1200,
                         colormap="Set3",
                         max_font_size=500,
                         min_word_length=4,
                         stopwords=stopwords[0]).generate(text)
"""
wc = wordcloud.WordCloud(width = 2400,
                         height = 1200,
                         colormap="Set3",
                         max_font_size=500,
                         ).fit_words(dict(tfidf_weights))
# Display the generated image:
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
