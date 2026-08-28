import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#preprocessing function
def text_processing(email):
    email = re.sub(r"[^a-zA-Z0-9]", " ", email.lower().strip()) #convert everything to lower case and remove punctuation
    token = word_tokenize(str(email)) #tokenise email - split text into words
    taggedwords = nltk.pos_tag(token) #part-of-speech tag - tags word as Noun, Verb etc
    words = [i for i in token if i not in STOPWORDS] #removes all stop words #words like 'the', 'a','an'
    lemmedwords = [WordNetLemmatizer().lemmatize(i) for i in words] #lemmatizes words
    clean_token=[]
    for w in lemmedwords:
        clean_token.append(w)
    email = " ".join(clean_token)  #rejoins token into string
    return email