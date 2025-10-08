import nltk
nltk.download('punkt_tab')

def split_into_sentences(text):
    """
    Split text into sentences using NLTK's sentence tokenizer.
    """
    return nltk.sent_tokenize(text)