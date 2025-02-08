import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources if not already present.
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize stopwords and lemmatizer.
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Cleans and preprocesses a text string by:
      - Removing HTML tags.
      - Removing non-alphabet characters.
      - Converting text to lowercase.
      - Tokenizing and lemmatizing while removing stopwords.
    """
    text = re.sub(r'<.*?>', '', text)           # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)       # Remove special characters/numbers
    text = text.lower()                          # Lowercase
    words = text.split()                         # Tokenize (simple whitespace split)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def prepare_dataframe(df, text_column='review', target_column='sentiment'):
    """
    Applies text preprocessing to the specified text column and creates a binary label column.
    Assumes that the sentiment is given as 'positive' or 'negative'.
    """
    df['cleaned_review'] = df[text_column].apply(preprocess_text)
    df['label'] = (df[target_column] == 'positive').astype(int)
    return df
