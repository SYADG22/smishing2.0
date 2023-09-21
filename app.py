from flask import Flask, render_template, request, url_for, jsonify
import pickle  # or the appropriate library for loading your model
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from extensions import db
from models import SMSMessage
import os

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
#'postgres://jskteoiomwdbsd:8deac3f1a77e50bebca9b3fac1cba5ea596d7074e3808778da6d55ee55b075a8@ec2-34-236-103-63.compute-1.amazonaws.com:5432/dah1ks5gv211h4'

db.init_app(app)

ps = PorterStemmer()

def transform_text(text):
     #Lowercase
    text = text.lower()
    
    #Tokenization
    text = nltk.word_tokenize(text)
    
    #alpha numerical values
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    #Removing stopwords
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    #Stemming
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)  

#2 Loading model and vectorizer

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')

#3 Getting text from user input

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        msg = request.form['textHere']

        transform_text(msg)

        data = [msg]

        vect = tfidf.transform(data)

        result = model.predict(vect)
    
    if result == 1:
        db_msg = 'smishing'
    else:
        db_msg = 'legit'
    new_message = SMSMessage(text=msg, result=db_msg)
    db.session.add(new_message)
    db.session.commit()
    
#4. Display result on html page

    if result == 1:
        return render_template("index.html", data="Smishing")
    else:
        return render_template("index.html", data="Legit")

#Point for API documentation
@app.route('/API')
def API():
    return render_template('APIDocumentation.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/detect', methods=['GET'])
def detect():
    
    data = request.get_json()
    if 'sms_message' not in data:
        return jsonify({'error': 'Missing "sms_message" field in the request'}), 400
    
    sms_message = data
    transform_text(sms_message)

    # Preprocess the SMS message
    preprocessed_sms = [sms_message]

    vect = tfidf.transform(preprocessed_sms)
    
    # Make predictions using the pre-trained model
    prediction = model.predict(vect)[0]
    
    result = "Smishing" if prediction == 1 else "Legitimate"
    
    # Store the SMS message and result in the database
    new_message = SMSMessage(text=preprocessed_sms, result=result)
    db.session.add(new_message)
    db.session.commit()
    
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)
