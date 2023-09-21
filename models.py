from extensions import db

class SMSMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500))
    result = db.Column(db.String(20))
