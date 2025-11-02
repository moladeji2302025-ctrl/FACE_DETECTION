# db_init.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pathlib import Path
from flask import Flask
import os

app = Flask(__name__)
db_path = Path('database')
db_path.mkdir(parents=True, exist_ok=True)
db_file = db_path / 'emotion_app.db'
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_file}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Usage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=True)  # name of user (if provided)
    image_path = db.Column(db.String(300), nullable=False)
    predicted_emotion = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Usage {self.id} {self.predicted_emotion} {self.confidence}>"

def init_db():
    db.create_all()
    print("Database created at", db_file)

if __name__ == '__main__':
    init_db()
