from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
# Load the model and vectorizer
with open('model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    X = vectorizer.transform([user_input])
    response = model.predict(X)[0]
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
