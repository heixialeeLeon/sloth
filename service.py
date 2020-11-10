from flask import Flask, request, Response
import base64

app = Flask(__name__)

@app.route("/webtest", methods=['get', 'post'])
def web_test():
    return "<h1 style='color:blue'>Hello WebTest!</h1>"