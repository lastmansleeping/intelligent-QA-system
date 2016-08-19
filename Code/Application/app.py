from flask import Flask, render_template, request, jsonify, make_response
import answer_question as aq

# Initialize the Flask application
app = Flask(__name__)

#Secret Key for Sessions.
app.secret_key = '\xd4}C\xa4\x03b\n\xfdo\xbc\xab\xa4\x01\x91JJ\xfe-\x8d\xc7\x04\xe0[('
def getIndex(question_type):
    # DESC, NUM, ENTY, LOC, HUM, ABBR
    lst = ['DESC', 'NUM', 'ENTY', 'LOC', 'HUM', 'ABBR']
    return lst.index(question_type)

def getNextType(current_question_type):
    global mat
    lst = mat[current_question_type]
    return lst.index(max(lst))

def getQuestionType(question):
    return 'DESC'

@app.route("/")
def hello():
	return "Hello World!"
# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/index')
def index():
    return render_template('index3.html')

@app.route("/Refresh")
def Refresh():
    global refresh
    refresh = True
    print refresh
    return render_template('index3.html')
# Route that will process the AJAX request, sum up two
# integer numbers (defaulted to zero) and return the
# result as a proper JSON response (Content-Type, etc.)
@app.route('/GetAnswer')
def GetAnswer():
    global refresh,mat,total,prev_question
    question = request.args.get('question')
    #x = questionType(prev_question)
    #y = questionType(question)
    #mat[x][y]
    #b = ['Rajshree','Gokul','Ashish']
    if refresh:
        prev_question = aq.getQuestionType(question)
        refresh = False
    else:
        current_question = aq.getQuestionType(question)
        #Update matrix
        mat[prev_question][getIndex(current_question)] += 1
        prev_question = current_question
    b = aq.getFeaturesPipeline(question)
    #print mat['DESC']
    return jsonify(result=b)

if __name__ == '__main__':
    global refresh
    global mat #matrix
    global total
    global prev_question
    mat = {'DESC': [25,30,22,29,35,8], 'NUM': [26,34,22,27,19,2], 'ENTY': [31,30,28,36,21,9], 'LOC': [23,25,18,32,24,0], 'HUM': [23,25,18,32,24,0], 'ABBR': [27,27,40,23,25,6]}
    #mat = [['DESC','NUM','ENTY','LOC','HUM','ABBR'],[25,30,22,29,35,8],['ENTY',26,34,22,27,19,2],['LOC',31,30,28,36,21,9],['HUM',23,25,18,32,24,0],['ABBR',27,27,40,23,25,6]]
    print mat
    refresh = True
    app.run(
    	port=5001,
        debug=True
    )

