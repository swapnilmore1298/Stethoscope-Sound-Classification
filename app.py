from flask import Flask, render_template
from flask import *

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")
    
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/aboutproject")
def aboutproject():
    return render_template("aboutproject.html")

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file'] 
        b=f.filename 
        """
        for i,e in enumerate(b):
        	if(e=='.'):
        		x=b[i:]
        		break
        """
        fname = "sampleaudio"+b[-4:]
        f.save(fname)  
        return render_template("success.html", name = fname)  

@app.route('/success2', methods = ['POST'])  
def success2():  
    if request.method == 'POST':  
        f = request.files['file2'] 
        b=f.filename 
        """
        for i,e in enumerate(b):
        	if(e=='.'):
        		x=b[i:]
        		break
        """
        fname = "sampletext"+b[-4:]
        f.save(fname)  
        return render_template("success2.html", name = fname)  

@app.route('/processing')  
def processing():
    return render_template("processing.html")

if __name__ == "__main__":
    app.run(debug=True)
