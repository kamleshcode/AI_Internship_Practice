from flask import Flask,render_template,request,jsonify,redirect,url_for
from flasgger import Swagger

app = Flask(__name__)
swagger= Swagger(app)

ai_interns=[]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/add_interns",methods=["POST"])
def add_interns():
    """
    Add a new intern
    ---
    parameters:
      - name: name
        in: formData
        type: string
        required: true
      - name: age
        in: formData
        type: string
        required: false
      - name: degree
        in: formData
        type: string
        required: false
    responses:
      200:
        description: Intern added successfully
    """
    name = request.form.get("name")
    age=request.form.get("age")
    degree=request.form.get("degree")

    ai_intern = {
        "id": len(ai_interns) + 1,
        "name": name,
        "age": age,
        "degree": degree
    }
    ai_interns.append(ai_intern)
    return redirect(url_for("view_interns"))

@app.route("/interns")
def view_interns():
    return render_template("interns.html",ai_interns=ai_interns)

@app.route("/delete/<int:id>")
def delete_interns(id):
    global ai_interns
    for ai in ai_interns:
        if ai["id"] == id:
            ai_interns.remove(ai)
            break
    return redirect(url_for("view_interns"))


if __name__ == "__main__":
    app.run(debug=True)