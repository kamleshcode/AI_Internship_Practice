from flask import Flask, render_template, request, redirect,url_for,Response,session

app = Flask(__name__)
app.secret_key = 'secret_key'

@app.route("/", methods=['GET', 'POST'])
def login():
    if request.method =="POST":
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'kamlesh' and password == '123':
            session["user"] = username #store username in session
            return redirect(url_for("welcome"))
        else:
            return "Invalid Credentials!! Try Again!"

    return render_template("login.html")

@app.route("/welcome")
def welcome():
    if "user" in session:
        return f'''
        <h2>Welcome, {session["user"]}!</h2>
        <a href={url_for("logout")}>Logout</a>
       '''
    return redirect(url_for("login"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)

