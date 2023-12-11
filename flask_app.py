import flask
from configFolder import LLMConfig
from utils import lang_chain_llm
import os
import pandas as pd
app=flask.Flask(__name__)
app.secret_key="ControlBuddy"

UserID_DB={"indreshb":"password"}

@app.route("/",methods=["GET","POST"])
def login():
    isLogin=True
    if "UserLogged" in flask.session and flask.session["UserLogged"]:
        return flask.redirect(flask.url_for("application"))

    if flask.request.method=="POST":
        email=flask.request.form["email"]
        password=flask.request.form["password"]

        if email in UserID_DB:
            if password==UserID_DB[email]:
                flask.session["UserLogged"]=True
                return flask.redirect(flask.url_for("application"))
            else:
                return flask.render_template("login.html",isLogin=isLogin,
                                             errorMsg="Password is Incorrect!")

        else:
            return flask.render_template("login.html",isLogin=isLogin,
                                         errorMsg="User ID is not registered! Please contract Admin")
    return flask.render_template("login.html",isLogin=isLogin)

@app.route("/application/",methods=["GET","POST"])
def application():
    if flask.request.method=="POST":
        control=flask.request.form["requirement"]
        output=model.predict(control)
        output=output.split("·\xa0\xa0\xa0")
        output=[o for o in output if o.strip()!=""]
        return flask.render_template("application.html",requirement=control, output=output) 
    return flask.render_template("application.html")

@app.route("/regiterdocument/",methods=["GET","POST"])
def regiterdocument():
    if flask.request.method=="POST":
        f = flask.request.files['file'] 
        path=os.path.join("static",f.filename)
        f.save(path) 
        df=pd.read_excel(path)
        columns=list(df.columns)
        return flask.render_template("regiterdocument.html",
                                     isSuccess=True,
                                     columns=columns,
                                     filename=f.filename
                                     )
    return flask.render_template("regiterdocument.html")

@app.route("/logout/")
def loggout():
    flask.session["UserLogged"]=False
    return flask.redirect(flask.url_for("login"))

if __name__=="__main__":
    embeddings=lang_chain_llm.Embedding().load_embedding()
    llm_gpt_4=lang_chain_llm.ModelFactory("AzureChatOpenAI",{"temperature":0}).load_model()
    model=lang_chain_llm.ModelLLMPersitantStorage(embeddings,llm_gpt_4)
    app.run(debug=True)