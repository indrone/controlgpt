import flask
from configFolder import LLMConfig
from utils import lang_chain_llm
import os
import pandas as pd
app=flask.Flask(__name__)
app.secret_key="ControlBuddy"

UserID_DB={"indreshb":"password",
           "admin":"admin",
           "jeremys": "password",
           "peterd":"password",
           "dilettad":"password",
           "rachelw": "password",
           "varung":"password",
           "rajaneeshk":"password"
           }



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
        delivery=flask.request.form["Delivery"]
        type_of=flask.request.form["Type"]
        priority=flask.request.form["Priority"]
        MileStone=flask.request.form["MileStone"]
        print(delivery,type_of,priority,MileStone)
        #output=model.predict(control)
        #llm_gpt_4=lang_chain_llm.ModelFactory("AzureChatOpenAI",{"temperature":0.5}).load_model()
        #model=lang_chain_llm.ModelLLMTechnical(llm_gpt_4)
        control_list=control.split("-")
        list_of_outputs=[]
        ll=['Section_Folders','Title_test case name or scenario','STEPS (Descrption of the step)','EXPECTED RESULTS','Delivery Team',	'Delivery Subteam',	'Thread','Type','PRIORITY','RICEFW','Milestone','Reference','Test Case Owner',	'Tcode'	,'TEMPLATE HEADERS']
        for idx,c in enumerate(control_list):
            if c!="":
                #print("***********",c)
                prompt="Write the Test cases and expected Results for "

                #prompt="You are an AI assistant for Manual Testing and to generate the unit test cases and execpeted resutls. "

                if str(delivery)!="Select dropdown":
                    prompt= prompt + " Delivery Team : " +str(delivery) + ", "

                if str(type_of)!="Select dropdown":
                    prompt= prompt + " Testing Type : " + str(type_of) + ", "

                if str(MileStone)!="":
                    prompt= prompt +  " Milestone : " + str(MileStone) + ", "

                if str(c)!="":
                    prompt= prompt +  " for the following requirement : " + str(c)
                print(prompt)
                steps,outcomes=model.response(prompt)
                #steps,outcomes=model.response(c)
                #c=prompt
                _steps=steps.split("#")
                _outcomes=outcomes.split("#")
                print("STEPS",_steps)
                print("EXPECTED",_outcomes)
                for s,o in zip(_steps,_outcomes):
                    if s=="" and o=="":
                        continue
                    _temp={
                        "idx": f"REQ_ID_{idx}",
                        'Section_Folders':"",
                        'Title_test case name or scenario':c,
                        "STEPS (Descrption of the step)":s,
                        "EXPECTED RESULTS":o,
                        'Delivery Team':delivery,
                        'Delivery Subteam':"",
                        'Thread':"",
                        'Type':type_of,
                        'PRIORITY':priority,
                        'RICEFW':"",
                        'Milestone':MileStone,
                        'Reference':"",
                        'Test Case Owner':"",	
                        'Tcode':""	,
                        'TEMPLATE HEADERS':""

                    }
                    list_of_outputs.append(_temp)
        df=pd.DataFrame(list_of_outputs)
        #df.columns=["Jira Id","Business Requirement","Test Steps","Expected Outcomes (should match 1:1 for each test step)"]


        df.to_csv("static/Output.csv",index=False)
        return flask.render_template("applicationv2.html",requirement=control, output=list_of_outputs) 
    return flask.render_template("applicationv2.html")



# @app.route("/application/",methods=["GET","POST"])
# def application():
#     if flask.request.method=="POST":
#         control=flask.request.form["requirement"]
#         delivery=flask.request.form["Delivery"]
#         type_of=flask.request.form["Type"]
#         priority=flask.request.form["Priority"]
#         MileStone=flask.request.form["MileStone"]
#         print(delivery,type_of,priority,MileStone)
#         #output=model.predict(control)
#         llm_gpt_4=lang_chain_llm.ModelFactory("AzureChatOpenAI",{"temperature":0.5}).load_model()
#         model=lang_chain_llm.ModelLLMTechnical(llm_gpt_4)
#         print(model)
#         control_list=control.split("-")
#         list_of_outputs=[]
#         for idx,c in enumerate(control_list):
#             if c!="":
#                 print("***********",c)
#                 prompt="You are an AI assistant for Manual Testing and to generate the unit test cases and execpeted resutls. " + "Testing Type :" + str(" E2E Testing ")+c 
#                 steps,outcomes=model.response(prompt)
                
#                 _steps=steps.split("#")
#                 _outcomes=outcomes.split("#")
#                 #print("STEPS",_steps)
#                 #print("EXPECTED",_outcomes)
#                 for s,o in zip(_steps,_outcomes):
#                     if s=="" and o=="":
#                         continue
#                     _temp={
#                         "idx": f"REQ_ID_{idx}",
#                         "control":c,
#                         "test_steps":s,
#                         "expected":o
#                     }
#                     list_of_outputs.append(_temp)
#         df=pd.DataFrame(list_of_outputs)
#         df.columns=["Jira Id","Business Requirement","Test Steps","Expected Outcomes (should match 1:1 for each test step)"]
#         df.to_csv("static/Output.csv",index=False)
#         return flask.render_template("applicationv1.html",requirement=control, output=list_of_outputs) 
#     return flask.render_template("applicationv1.html")

# @app.route("/regiterdocument/",methods=["GET","POST"])
# def regiterdocument():
#     if flask.request.method=="POST":
#         f = flask.request.files['file'] 
#         path=os.path.join("static",f.filename)
#         f.save(path) 
#         df=pd.read_excel(path)
#         columns=list(df.columns)
#         return flask.render_template("regiterdocument.html",
#                                      isSuccess=True,
#                                      columns=columns,
#                                      filename=f.filename
#                                      )
#     return flask.render_template("regiterdocument.html")


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

@app.route("/downloadExcel/")
def excel():
    return flask.send_from_directory("static","Output.csv")

if __name__=="__main__":
    #embeddings=lang_chain_llm.Embedding().load_embedding()
    embeddings=None
    llm_gpt_4=lang_chain_llm.ModelFactory("AzureChatOpenAI",{"temperature":0.5}).load_model()
    model=lang_chain_llm.ModelLLMTechnical(embeddings,llm_gpt_4)
    app.run(host='0.0.0.0',port=8000,debug=False)