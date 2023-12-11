import difflib
def getchanges(original,corrected):
    original=original.split()
    corrected=corrected.split()
    endings="***************"
    flag=False
    changes=[]
    _temp=""
    for i in difflib.context_diff(original,corrected):
        print(i)
        print("PATA NEHI ")
        if i.startswith("---"):
            #print("start----")
            flag=True
            continue
        if flag:
            if i.strip()==endings:
                flag=False
                changes.append(_temp)
                _temp=''
                continue
            i=i.strip("+").strip("!").strip()
            _temp+=f" {i}"

    changes.append(_temp)
    return [i for i in changes if i!=""]
def bold_out(corrected,changes):
    new_sent=""
    for c in changes:
        corrected=corrected.replace(c,f"<strong>{c}</strong>")
        '''
        new_sent+=corrected[:corrected.index(c)]
        new_sent+="<strong>"+corrected[corrected.index(c):corrected.index(c)+len(c)]+"</strong>"
    new_sent+=corrected[corrected.index(c)+len(c):]
    '''
    return corrected

    
def compare(original,corrected):
    d = difflib.Differ()
    # calculate the difference between the two texts
    diff = d.compare(original.split(), corrected.split())
    # output the result
    original=[]
    changes=[]
    original_template='<span class="tag is-danger">{word}</span>'
    changes_template='<span class="tag is-success">{word}</span>'

    for d in diff:
        if d.startswith("-"):
            original.append(original_template.format(word=d.strip("-").strip()))
        elif d.startswith("+"):
            changes.append(changes_template.format(word=d.strip("+").strip()))
        elif d.startswith("?"):
            continue
        else:
           original.append(d.strip())
           changes.append(d.strip())

    return " ".join(original)," ".join(changes)