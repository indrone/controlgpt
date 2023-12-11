def extract_doc(docs,doc_name:str,clean:bool=True):
    data_format:list=[]
    clean_list:list=["\n"]
    for page_no in range(docs.page_count):
        para:list=[]
        last_para:str=""
        starts_with:tuple=('',"(","•","¾")
        merge_next:bool=False
        merge_next_char:str=""
        for b in docs[page_no].get_text("blocks"):
            x0,y0,x1,y1,text,block_id,block_type=b
            if clean:
                for c in clean_list:
                    text=text.replace(c,"")
                    text=text.strip()
            if text.strip()=="":
                continue
            if merge_next:
                if len(para)==0:
                    para.append("")
                para[-1]=last_para+text
                last_para+=text
                op,character=merge_next_char.split("_")
                if op=="HAS":
                    if character in text:
                        merge_next=False
                        merge_next_char=""
                elif op=="ENDS":
                    if text.strip().endswith(character):
                        merge_next=False
                        merge_next_char=""
                continue

            if "(" in text and ")" not in text:
                merge_next_char="HAS_)"
                merge_next=True
            elif "[" in text and "]" not in text:
                merge_next_char="HAS_]"
                merge_next=True
            elif "{" in text and "}" not in text:
                merge_next_char="HAS_}"
                merge_next=True
            if text.startswith(starts_with):
                if len(para)==0:
                    para.append("")
                para[-1]=last_para+text
                last_para+=text
                continue
            if not text.strip().endswith("."):
                merge_next=True
                merge_next_char="ENDS_."
            elif text.strip()[0].islower():
                
                if len(para)==0:
                    para.append("")
                para[-1]=last_para+text
                last_para+=text
                continue

            para.append(text)
            last_para=text
            
        for p in para:
            data_format.append({
                "doc_name":doc_name,
                "page_no":page_no,
                "block":p
            })
    return data_format