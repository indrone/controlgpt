import os
PATH=os.getcwd()


PROMPT_INPUT_VAR=["control_desc", "test_performed"]
PROMPT_TEMPLATE="Input: {control_desc}\nOutput: {test_performed}"


FEW_SHOT_PROMPT_INPUT_VAR=["control_desc"]
FEW_SHOT_PROMPT_TEMPLATE="Input: {control_desc}\nOutput:"

PATH_TO_DB=os.path.join(PATH,"DataBase","ControlDesc2TestMapping.xlsx")

columns_to_consider={"InputColumn":"Control Description","OutputColumn":"Test Performed"}
#PATH_TO_DB="D:\KPMG\AI Project\Code\\train Data.xlsx"
