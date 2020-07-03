import pandas as pd
from student_group_maps import *
from save_load_pickle import *
from datetime import datetime

trace = pd.read_csv("../trace_data/trace.csv")
trace = trace.sort_values(by=['userID', 'time'])
trace = trace[trace.assignmentID=='day2']

code_data = pd.DataFrame(columns = ['teamID', 'code'])

old_team_id = ''
for i in tqdm(trace.index):
    if not isinstance(trace.at[i, 'code'], str):
        continue
    user_id = trace.at[i, 'userID']
    try:
        team_id = userID_teamID_maps[user_id]
    except:
        continue
    if team_id != old_team_id:
        print("team_id: ", team_id)
        if old_team_id:
            new_row = {"teamID": old_team_id, "code": code}
            code_data.loc[len(code_data)] = new_row
        old_team_id = team_id
    code = trace.at[i, 'code']

code_data = code_data.sort_values(by=['teamID'])
save_obj(code_data, 'code_data', 'results')

