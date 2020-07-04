import pandas as pd
from student_group_maps import *
from save_load_pickle import *
from datetime import datetime

trace = pd.read_csv("../trace_data/trace.csv")
trace = trace.sort_values(by=['userID', 'time'])
trace = trace[trace.assignmentID=='day2']



def trace2code(trace):
    code_data = pd.DataFrame(columns=['teamID', 'code'])
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

def code2xml(code_data):
    combined_code_data = pd.DataFrame(columns = ['combinedID', 'code'])
    old_combined_team_id = ''
    for i in code_data.index:
        team_id = code_data.at[i, 'teamID']
        combined_team_id = team_id[0:3]
        if combined_team_id != old_combined_team_id:
            if old_combined_team_id:
                new_row = {"combinedID": combined_team_id, "code": code}
                combined_code_data.loc[len(combined_code_data)]= new_row
            old_combined_team_id = combined_team_id
        code = code_data.at[i, 'code']
    save_obj(combined_code_data, 'combined_code_data', 'results')

code_data = load_obj("code_data", 'results')
code2xml(code_data)