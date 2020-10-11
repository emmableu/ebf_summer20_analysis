import pandas as pd
from student_group_maps import *
from save_load_pickle import *
from datetime import datetime

trace = pd.read_csv("../trace_data/trace.csv")
trace = trace.sort_values(by=['assignmentID', 'userID', 'time'])
trace = trace[trace.assignmentID.isin(['day2', 'day3'])]

teamID_reversed = {}
for i in userID_teamID_maps:
    teamID_reversed[userID_teamID_maps[i]] = i

def trace2code(trace):
    code_data = pd.DataFrame(columns=['assignmentID', 'teamID', 'code', 'time', "projectID"])
    old_team_id = ''
    for i in tqdm(trace.index):
        if not isinstance(trace.at[i, 'code'], str):
            continue
        user_id = trace.at[i, 'userID']
        assignment_id = trace.at[i, 'assignmentID']
        project_id = trace.at[i, 'projectID']
        time = trace.at[i, 'time']
        try:
            team_id = userID_teamID_maps[user_id]
        except:
            continue
        if team_id != old_team_id:
            print("team_id: ", team_id)
            if old_team_id:
                new_row = {"teamID": old_team_id, "assignmentID" : assignment_id, "code": code, "time": time, "projectID": project_id}
                code_data.loc[len(code_data)] = new_row
            old_team_id = team_id
        code = trace.at[i, 'code']

    code_data = code_data.sort_values(by=['assignmentID', 'teamID'])
    save_obj(code_data, 'code_data', 'results')

def code2xml_with_name(code_data):
    combined_code_data = pd.DataFrame(columns = ['combinedID', 'code'])
    old_combined_team_id = ''
    for i in code_data.index:
        team_id = code_data.at[i, 'teamID']
        print("team_id", team_id)
        # team_id = team_id[:3]
        # written_id = teamID_reversed[team_id + "_0"]
        written_id = teamID_reversed[team_id]
        written_id0 = written_id.split("/")[0]
        written_id1 = written_id.split("/")[1]
        final_id = written_id0 + "_" + written_id1

        combined_team_id = code_data.at[i, 'assignmentID'] + "_"+ final_id
        if combined_team_id != old_combined_team_id:
            if old_combined_team_id:
                new_row = {"combinedID": combined_team_id, "code": code}
                combined_code_data.loc[len(combined_code_data)]= new_row
            old_combined_team_id = combined_team_id
        code = code_data.at[i, 'code']
        f = open("results/xml_with_name_new/" + combined_team_id + ".xml", "w")
        f.write(code)
        f.close()

    save_obj(combined_code_data, 'combined_code_data_with_name', 'results')

trace2code(trace)
code_data = load_obj("code_data", 'results')
code2xml_with_name(code_data)