import pandas as pd
from student_group_maps import *
from save_load_pickle import *
from datetime import datetime, date, time

trace = pd.read_csv("../trace_data/trace.csv")
trace = trace.sort_values(by=['userID', 'time'])
trace = trace[trace.assignmentID.isin(['day2', 'day3'])]


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
        # if team_id.split("_")[0] == "b":
            # print(trace.loc[i])
        if team_id != old_team_id:
            if old_team_id:
                new_row = {"teamID": old_team_id, "code": code}
                code_data.loc[len(code_data)] = new_row
            old_team_id = team_id
        code = trace.at[i, 'code']

    code_data = code_data.sort_values(by=['teamID'])
    save_obj(code_data, 'code_data', 'results')


trace2code(trace)

# trace.head()

teams = ["a_1_0", "a_1_1", "a_2_0", "a_2_1", "a_3_0", "a_3_1", "a_4_0", "a_4_1", "a_5_0", "a_5_1",
         "a_6_0", "a_6_1", "a_7_0", "a_7_1", "b_1_0", "b_1_1", "b_2_0", "b_2_1", "b_3_0", "b_3_1", "b_4_0",
         "b_4_1", "b_5_0", "b_5_1", "b_6_0", "b_6_1"]


# teams = ["a_1_0", "a_1_1"]

messageDict = {
    'EBFSummer20.iSnap.ExampleDialog.popUp', 'EBFSummer20.iSnap.ExampleDialog.Close.By.gifClick',
    'EBFSummer20.iSnap.ExampleDialog.Close.By.okClick', 'EBFSummer20.iSnap.ExampleDialog.Close.By.Gif.Overlay.crossClick',
    'EBFSummer20.iSnap.Gallery.Overlay.close', 'ExampleFeedback.logFeedback', 'EBFSummer20.iSnap.ExampleDialog.selfExplation.InputSlot.editted'
}

whenClosed = {'EBFSummer20.iSnap.ExampleDialog.Close.By.gifClick', 'EBFSummer20.iSnap.Gallery.Overlay.close',
              'EBFSummer20.iSnap.ExampleDialog.Close.By.okClick'}

trivialExp = {"\n", "Explanation...\n\n\n", "Explanation...\n\n\nT"}


finalResultsSkeleton = {'teamID': None, 'userID': None, 'gifName': None, 'timeOnExample': None, 'selfExplanation': None, 'openingTime': None, 'closingTime': None}



# Functions and other data declared/defined here
def checkLogFeedBackConditions(result, i):
    if result.at[i, "message"] == 'ExampleFeedback.logFeedback' and eval(result.at[i, "data"])['explanation'] not in trivialExp: return True
    return False

def mapper(id):
    return userID_teamID_maps[id]

def getTimeDifference(start, end):
    # This is a dummy date since we do not need the date stamp as all of our results are for day2 currently.
    # But it should be changed if need be
    _date = date(2005, 7, 14)
    datetime1 = datetime.combine(_date, pd.to_datetime(result.loc[[start]].time).dt.time[start])
    datetime2 = datetime.combine(_date, pd.to_datetime(result.loc[[end]].time).dt.time[end])
    return (datetime2 - datetime1).total_seconds()

finalResults = pd.DataFrame(columns=['teamID', 'gifName', 'timeOnExample', 'selfExplanation','openingTime', 'closingTime'])

def processData(start, end, explanation):
    finalResultsSkeleton['userID'] = result.at[start, "teamID"]
    finalResultsSkeleton['teamID'] = result.at[start, "teamID"][:3]
    finalResultsSkeleton['gifName'] = result.at[start, "data"]
    finalResultsSkeleton['timeOnExample'] = getTimeDifference(start, end)
    finalResultsSkeleton['selfExplanation'] = explanation.replace('Explanation...\n\n\n', '')
    finalResultsSkeleton['openingTime'] = result.at[start, "time"]
    finalResultsSkeleton['closingTime'] = result.at[end, "time"]
    return finalResults.append(finalResultsSkeleton, ignore_index=True)

out = trace[trace['userID'].isin(userID_teamID_maps)].reset_index()
del out['index']
del out['sessionID']
del out['browserID']
del out['serverTime']
del out['id']
del out['projectID']

out = out[out['message'].isin(messageDict)].reset_index(drop = True)

teamId = list(map(mapper, out['userID']))
out.insert(loc=0, column='teamID', value=teamId)
del out['userID']

for team in teams:

    result = out[:]
    result = result.reset_index(drop=True)
    result = result.loc[result['teamID'].isin([team])].reset_index(drop=True)
    result = result.sort_values(by=['time']).reset_index(drop=True)

    i, start = 0, 0
    while i < result.shape[0]:
        start = i
        if result.at[i, "message"] == 'EBFSummer20.iSnap.ExampleDialog.popUp':
            i += 1
            exp = 'No Explanation'
            end = 0
            while i < result.shape[0] and result.at[i, "message"] != 'EBFSummer20.iSnap.ExampleDialog.popUp':
                if checkLogFeedBackConditions(result, i):  exp = eval(result.at[i, "data"])['explanation']
                if result.at[i, "message"] in whenClosed and not end: end = i
                i += 1
            finalResults = processData(start, end, exp)
        else:
            i += 1



right, left = 0, 0
i, start = 0, 0

whenClosed = {'EBFSummer20.iSnap.ExampleDialog.Close.By.gifClick', 'EBFSummer20.iSnap.Gallery.Overlay.close',
              'EBFSummer20.iSnap.ExampleDialog.Close.By.okClick'}

trivialExp = {"\n", "Explanation...\n\n\n", "Explanation...\n\n\nT"}

while i < result.shape[0]:

    start = i
    if result.at[i, "message"] == 'EBFSummer20.iSnap.ExampleDialog.popUp':
        i += 1
        exp = 'No Explanation'
        end = 0
        while i < result.shape[0] and result.at[i, "message"] != 'EBFSummer20.iSnap.ExampleDialog.popUp':

            if result.at[i, "message"] == 'ExampleFeedback.logFeedback' and eval(result.at[i, "data"])[
                'explanation'] not in trivialExp:
                exp = eval(result.at[i, "data"])['explanation']
                # print(eval(result.at[i, "data"])['explanation'])

            if result.at[i, "message"] in whenClosed and not end:
                end = i

            i += 1

        #         end = i-1
        finalResults = processData(start, end, exp)
    else:
        i += 1

print(finalResults)
save_obj(finalResults, "final_results", "results")
