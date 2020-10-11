import pandas as pd
from save_load_pickle import *
from datetime import datetime, date, time

trace = pd.read_csv("trace_data/trace.csv")


whenClosed = {'EBFSummer20.iSnap.ExampleDialog.Close.By.gifClick', 'EBFSummer20.iSnap.Gallery.Overlay.close',
              'EBFSummer20.iSnap.ExampleDialog.Close.By.okClick'}

trivialExp = {"\n", "Explanation...\n\n\n", "Explanation...\n\n\nT"}


# Functions and other data declared/defined here
def check_feedback_conditions(result, i):
    if result.at[i, "message"] == 'ExampleFeedback.logFeedback' and eval(result.at[i, "data"])[
        'explanation'] not in trivialExp:
        return True
    return False


def getTimeDifference(start, end):
    # This is a dummy date since we do not need the date stamp as all of our results are for day2 currently.
    # But it should be changed if need be
    _date = date(2005, 7, 14)
    datetime1 = datetime.combine(_date, pd.to_datetime(trace.loc[[start]].time).dt.time[start])
    datetime2 = datetime.combine(_date, pd.to_datetime(trace.loc[[end]].time).dt.time[end])
    return (datetime2 - datetime1).total_seconds()


def processData(start, end, explanation):
    new_row = {}
    new_row['userID'] = trace.at[start, "userID"]
    new_row['projectID'] = trace.at[start, "projectID"]
    new_row['gifName'] = trace.at[start, "data"]
    new_row['timeOnExample'] = getTimeDifference(start, end)
    new_row['selfExplanation'] = explanation.replace('Explanation...\n\n\n', '')
    new_row['openingTime'] = trace.at[start, "time"]
    new_row['closingTime'] = trace.at[end, "time"]
    return new_row


result_data = pd.DataFrame(
    columns=['projectID', 'gifName', 'timeOnExample', 'selfExplanation', 'openingTime', 'closingTime'])


trace.userID = trace.userID.apply(lambda x: x.split("[DELIM]")[0])
trace = trace.sort_values(by=['userID', 'time'])

user_ids = trace.userID.unique()


# for i in trace.index:
#     if trace.at[i, 'message'] == 'EBFSummer20.iSnap.ExampleDialog.popUp':
#         start = i
#         end = 0
#         exp = ""
#         if i < trace.shape[0] and trace.at[i, "message"] != 'EBFSummer20.iSnap.ExampleDialog.popUp':
#             if check_feedback_conditions(trace, i):
#                 exp = eval(trace.at[i, "data"])['explanation']
#             if trace.at[i, "message"] in whenClosed and not end:
#                 end = i
#         new_row = processData(start, end, exp)
#         result_data = result_data.append(new_row, ignore_index=True)

# print(result_data)


for user_id in user_ids:
    sub_trace = trace[trace.userID.eq(user_id)]
    gif_name_set = {}
    for i in sub_trace.index:
        if sub_trace.at[i, 'message'] == 'EBFSummer20.iSnap.ExampleDialog.popUp':
            gif_name_set.add()
    print(sub_trace)





save_obj(result_data, 'result_data', 'results')