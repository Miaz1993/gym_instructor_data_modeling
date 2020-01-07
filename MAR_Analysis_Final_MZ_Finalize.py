#!/usr/bin/env python
# coding: utf-8

import sys
# No default value, all input arg needs to be specified
print 'Number of input args = ',len(sys.argv)

date       = sys.argv[1]
modelDur   = int(sys.argv[2])
dbAddress  = sys.argv[3]
port       = int(sys.argv[4])
dbName     = sys.argv[5]
dbId       = sys.argv[6]
dbPasswd   = sys.argv[7]
outCurrWeek = sys.argv[8]

# 20190818 14 ds111514-a0.mlab.com 11514 inShape dbuser X8NBV4S2P1S False

import pymongo
import pandas as pd
import numpy as np
import pymongo
import re
import patsy
import collections
from   datetime import datetime
from   datetime import timedelta
from   holidays import UnitedStates
from   bdateutil import isbday
from statsmodels.api import add_constant
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
import itertools
import scipy.stats as ss
from scipy import stats

#Collection names
COLLNAME_CLASS_CLASSATTENDANCE         = 'groupEx_schedule'
COLLNAME_CLASS_REPORTOUTPUT            = 'ReportClassTest'
# COLLNAME_CLASS_OUTPUT                  = 'ReportClass'
COLLNAME_CLASS_NAME                    = 'groupEx_class'
COLLNAME_CLASS_CLASSMEMBER             = 'groupEx_groupExMember'
COLLNAME_CLEANING                      = 'extraCleaning'


CONSTANT_CLASS_HOURWEEKGROUP = {'Weekday Early Morning':{
                            'StartHour':[5,6,7],
                            'weekday':['Monday','Tuesday','Wednesday','Thursday','Friday']},
               'Weekday Morning':{
                            'StartHour':[8,9,10,11],
                            'weekday':['Monday','Tuesday','Wednesday','Thursday','Friday']},
                'Weekday Noon':{
                            'StartHour':[12],
                            'weekday':['Monday','Tuesday','Wednesday','Thursday','Friday']},  
                'Weekday Afternoon':{
                            'StartHour':[13,14,15,16],
                             'weekday':['Monday','Tuesday','Wednesday','Thursday','Friday']},
                'Weekday Evening':{
                            'StartHour':[17,18],
                            'weekday':['Monday','Tuesday','Wednesday','Thursday','Friday']},               
                'Weekday Night':{
                             'StartHour':[0,1,2,3,4,19,20,21,22,23],
                             'weekday':['Monday','Tuesday','Wednesday','Thursday','Friday']}, 
                'Saturday':{
                            'StartHour':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
                            'weekday':['Saturday']},
                'Sunday':{
                            'StartHour':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
                            'weekday':['Sunday']}             
              }


CONSTANT_CLASS_WEEKDAYGROUP  = {'M-W':['Monday','Tuesday','Wednesday'],
          'Thursday':['Thursday'],
          'Friday':['Friday'],
          'Saturday':['Saturday'],
          'Sunday':['Sunday']}


CONSTANT_CLASS_HOURGROUP = {'Early morning':[5,6,7],
          'Morning':[8,9,10,11],
          'Noon':[12],
          'Afternoon':[13,14,15,16],
          'Evening':[17,18],
          'Night':[0,1,2,3,4,19,20,21,22,23]}


CONSTANT_CLASS_WEEKDAYMAP = {'Monday':'0_Mon',
             'Tuesday':'1_Tue',
             'Wednesday':'2_Wed',
             'Thursday':'3_Thu',
             'Friday':'4_Fri',
             'Saturday':'5_Sat',
             'Sunday':'6_Sun'}   


LOCALHOLIDAYS = UnitedStates()


#Methods
def datestr2datetime(date):
    """
    #input:    date string (yyyymmdd). e.g., 20171130
    #output:   datetime
    """
    year        = int(date[0:4])
    month       = int(date[4:6])
    day         = int(date[6:8])
    y           = datetime(year,month,day,23,59,59) 
    return y


def weeknum2month(week):
    year    = week[0:4]
    weeknum = week[4:]
    d       = year+'-W'+weeknum + '-1' #Pick Monday, which is the first date   
    firstday= datetime.strptime(d, "%Y-W%W-%w")
    return firstday.strftime('%Y%m')


def getCONSTANT_CLASS_HOURWEEKGROUP(hr,weekday):
    """
    - function: output schedule group given hour and weekday
    - inputs: 
        - hr: hour (e.g., 23 for 11pm)
        - weekday: weekday (e.g., Monday)
    - output: schedule group (e.g., Weekday_Evening)
    - requires: global variable CONSTANT_CLASS_HOURWEEKGROUP
    """    
    for key,val in CONSTANT_CLASS_HOURWEEKGROUP.iteritems():
        if(hr in val['StartHour'] and weekday in val['weekday']):
            return key
    print 'no match'
    return 'noMatch'


def getHourGroup(x):
    """
    - function: output hour group given hour
    - inputs: 
        - hr: hour (e.g., 23 for 11pm)
    - output: hour group (e.g., Evening)
    - requires: global variable CONSTANT_CLASS_HOURGROUP
    """        
    for key,val in CONSTANT_CLASS_HOURGROUP.iteritems():
        if(x in val):
            return key
    return 'null'


def getWeekdayGroup(x):
    """
    - function: output Weekday group given Weeiday
    - inputs: 
        - hr: Weekday (e.g., Monday)
    - output: Weekday group (e.g., M-W)
    - requires: global variable CONSTANT_CLASS_WEEKDAYGROUP
    """ 
    for key,val in CONSTANT_CLASS_WEEKDAYGROUP.iteritems():
        if(x in val):
            return key
    return 'null'


def getDateTime(x):
    try:
        return datetime.strptime(x, "%m-%d-%Y")
    except:
        return pd.NaT



def extra_cleaning(classAttendance,keywords):
    """
        - Function: extra clean on classAttendance data
        - Inputs: 
            - keywords: exclusion cases
            - classAttendance: raw attendance information
        - Output: clean classAttendance
    """

    for item in keywords:
        if item['category'] == 'classname':
            classAttendance = classAttendance[~classAttendance.ClassName.str.lower().str.contains(item['key'])]
        elif item['category'] == 'instructor':
            classAttendance = classAttendance[~classAttendance.Instructor.str.lower().str.contains(item['key'])]
        elif item['category'] == 'classroom':
            classAttendance = classAttendance[~classAttendance.Classroom.str.lower().str.contains(item['key'])]
        elif item['category'] == 'classtype':
            classAttendance = classAttendance[~classAttendance.ClassType.str.lower().str.contains(item['key'])]
        elif item['category'] == 'description':
            classAttendance = classAttendance[~classAttendance.ClassDescription.str.lower().str.contains(item['key'])]
        else:
            print 'Unknown Category in extra_cleaning collection' + item['category'] + 'is ignored.'
    return classAttendance



def potential_removal_classes(var, cum_sum, lower, higher):    
    
    """
        - Function: Find out list of classes which is potentially removable
        - Inputs
            - var: a groupby variable which is used to find out the median under certain groupby conditions
            - cum_sum: cumulative frequency, e.g. 0.96 means 96% classes are included
            - lower: lower cutting bound to determine outliers
            - higher: higher cutting bound to determine outliers
        - Output: List of potentially removable classes under certain conditions
    """
    
    class_percentage = pd.DataFrame(classAttendance.ClassName.str.lower().value_counts()/len(classAttendance)).cumsum()
    low_freq_class = class_percentage[class_percentage.ClassName>cum_sum].index.tolist()
    low_freq_class_data = classAttendance[classAttendance.ClassName.str.lower().isin(low_freq_class)].groupby([var,'ClassName'])\
                            .median().reset_index()[['ClassName',var,'HeadCount']]

    standard = classAttendance[['ClassName',var,'HeadCount']]

    standard = standard.groupby([var]).quantile([lower,higher]).\
    unstack(level = -1).rename(columns = {lower:'HC_lower',higher:'HC_upper'}).reset_index()
    check_stat = pd.merge(low_freq_class_data, standard, how='left', \
                          on=[var]).reset_index(drop=True)
    check_stat.columns = ['ClassName',var,'HeadCount','HC_lower','HC_upper']
    potential_remove_list = list(set(check_stat[(check_stat.HeadCount>=check_stat.HC_upper)|(check_stat.HeadCount<=check_stat.HC_lower)].ClassName.tolist()))

    return potential_remove_list



def final_removal_list(list_1, list_2, list_3, list_4):
    
    """
        - Funtion: 
            - Voting for potential remove classes. 
              If the classname has appeared more than three of the lists, 
              it would be attached on final remove list.
        - Inputs:
            - Lists of potential remove classes
        - Output:
            - Remove list after voting
    """
    
    final_list = list_1 + list_2 + list_3 + list_4
    final_dict = collections.Counter(final_list)
    return [key for (key,value) in final_dict.items() if value >=3] 



def aggrWeekly(mar,groupList,numIter):
    results = mar.groupby(groupList)                  ['HeadCount','MAR']                 .median()
    results = results.rename(columns={'HeadCount':'Avg Attendance'})
    #Confidence Interval of average MAR
    tempInt = mar.groupby(groupList+['iteration'])['MAR']                .mean()                .reset_index()                .groupby(groupList)['MAR']                .quantile([0.05,0.95])                .unstack(level=-1)                .rename(columns={0.05:'MAR_5Percent',0.95:'MAR_95Percent'})
    results = pd.merge(results,tempInt,left_index = True,right_index=True)
    #Range of Raw MAR values
    tempInt = mar.groupby(groupList+['iteration'])['MAR']                .agg(np.ptp)                .reset_index()                .groupby(groupList)['MAR']                .mean()
    results['MAR_range'] = tempInt    
    #Count the number of classes
    tempCnt = mar.groupby(groupList)                  ['MAR']                  .count()
    tempCnt = tempCnt/numIter
    tempCnt = pd.DataFrame(tempCnt)
    tempCnt = tempCnt.rename(columns={'MAR':'Num Class'})
    results = pd.merge(results,tempCnt,left_index = True,right_index=True)
    #Total number of attendance
    tempCnt = mar.groupby(groupList)                 ['HeadCount']                 .sum()
    tempCnt = tempCnt/numIter
    tempCnt = pd.DataFrame(tempCnt)
    tempCnt = tempCnt.rename(columns={'HeadCount':'Total Attendance'})
    results = pd.merge(results,tempCnt,left_index = True,right_index=True)    
    return results


#compute MAR
def compute_MAR(classAttendance, regression_str):
    classAttendance['BranchId_Classroom'] = classAttendance['BelongToBranchId'].                                    astype(str) +'_'+ classAttendance['Classroom']
    yall_1, Xall_1= patsy.dmatrices(regression_str,classAttendance, return_type='dataframe')

    #Compute MAR
    resultDf                       = pd.DataFrame()
    mar                            = pd.DataFrame()
    sampPct                        = 1.0
    numIter                        = 100
    minNumClasses                  = 0 #More than a class per week
    classAttendance.dropna(subset=['HeadCount'],inplace=True)

    #The following implementation is inefficient Especially for Age/Gender
    Instructor                     = classAttendance['Instructor']
    ClassName                      = classAttendance['ClassName']
    ClassType                      = classAttendance['ClassType']
    HourWeekGrp                    = classAttendance['HourWeekGrp']
    WeekdayGrp                     = classAttendance['WeekdayGrp']
    week                           = classAttendance['week']
    month                          = classAttendance['month']
    BelongToBranchId               = classAttendance['BelongToBranchId']
    Classroom                      = classAttendance['Classroom']

    #Bootstraping
    n=0
    while(n<numIter):
    #for n in range(0,numIter):
        try:
            numSamp = int(len(classAttendance)*sampPct)
            tempDf1 = classAttendance.sample(numSamp,replace=True)
            tempDf1.reset_index(inplace=True)
            y, X = patsy.dmatrices(regression_str, tempDf1, return_type="dataframe")
            X.fillna(X.median(),inplace=True)
            if(n == 0): 
                print "Use Poisson regression,prediction start..."
            predModel                  = sm.GLM(y, X, family=sm.families.Poisson())
            result                     = predModel.fit()
            ypred                      = result.predict(Xall_1)
            error                      = yall_1['HeadCount']-ypred
            marTemp                    = pd.DataFrame({
                                            'Month':month,
                                            'Week':week,
                                            'ClassType':ClassType,            
                                            'ClassName':ClassName,
                                            'Classroom':Classroom,
                                            'Instructor':Instructor,
                                            'HourWeekGrp':HourWeekGrp,
                                            'WeekdayGrp':WeekdayGrp,
                                            'BelongToBranchId':BelongToBranchId,
                                            'HeadCount':yall_1['HeadCount'],
                                            'MAR':error,
            })
            marTemp['iteration']       = n
            mar                        = pd.concat([mar,marTemp],axis=0)
            resultDf1 = pd.DataFrame({'param':result.params,
                                      'pvalues':result.pvalues})
            resultDf1['iteration']     = n
            resultDf                   = pd.concat([resultDf,resultDf1],axis=0)
            n = n + 1
        except:
            print 'Error because some group/classroom is missing during sampling'

    predictor = resultDf.groupby(resultDf.index)['param','pvalues'].mean()
    print 'MAR analysis completed'

    mar['Month'] = mar['Week'].apply(lambda x:weeknum2month(x))
    marDemoGraph = mar.loc[mar['iteration']==0]

    individ_group                 = ['BelongToBranchId',
                                     'Month',
                                     'Week',
                                     'HourWeekGrp',
                                     'ClassName',
                                     'ClassType',
                                     'Instructor']
    instructor_group              = ['BelongToBranchId',
                                     'Month',
                                     'Week',
                                     'Instructor']
    class_group                   = ['BelongToBranchId',
                                     'Month',
                                     'Week',
                                     'ClassName']
    individ_class_weekly          = aggrWeekly(mar,
                                     individ_group,
                                     numIter)
    instructor_weekly             = aggrWeekly(mar,
                                               instructor_group,
                                               numIter)
    class_weekly                  = aggrWeekly(mar,
                                               class_group,
                                               numIter)
    all_weekly                    = aggrWeekly(mar,
                                               ['Week'],
                                               numIter)

    individ_class_weekly          = individ_class_weekly.reset_index() 
    instructor_weekly             = instructor_weekly.reset_index()
    class_weekly                  = class_weekly.reset_index()
    
    return individ_class_weekly, instructor_weekly, class_weekly

#Get MAR from all classes
def compute_adjustment(individ_class_weekly):
    """
    Function: Check if instructors are underestimate or overestimate
    Input: individ_class_weekly
    Output: 
    
    underestimate : underestimate ratio
    overestimate  : overestimate ratio, 
    adjustment_low: underestimated classname
    adjustment_high:overestimate classname
    
    """
    uniq_class = individ_class_weekly.ClassName.unique()
    Whole_list = []
    for uniq in uniq_class:
        a = individ_class_weekly[individ_class_weekly.ClassName == uniq]['MAR'].tolist()
        Whole_list.append(a)

    #Check class mar positive ratio
    ratio_all = []
    for i in range(len(Whole_list)):
        a = [item>0 for item in Whole_list[i]]
        ratio = float(sum(a))/len(a)
        ratio_all.append(ratio)

    class_check = pd.DataFrame()
    class_check['ClassName'] = uniq_class
    class_check['Positive_Ratio'] = ratio_all

    #Check Instructor number per class
    class_instructor = individ_class_weekly.groupby(['ClassName','Instructor']).mean().groupby('ClassName').size().to_frame('Num of Instructor').reset_index()

    # Adding Instructor caused bias
    adjustment = class_check.join(class_instructor.set_index('ClassName'), on='ClassName')
    adjustment_low = adjustment[(adjustment['Num of Instructor']>1)&(adjustment['Positive_Ratio']<0.1)]
    adjustment_high = adjustment[(adjustment['Num of Instructor']>1)&(adjustment['Positive_Ratio']>0.9)]
    underestimate = float(len(adjustment_low))/(len(adjustment))
    overestimate = float(len(adjustment_high))/(len(adjustment))

    return underestimate, overestimate, adjustment_low, adjustment_high


# add ajustment term based on conditions
def adjustment_check(underestimate, overestimate):
    """
    Function: Check adjustment ratio if re-fit model needed
    input: underestimate, overestimate 
    output: True or False
    
    """
    sum_abnormal_class = underestimate + overestimate
    print(sum_abnormal_class)
    if sum_abnormal_class< 0.1:
        print 'MAR Analysis Complete Final'
        return False

    else:
        print 'Adjustment needed'
        return True


def re_fit_model(adjustment_low,adjustment_high,classAttendance):
    """
    Function: re-fit model by adding adjustment term
    Input: adjustment_low,adjustment_high,classAttendance
    Output: individ_class_weekly, instructor_weekly, class_weekly
    
    """
    adjustment_low_classlist = adjustment_low['ClassName'].tolist()
    adjustment_high_classlist = adjustment_high['ClassName'].tolist()
    index_low = classAttendance[classAttendance.ClassName.isin(adjustment_low_classlist)].index
    index_high = classAttendance[classAttendance.ClassName.isin(adjustment_high_classlist)].index

    classAttendance['Ajustment'] = 'Group0'
    classAttendance['Ajustment'].loc[index_low] = 'Group1'
    classAttendance['Ajustment'].loc[index_high] = 'Group2'

    individ_class_weekly, instructor_weekly, class_weekly = compute_MAR(classAttendance,'HeadCount ~ C(month) + C(BranchId_Classroom) + C(HourGrp):C(WeekdayGrp) + C(HourGrp):C(WeekdayGrp):C(BranchId_Classroom) + C(Ajustment)')
    print 'MAR Analysis Complete with Adjustment!'

    return individ_class_weekly, instructor_weekly, class_weekly



#Initialization 
endDate         = datestr2datetime(date)
curWeek         = endDate.strftime('%Y%V') 
startDate       = endDate - timedelta(days = 7*modelDur) #Weekly aggregation
print 'start',startDate,', end',endDate

# Data Retrieval
client          = pymongo.MongoClient(dbAddress, port)
client[dbName].authenticate(dbId,dbPasswd, mechanism='SCRAM-SHA-1')
db              = client[dbName]

#Get Exclusion
COLLNAME_CLASS_NAME                    = 'groupEx_class'
cursor_1                               = db[COLLNAME_CLASS_NAME].find({
        'CalPerformance':False
    })
exclusionList                          = pd.DataFrame(list(cursor_1))

cursor_2          = db[COLLNAME_CLASS_CLASSATTENDANCE].find({
        'StartTimeLocal':{'$gt':startDate},
        'EndTimeLocal':{'$lt':endDate},
        'Canceled':{'$ne':True},
        'Deleted':{'$ne':True},
        'SubFor':None
    })

classAttendance   = pd.DataFrame(list(cursor_2))

#Get Keywords
cursor_3          = db[COLLNAME_CLEANING].find({})
keywords          = list(cursor_3)

try:
    exclusionClassNames = exclusionList['ClassName'].unique()
except:
    exclusionClassNames = []

def excludeCompute(x,y):
    if(x['ClassName'] in y):
        print 'exclude ',x['ClassName'],' by setting nan'
        return np.nan
    else:
        return x['HeadCount']
    
#Data processing

classAttendance['month']       = classAttendance['StartTimeLocal']                                    .apply(lambda x:x.strftime('%Y%m'))
                                #year-month
classAttendance['week']        = classAttendance['StartTimeLocal']                                    .apply(lambda x:x.strftime('%Y%V'))
                                #GroupMon-Sun                                
classAttendance['bday']        = classAttendance['StartTimeLocal']                                    .apply(lambda x:isbday(
                                                    x,
                                                    holidays=LOCALHOLIDAYS))  
classAttendance['date']        = classAttendance['StartTimeLocal']                                    .apply(lambda x:x.strftime('%Y%m%d'))

classAttendance['HourWeekGrp'] = classAttendance                                    .apply(lambda x:getCONSTANT_CLASS_HOURWEEKGROUP(
                                                    x['StartHour'],
                                                    x['Weekday']),
                                                    axis=1)                             
classAttendance['WeekdayGrp']     = classAttendance['Weekday']                                    .apply(lambda x:getWeekdayGroup(x))
classAttendance['HourGrp']     = classAttendance['StartHour']                                    .apply(lambda x:getHourGroup(x)) 

#Remove invalid data
classAttendance                = classAttendance                                    .loc[np.logical_not(
                                        (classAttendance['bday']==False) 
                                        & (classAttendance['HeadCount']==0)
                                        )]


# No member data
classAttendance['HeadCount'] = classAttendance.apply(lambda x:excludeCompute(x,exclusionClassNames),axis=1)
classAttendance.dropna(subset=['HeadCount'],axis=0,inplace=True)

classAttendance = extra_cleaning(classAttendance,keywords)


# prepare potential remove tail class list
potential_remove_list_1 = potential_removal_classes(var = 'Classroom', cum_sum = 0.95, lower= 0.1, higher = 0.9 )
potential_remove_list_2 = potential_removal_classes(var = 'ClassType', cum_sum = 0.95, lower= 0.1, higher = 0.9 )
potential_remove_list_3 = potential_removal_classes(var = 'Instructor', cum_sum = 0.95, lower= 0.1, higher = 0.9)
potential_remove_list_4 = potential_removal_classes(var = 'HourWeekGrp', cum_sum = 0.95, lower= 0.1, higher = 0.9 )


#output final remove class list
remove_class_list = final_removal_list(potential_remove_list_1,potential_remove_list_2,potential_remove_list_3,potential_remove_list_4)


#exclude removal class list
classAttendance = classAttendance[~classAttendance.ClassName.isin(remove_class_list)]


print 'Potential removable tail classes are:', remove_class_list
print len(classAttendance)


# compute MAR and adjustment
individ_class_weekly, instructor_weekly, class_weekly = compute_MAR(classAttendance, 'HeadCount ~ C(month) + C(BranchId_Classroom) + C(HourGrp):C(WeekdayGrp) + C(HourGrp):C(WeekdayGrp):C(BranchId_Classroom)')

underestimate, overestimate, adjustment_low, adjustment_high = compute_adjustment(individ_class_weekly)

if adjustment_check(underestimate, overestimate) == True :
    individ_class_weekly, instructor_weekly, class_weekly = re_fit_model(adjustment_low,adjustment_high,classAttendance)
    

# Write to database
# Can choose to output result from current week or all results from model

if outCurrWeek == 'False':
    print 'remove all results'
    db[COLLNAME_CLASS_REPORTOUTPUT].delete_many({})
else:
    print 'only output/save current week result'
    db[COLLNAME_CLASS_REPORTOUTPUT].delete_many({'Week':curWeek})
    individ_class_weekly    = individ_class_weekly.loc[individ_class_weekly['Week']==curWeek]
    instructor_weekly       = instructor_weekly.loc[instructor_weekly['Week']==curWeek]
    class_weekly            = class_weekly.loc[class_weekly['Week']==curWeek] 

individ_class_weekly['BranchName']  = individ_class_weekly['BelongToBranchId'].apply(lambda x:str(x))
individ_class_weekly['type']        = 'classDetail'
instructor_weekly['BranchName']     = instructor_weekly['BelongToBranchId'].apply(lambda x:str(x))
instructor_weekly['type']           = 'instructor'
class_weekly['BranchName']          = class_weekly['BelongToBranchId'].apply(lambda x:str(x))
class_weekly['type']                = 'class'

print 'insert new results'
db[COLLNAME_CLASS_REPORTOUTPUT].insert_many(individ_class_weekly.to_dict('records'))
db[COLLNAME_CLASS_REPORTOUTPUT].insert_many(instructor_weekly.to_dict('records'))
db[COLLNAME_CLASS_REPORTOUTPUT].insert_many(class_weekly.to_dict('records'))
print 'done'

client.close()




