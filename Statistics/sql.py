import MySQLdb
from random import sample
from datetime import datetime, date, timedelta
import pandas as pd


def OpenMydb(database="Reefscapers2020_v2"):

    # Opens the specified database

    db = MySQLdb.connect(host="localhost", user="root", passwd="root", db=database)
    db.autocommit(True)
    return db


def download_frag_data():

    request = """SELECT Frags.FragmentId, FSFrames.Transplanted AS StartDate, Frags.Transplanted,
                    T, E, Frags.Type, Size, Zone, Depth, Elev, LEFT(Sizes.Tag, 2) as Resort FROM 
            (
                    SELECT Fragments.FragmentId, Transplanted, DATEDIFF(STR_TO_DATE(MAX(Date),'%y%m%d'), Transplanted) AS T, 0 AS E, Fragments.Type, RIGHT(Bar,1) AS Elev FROM `Fragments`
                    INNER JOIN Status ON Fragments.FragmentId = Status.FragmentId
                    WHERE Status.Type IN ('Live Coral', 'Bleached') AND Dead IS NULL AND Fragments.Type IN ('Pocillopora', 'Acropora') 
                    GROUP BY FragmentId
                UNION SELECT FragmentId, Transplanted, DATEDIFF(Dead, Transplanted) AS T, 1 AS E, Type, RIGHT(Bar,1) AS Elev FROM `Fragments` WHERE Type IN ('Pocillopora', 'Acropora') AND Dead IS NOT NULL AND Transplanted IS NOT NULL
            ) Frags INNER JOIN
            (
                
                SELECT B.FragmentId, ROUND(AdjWidth + AdjHeight, 2) AS Size, Tag
                FROM
                (
                    SELECT FragmentId, MIN(Date) AS min_date
                    FROM Observations
                    GROUP BY FragmentId
                ) B INNER JOIN Observations ON B.FragmentId = Observations.FragmentId AND B.min_date = Observations.Date

            ) Sizes ON Frags.FragmentId = Sizes.FragmentId
            LEFT JOIN FSFrames ON Sizes.Tag = FSFrames.Tag"""

    db = OpenMydb()
    cur = db.cursor()
    cur.execute(request)
    
    df = pd.DataFrame(cur.fetchall())
    desc = cur.description
    df.columns = [col[0] for col in desc]
    df['Depth'] = pd.to_numeric(df['Depth'], errors='coerce')
    df['StartDate'] = df['StartDate'].apply(lambda x:datetime.strptime('20' + str(x).zfill(6), '%Y%m%d'))
    df['Transplanted'] = pd.to_datetime(df['Transplanted'], format='%Y-%m-%d')
    df['Sandy'] = df['Zone'].isin(['Blu (LG)', 'Blu Deep (LG)', 'House Reef (KH)'])
    df.loc[df['Resort'] == 'RE', 'Resort'] = 'LG'
    
    db.close()
    
    return(df)


def get_volume_observations():

    db = OpenMydb()
    cur = db.cursor()

    request = """SELECT Observations.FragmentId, Date As MonDate,  POWER(AdjHeight * AdjWidth, 1.5) AS vol FROM Observations
                INNER JOIN Fragments ON Observations.FragmentId = Fragments.FragmentId
                WHERE Fragments.Type IN ('Pocillopora', 'Acropora') ORDER BY FragmentId, MonDate"""
    cur.execute(request)
    df = pd.DataFrame(cur.fetchall())
    desc = cur.description
    df.columns = [col[0] for col in desc]
    df['Date'] = df['MonDate'].apply(lambda x:datetime.strptime('20' + str(x).zfill(6), '%Y%m%d')).dt.date

    request = """SELECT Fragments.FragmentId, Type, Transplanted, Dead, FallDate FROM Fragments 
                LEFT JOIN (SELECT FragmentId, MIN(Date) AS FallDate FROM Status WHERE Type = 'Fallen' GROUP BY FragmentId ) A
                ON Fragments.FragmentId = A.FragmentId
                WHERE Type IN ('Acropora', 'Pocillopora')"""
    cur.execute(request)
    frags = {}
    for row in cur.fetchall():
        frags[row[0]] = {'Type' : row[1], 'Transplanted' : row[2], 'Dead' : row[3], 'FallDate' : row[4]}

    return(df, frags)


def get_mortality():

    db = OpenMydb()
    cur = db.cursor()

    request = """SELECT Fragments.FragmentId, Fragments.Type AS Specie, Fragments.Tag, Fragments.Transplanted, Zone,
                Date1, Status1.Type AS Stat1, Date2, Status2.Type AS Stat2
                FROM Fragments
                INNER JOIN
                (   SELECT Status1.FragmentId, Status1.Date As Date1, MIN(Status2.Date) AS Date2
                    FROM Status Status1
                    INNER JOIN Status Status2 ON Status1.FragmentId = Status2.FragmentId AND Status1.Date < Status2.Date
                    GROUP BY FragmentId, Date1) A ON Fragments.FragmentId = A.FragmentId
                INNER JOIN Status Status1 ON A.FragmentId = Status1.FragmentId AND A.Date1 = Status1.Date
                INNER JOIN Status Status2 ON A.FragmentId = Status2.FragmentId AND A.Date2 = Status2.Date
                INNER JOIN FSFrames ON Fragments.Tag = FSFrames.Tag
                WHERE Fragments.Type IN ('Pocillopora', 'Acropora')"""
    cur.execute(request)
    data = pd.DataFrame(cur.fetchall())
    desc = cur.description
    data.columns = [col[0] for col in desc]

    data['D1'] = data['Date1'].apply(lambda x:datetime.strptime('20' + str(x).zfill(6), '%Y%m%d'))
    data['D2'] = data['Date2'].apply(lambda x:datetime.strptime('20' + str(x).zfill(6), '%Y%m%d'))
    
    data['Median'] = data['D1'] + (data['D2'] - data['D1']) / 2

    data['tdelta'] = data['Median'].dt.date - data['Transplanted']
    data = data[[td.days > 180 for td in data['tdelta']]]

    survived = data[(data['Stat1'] == 'Live Coral') & (data['Stat2'] == 'Live Coral')]
    dead = data[(data['Stat1'] == 'Live Coral') & (data['Stat2'] != 'Live Coral')]

    return(survived, dead)


def get_growth_data():

    mainzones = ['Al Barakat (LG)', 'Anchor Point (LG)', 'Blu (LG)', 'Blu Deep (LG)', 'Coral Trail (LG)', 'Dive Site (LG)',
              'Water Villas (LG)', 'Blue Holes (LG)', 'Parrot Reef (LG)', 'Water Villas (KH)', 'House Reef (KH)', 'Channel (KH)']
    db = OpenMydb()
    cur = db.cursor()

    request = """SELECT Obs1.FragmentId, Fragments.Type, Fragments.Tag, Zone,
                Obs1.Date As Date1, Status1.Type AS Stat1, 0.5*(Obs1.AdjHeight + Obs1.AdjWidth) AS len1,
                Obs2.Date As Date2, Status2.Type AS Stat2, 0.5*(Obs2.AdjHeight + Obs2.AdjWidth) AS len2
                FROM Observations Obs1
                INNER JOIN Observations Obs2 ON Obs1.FragmentId = Obs2.FragmentId AND Obs1.Date < Obs2.Date
                INNER JOIN Fragments ON Obs1.FragmentId = Fragments.FragmentId
                INNER JOIN FSFrames ON Fragments.Tag = FSFrames.Tag
                INNER JOIN Status Status1 ON Obs1.FragmentId = Status1.FragmentId AND Obs1.Date = Status1.Date
                INNER JOIN Status Status2 ON Obs2.FragmentId = Status2.FragmentId AND Obs2.Date = Status2.Date
                WHERE Fragments.Type IN ('Pocillopora', 'Acropora')"""
    cur.execute(request)
    data = pd.DataFrame(cur.fetchall())
    desc = cur.description
    data.columns = [col[0] for col in desc]

    df = data.copy()
    df['len1'], df['len2'] = 0.3296 * df['len1'], 0.3296 * df['len2']
    df['NDate1'] = df['Date1'].apply(lambda x:datetime.strptime('20' + str(x).zfill(6), '%Y%m%d'))
    df['NDate2'] = df['Date2'].apply(lambda x:datetime.strptime('20' + str(x).zfill(6), '%Y%m%d'))
    df['Median'] = (df['NDate1'] + (df['NDate2'] - df['NDate1']) / 2).apply(lambda x: str(x.year) + 'M' + str(x.month).zfill(2))
    df['tdelta'] = (df['NDate2'] - df['NDate1']).dt.days / 365
    df['avgrowth'] = (df['len2'] - df['len1'])/df['tdelta']
    df = df[df['tdelta'] > 5/12]
    df = df[df['Zone'].isin(mainzones)]
    livedf = df[(df['Stat1'] == 'Live Coral') & (df['Stat2'] == 'Live Coral')]
    livedf = livedf.groupby(['Tag', 'FragmentId', 'Type', 'Zone']).mean()
    livedf.reset_index(level=[0,1,2,3], inplace=True)
    livedf.set_index('FragmentId', inplace = True)

    return(livedf)


def get_bleached_data():

    db = OpenMydb()
    cur = db.cursor()

    request = """SELECT Fragments.FragmentId, Fragments.Transplanted, Fragments.Type, bleached_date1 AS Date, Zone, Depth, Outcome
                FROM Fragments
                INNER JOIN FSFrames ON Fragments.Tag = FSFrames.Tag
                INNER JOIN (SELECT Status.FragmentId, bleached_date1, Status.Type AS Outcome FROM Status INNER JOIN
                                (SELECT Status.FragmentId, bleached_date1, MIN(Date) AS outcome_date FROM Status INNER JOIN
                                    (SELECT Status.FragmentId, MIN(Date) AS bleached_date1, MAX(Date) AS bleached_date2 FROM Status WHERE Type = 'Bleached Coral' GROUP BY FragmentId) A
                                ON Status.FragmentId = A.FragmentId
                                WHERE Date > bleached_date2
                                GROUP BY FragmentId
                                ) B ON Status.FragmentId = B.FragmentId AND Status.Date = B.outcome_date
                            ) C ON Fragments.FragmentId = C.FragmentId 
                WHERE Fragments.Type IN ('Pocillopora', 'Acropora')"""

    cur.execute(request)
    data = pd.DataFrame(cur.fetchall())
    desc = cur.description
    data.columns = [col[0] for col in desc]
    data['ObsDate'] = data['Date'].apply(lambda x:datetime.strptime('20' + str(x).zfill(6), '%Y%m%d')).dt.date
    data['ObsDate'] = pd.to_datetime(data['ObsDate'])
    data['Transplanted'] = pd.to_datetime(data['Transplanted'])
    print(data[['ObsDate', 'Transplanted']].dtypes)
    data['BleachDelay'] = (data['ObsDate'] - data['Transplanted']).dt.days

    return(data)