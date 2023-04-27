import MySQLdb

####################################################################################
#                                                                                  #
# This file contains all the needed functions to interact with the AI SQL database #
#              A description is included in each function definition               #
#                                                                                  #
####################################################################################

def OpenMydb(database="Reefscapers2020"):

    # Opens the specified database

    db = MySQLdb.connect(host="", user="", passwd="", db=database)
    db.autocommit(True)
    return db


def get_monitoring_sets(tag, table = 'FSMonitoring'):

    # Gets all monitoring sets of the specified frame

    db = OpenMydb(database = "")
    cur = db.cursor()

    request = "SELECT * FROM {} WHERE Tag='{}'".format(table, tag)
    cur.execute(request)

    desc = cur.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in cur.fetchall()]
    
    db.close()
    return(data)


def get_monitoring_dates(tag):

    # Gets all monitoring dates for a specific tag, IN ORDER.

    request = "SELECT DISTINCT `Date Code` FROM FSMonitoring WHERE Tag = '{}' ORDER BY `Date Code`".format(tag)
    db = OpenMydb()
    cur = db.cursor()
    cur.execute(request)

    dates = [int(row[0]) for row in cur.fetchall()]

    db.close()
    return(dates)


def get_unanalyzed_sets(tag):

    # Gets monitoring dates that have not been analyzed yet

    request = "SELECT DISTINCT `Date Code` FROM FSMonitoring LEFT JOIN FrameParams ON `View 1` = MonitoringPicture WHERE Tag = '" + tag + "' AND MonitoringPicture IS NULL"
    db = OpenMydb()
    cur = db.cursor()
    cur.execute(request)

    dates = [int(row[0]) for row in cur.fetchall()]

    db.close()
    return(dates)


def need_to_update():
    request = """SELECT DISTINCT A.Tag FROM (
                SELECT DISTINCT Tag FROM FSMonitoring LEFT JOIN FrameParams ON FSMonitoring.`View 1` = FrameParams.MonitoringPicture
                WHERE Score IS NULL ) A
                INNER JOIN Observations ON A.Tag = Observations.Tag"""
    db = OpenMydb()
    cur = db.cursor()
    cur.execute(request)

    tags = [row[0] for row in cur.fetchall()]

    db.close()
    return(tags)


def insert_annotations(filename, boxes, classes, scores, category_index, threshold = 0.8):

    # Insert annotations for selected monitoring picture in Annotations table
    # Overwrites existing annotations
    # Meant to represent a specific fragment at a specific time on a specific monitoring picture
    
    db = OpenMydb()
    cur = db.cursor()
    
    request = "DELETE FROM Annotations WHERE MonitoringPicture='" + filename + "'"
    cur.execute(request)
    
    for i in range(boxes.shape[0]):
        if(scores[i] >= threshold):
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            
            Type = str(category_index[classes[i]]['name'])
            Score = str(scores[i])
            TopLeftX = str(xmin)
            TopLeftY = str(ymin)
            Width = str(xmax - xmin)
            Height = str(ymax - ymin)
            MonitoringPicture = filename

            request = "INSERT INTO Annotations (MonitoringPicture, Type, Score, TopLeftX, TopLeftY, Width, Height) VALUES ('" + \
                    filename + "','" + Type + "'," + Score + "," + TopLeftX + "," + TopLeftY + "," + Width + "," + Height + ")"
            
            cur.execute(request)
            # print(request)

    db.close()


def get_annotations(filename):

    # Returns all annotations for a specific monitoring picture

    request = "SELECT * FROM Annotations WHERE MonitoringPicture='{}'".format(filename)
    db = OpenMydb()
    cur = db.cursor()
    cur.execute(request)

    desc = cur.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in cur.fetchall()]
    
    db.close()
    return(data)


def update_annot(annot_id, obs_id):

    # Adds observation ID to the given annotation record

    db = OpenMydb()
    cur = db.cursor()

    request = "UPDATE Annotations SET ObservationId={} WHERE AnnotationId={}".format(obs_id, annot_id)
    cur.execute(request)
    
    db.close()


def insert_frameparams(imagename, score, params):

    # Insert frame parameters into FrameParams database

    db = OpenMydb()
    cur = db.cursor()

    focal, rx, ry, rz, scaling, tu, tv = params

    request = "DELETE FROM FrameParams WHERE MonitoringPicture='{}'".format(imagename)
    cur.execute(request)

    request = "INSERT INTO FrameParams (MonitoringPicture, Score, Focal, Rx, Ry, Rz, Scaling, Tu, Tv) VALUES ('" + \
            imagename + "'," + str(score) + "," + str(focal) + "," + str(rx) + "," + str(ry) + "," + str(rz) + "," + str(scaling) + "," + str(tu) + "," + str(tv) + ")"
    cur.execute(request)
    
    db.close()


def get_params(filename):

    # Returns frame params for specified monitoring picture

    request = "SELECT * FROM FrameParams WHERE MonitoringPicture='{}'".format(filename)
    db = OpenMydb()
    cur = db.cursor()
    cur.execute(request)

    rows = cur.fetchall()
    db.close()

    if len(rows) == 0:
        print("No frame data for " + filename)
    else:
        return(list(rows[0])[2:])


def save_observations(tag, date, observations):

    # Save observations for a specific tag and date
    # Overwrites existing observations

    clear_observations(tag, date)

    for bar in observations:

        for frag in observations[bar]:

            obs_id = insert_observation(tag, date, frag[3], frag[4], frag[5], bar, frag[0])

            for a in frag[-1]:
                update_annot(a["AnnotationId"], obs_id)


def insert_observation(tag, date, type, height, width, bar, position):
    
    # Insert an observation in the Observations table
    # Table meant to represent a specific fragment at a specific time, regardless of monitoring pictures
    # Returns observation ID

    db = OpenMydb()
    cur = db.cursor()

    request = "INSERT INTO Observations (Tag, Date, Type, AdjWidth, AdjHeight, Bar, Position) VALUES ('" + \
            tag + "'," + date + ",'" + type + "'," + str(width)+ "," + str(height)+ ",'" + bar + "'," + str(position)+ ")"
    
    cur.execute(request)

    request = "SELECT LAST_INSERT_ID()"
    
    cur.execute(request)
    obs_id = cur.fetchall()[0]

    db.close()
    
    return(obs_id[0])


def get_observations(tag):

    # Returns all observations for a specific tag

    request = "SELECT * FROM Observations WHERE Tag='{}'".format(tag)
    db = OpenMydb()
    cur = db.cursor()
    cur.execute(request)

    desc = cur.description
    column_names = [col[0] for col in desc]
    data = [dict(zip(column_names, row)) for row in cur.fetchall()]
    
    db.close()
    return(data)
    

def update_obs(obs_id, frag_id):

    # Adds fragment ID to the given observation record

    db = OpenMydb()
    cur = db.cursor()

    request = "UPDATE Observations SET FragmentId={} WHERE ObservationId={}".format(frag_id, obs_id)
    cur.execute(request)
    
    db.close()


def clear_observations(tag, date):

    # Deletes all observations corresponding to the given Tag and Date

    db = OpenMydb()
    cur = db.cursor()
    
    request = "DELETE FROM Observations WHERE Tag='{}' AND Date={}".format(tag, date)
    cur.execute(request)

    db.close()


def save_fragments(tag, fragments):

    # Save fragments for a specific tag
    # Overwrites existing fragments

    frag_ids = []

    for frag in fragments:

        frag_id = insert_fragment(frag[0], frag[1], frag[2], frag[3])
        frag_ids.append(frag_id)

        for a in frag[-1]:
            update_obs(a["ObservationId"], frag_id)

    return(frag_ids)


def insert_fragment(tag, bar, position, type):

    # Insert a fragment in the Fragments table
    # Table meant to represent a specific fragment, regardless of time
    # Returns fragment ID
    
    db = OpenMydb()
    cur = db.cursor()

    request = "INSERT INTO Fragments (Tag, Bar, Position, Type) VALUES ('" + \
            tag + "','" + bar + "'," + str(position) + ",'" + type + "')"
    
    cur.execute(request)

    request = "SELECT LAST_INSERT_ID()"
    
    cur.execute(request)
    frag_id = cur.fetchall()[0]

    db.close()
    
    return(frag_id[0])


def update_frags(frag_ids, frag_dates):

    # Adds transplantation date to fragment data, along with Death/Fall date if available.

    db = OpenMydb()
    cur = db.cursor()

    for i in range(0, len(frag_ids)):
        if 'Dead' in frag_dates[i]:
            request = "UPDATE Fragments SET Transplanted='{}', Dead = '{}' WHERE FragmentId={}".format(frag_dates[i]['Transplanted'], frag_dates[i]['Dead'], frag_ids[i])
        else:
            request = "UPDATE Fragments SET Transplanted='{}' WHERE FragmentId={}".format(frag_dates[i]['Transplanted'], frag_ids[i])
        cur.execute(request)
    
    db.close()


def clear_fragments(tag):

    # Deletes all fragments corresponding to the given Tag

    db = OpenMydb()
    cur = db.cursor()
    
    request = "DELETE FROM Fragments WHERE Tag='{}'".format(tag)
    cur.execute(request)

    db.close()


def insert_status(frag_id, obs_id, ttype, datecode, days):

    # Creates new status entry

    db = OpenMydb()
    cur = db.cursor()

    request = "INSERT INTO Status (FragmentId, ObservationId, Type, Date, Days) VALUES (" + \
            str(frag_id) + "," + str(obs_id) + ",'" + ttype + "'," + str(datecode) + "," + str(days) + ")"
    
    cur.execute(request)

    db.close()
