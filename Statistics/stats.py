import pandas as pd
import numpy as np

from sql import *



def split_data(fragdata):

    dic = {}

    dic['all'] = fragdata

    poc = fragdata[fragdata['Type'] == 'Pocillopora']
    acro = fragdata[fragdata['Type'] == 'Acropora']

    dic['smallpoc'] = poc[poc['Size'] <= poc['Size'].median()]
    dic['largepoc'] = poc[poc['Size'] > poc['Size'].median()]
    dic['smallacro'] = acro[acro['Size'] <= acro['Size'].median()]
    dic['largeacro'] = acro[acro['Size'] > acro['Size'].median()]

    lg = fragdata[fragdata['Resort'] == 'LG']
    dic['lgpoc'] = lg[lg['Type'] == 'Pocillopora']
    dic['lgacro'] = lg[lg['Type'] == 'Acropora']
    kh = fragdata[fragdata['Resort'] == 'KH']
    dic['khpoc'] = kh[kh['Type'] == 'Pocillopora']
    dic['khacro'] = kh[kh['Type'] == 'Acropora']

    zones = ['KH Water Villas', 'LG Water Villas', 'Channel', 'Parrot Reef', 'Blu', 'Blu Deep', 'House Reef', 'Blue Holes', 'Dive Site', 'Coral Trail', 'Anchor Point', 'Al Barakat']
    for z in zones:
        dic[z] = fragdata[fragdata['Zone'] == z]

    shallow = fragdata[fragdata['Depth'] <= 3]
    dic['shallow'] = shallow
    dic['medium'] = fragdata[(fragdata['Depth'] > 3) & (fragdata['Depth'] <= 7)]
    dic['deep'] = fragdata[fragdata['Depth'] > 7]

    dic['shaded'] = shallow[shallow['Zone'] == 'KH Water Villas']
    dic['non-shaded'] = shallow[~shallow['Zone'].isin(['KH Water Villas', 'LG Water Villas'])]

    dic['a'] = fragdata[fragdata['Elev'] == 'A']
    dic['b'] = fragdata[fragdata['Elev'] == 'B']
    dic['c'] = fragdata[fragdata['Elev'] == 'C']

    dic['lg'], dic['kh'], dic['poc'], dic['acro'] = lg, kh, poc, acro

    return(dic)


def get_volume_series():

    df, frags = get_volume_observations()
    dates = list(pd.date_range(date(2016, 1, 1), date(2020, 11, 25), freq='d'))
    fallen = pd.DataFrame(0, index=dates, columns=['Acropora', 'Pocillopora'])
    dead = pd.DataFrame(0, index=dates, columns=['Acropora', 'Pocillopora'])
    live = pd.DataFrame(0, index=dates, columns=['Acropora', 'Pocillopora'])
    initial = pd.DataFrame(0, index=dates, columns=['Acropora', 'Pocillopora'])

    for frag_id in pd.unique(df['FragmentId']):
        t = frags[frag_id]['Type']
        obs = df[df['FragmentId'] == frag_id].reset_index().to_dict('index')
        obs_count = len(obs.keys())
        date0 = frags[frag_id]['Transplanted']
        init_volume = obs[0]['vol']
        firstdead = obs_count

        if frags[frag_id]['Dead'] is None:

            for i in range(1, obs_count):
                date1 = obs[i]['Date']
                added = pd.Series(obs[i - 1]['vol'], index=list(pd.date_range(date0, date1, freq='d', closed='left')))
                live[t] = live[t].add(added, fill_value = 0)
                date0 = date1

            init_added = pd.Series(init_volume, index=list(pd.date_range(frags[frag_id]['Transplanted'], date.today(), freq='d')))
            initial[t] = initial[t].add(init_added, fill_value = 0)
            added = pd.Series(obs[obs_count - 1]['vol'], index=list(pd.date_range(obs[obs_count - 1]['Date'], date.today(), freq='d')))
            live[t] = live[t].add(added, fill_value = 0)


        else:
            i = 1
            while i < obs_count and obs[i]['Date'] < frags[frag_id]['Dead']:
                date1 = obs[i]['Date']
                added = pd.Series(obs[i - 1]['vol'], index=list(pd.date_range(date0, date1, freq='d', closed='left')))
                live[t] = live[t].add(added, fill_value = 0)
                date0 = date1
                i += 1
            added = pd.Series(obs[i - 1]['vol'], index=list(pd.date_range(date0, frags[frag_id]['Dead'], freq='d', closed='left')))
            live[t] = live[t].add(added, fill_value = 0)
            init_added = pd.Series(init_volume, index=list(pd.date_range(frags[frag_id]['Transplanted'], frags[frag_id]['Dead'], freq='d', closed='left')))
            initial[t] = initial[t].add(init_added, fill_value = 0)

            if i < obs_count:
                added = pd.Series(obs[i - 1]['vol'], index=list(pd.date_range(frags[frag_id]['Dead'], obs[i]['Date'], freq='d', closed='left')))
                dead[t] = dead[t].add(added, fill_value = 0)
                date0 = obs[i]['Date']
                for j in range(i, obs_count - 2):
                    date1 = obs[j + 1]['Date']
                    added = pd.Series(obs[j]['vol'], index=list(pd.date_range(date0, date1, freq='d', closed='left')))
                    dead[t] = dead[t].add(added, fill_value = 0)
                    date0 = date1

            added = pd.Series(obs[obs_count - 1]['vol'], index=list(pd.date_range(obs[obs_count - 1]['Date'], date.today(), freq='d')))
            dead[t] = dead[t].add(added, fill_value = 0)

            if frags[frag_id]['FallDate'] is not None:
                act_date = obs[obs_count - 1]['Date'] + (datetime.strptime('20' + str(frags[frag_id]['FallDate']).zfill(6), '%Y%m%d').date() - obs[obs_count - 1]['Date'])/ 2
                added = pd.Series(obs[obs_count - 1]['vol'], index=list(pd.date_range(act_date, date.today(), freq='d')))
                fallen[t] = fallen[t].add(added, fill_value = 0)


    return(0.1568*0.001*live, 0.1568*0.001*dead, 0.1568*0.001*initial, 0.1568*0.001*fallen)


def get_disrupted_frag_data(disruption = ''):

    fragdata = split_data(download_frag_data())
    disruptedzones = ['Water Villas (LG)', 'Channel (KH)', 'Blue Holes (LG)', 'Parrot Reef (LG)', 'Water Villas (KH)', 'House Reef (KH)']
    df = fragdata['all'][fragdata['all']['Zone'].isin(disruptedzones)]

    disruptions = {'Water Villas (LG)': ['2019-06-01', '2020-03-01'], 'Channel (KH)': ['2018-05-01', '2020-02-01'],
                 'Parrot Reef (LG)': ['2017-02-15', '2018-08-01'], 'Blue Holes (LG)': ['2019-06-01', '2020-03-01'],
                 'Water Villas (KH)': ['2019-03-30', '2019-09-30'], 'House Reef (KH)': ['2019-03-30', '2019-09-30']}

    df['Size'] = 0.3296*df['Size']
    df['tdelta'] = (df['Transplanted'] - df['StartDate']).dt.days <= 90
    df = df[df['tdelta']]
    df['Start'], df['Stop'], df['Disrupted'] = 0, 0, 0
    df['Transplanted'] = df['Transplanted'].dt.date
    df['Deep'] = 1*(df['Depth'] > 3)

    long_df = pd.DataFrame(columns = list(df.columns))

    n0 = 4 if disruption == 'heatwave' else 0

    for z in disruptedzones[n0:]:
        zdf = df.loc[df['Zone'] == z].copy()
        s, e = datetime.strptime(disruptions[z][0], '%Y-%m-%d').date(), datetime.strptime(disruptions[z][1], '%Y-%m-%d').date()
        
        beginning = zdf.loc[zdf['Transplanted'] < s].copy()
        beginning['Stop'] = pd.concat([beginning['T'], (s-beginning['Transplanted']).dt.days], axis=1).min(axis=1)
        beginning['E'] = zdf['E']*(beginning['Stop'] == beginning['T'])
        long_df = long_df.append(beginning, ignore_index = True)
        
        middle = zdf.loc[(zdf['Transplanted'] < e) & (zdf['Transplanted'] + pd.to_timedelta(zdf['T'], 'D') > s)].copy()
        middle['Start'] = (s - middle['Transplanted']).dt.days.clip(lower = 0)
        middle['Disrupted'] = 1
        middle['Stop'] = pd.concat([(e - middle['Transplanted']).dt.days, zdf['T']], axis=1).min(axis=1)
        middle['E'] = zdf['E']*(middle['Stop'] == middle['T'])
        long_df = long_df.append(middle, ignore_index = True)
        
        ending = zdf.loc[zdf['Transplanted'] + pd.to_timedelta(zdf['T'], 'D') > e].copy()
        ending['Start'] = (e - ending['Transplanted']).dt.days.clip(lower = 0)
        ending['Stop'] = ending['T']
        ending['E'] = zdf['E']*(ending['Stop'] == ending['T'])
        long_df = long_df.append(ending, ignore_index = True)

    long_df = long_df.loc[~((long_df['Start'] == long_df['Stop']) & (long_df['Start'] == 0))]
    long_df = long_df.sort_values(['FragmentId', 'Start'])

    return(long_df)