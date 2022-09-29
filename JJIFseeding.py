'''

Reads in the registrations and makes a seeding based on the ranking list

'''
from datetime import datetime
import os
from pandas import json_normalize
from fpdf import FPDF

import streamlit as st
import plotly.graph_objects as go

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


class PDF(FPDF):
    '''
    overwrites the pdf settings
    '''
    def header(self):
        # Logo
        self.image('Logo_real.png', 8, 8, 30)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(70)
        # Title
        # self.cell(30, 10, 'Seeding' ,'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cell(0, 10, 'Printed ' + str(now) + ' Page ' +
                  str(self.page_no()) + '/{nb}', 0, 0, 'C')


def read_in_cat_rankid():
    ''' Read in file
     - HELPER FUNCTION
     Reads in a csv  and convert catergory ids to catergoy names

    '''
    inp_file = pd.read_csv("catID_rankID.csv", sep=';')
    rank_cat_id_out = inp_file[
        ['cat_id', 'ranking_id']].set_index('cat_id').to_dict()['ranking_id']

    return rank_cat_id_out


# uri of sportdataAPI
BASEURI = "https://www.sportdata.org/ju-jitsu/rest/"


key_map = {
    "1466": "U21 Jiu-Jitsu Women -45 kg",
    "1467": "U21 Jiu-Jitsu Women -48 kg",
    "1468": "U21 Jiu-Jitsu Women -52 kg",
    "1469": "U21 Jiu-Jitsu Women -57 kg",
    "1470": "U21 Jiu-Jitsu Women -63 kg",
    "1471": "U21 Jiu-Jitsu Women -70 kg",
    "1472": "U21 Jiu-Jitsu Women +70 kg",
    "1459": "U21 Jiu-Jitsu Men -56 kg",
    "1460": "U21 Jiu-Jitsu Men -62 kg",
    "1461": "U21 Jiu-Jitsu Men -69 kg",
    "1462": "U21 Jiu-Jitsu Men -77 kg",
    "1463": "U21 Jiu-Jitsu Men -85 kg",
    "1464": "U21 Jiu-Jitsu Men -94 kg",
    "1465": "U21 Jiu-Jitsu Men +94 kg",
    "1488": "U21 Duo Men",
    "1487": "U21 Duo Mixed",
    "1489": "U21 Duo Women",
    "1436": "U21 Fighting Women -45 kg",
    "1437": "U21 Fighting Women -48 kg",
    "1438": "U21 Fighting Women -52 kg",
    "1439": "U21 Fighting Women -57 kg",
    "1441": "U21 Fighting Women -63 kg",
    "1442": "U21 Fighting Women -70 kg",
    "1443": "U21 Fighting Women +70 kg",
    "1429": "U21 Fighting Men -56 kg",
    "1430": "U21 Fighting Men -62 kg",
    "1431": "U21 Fighting Men -69 kg",
    "1432": "U21 Fighting Men -77 kg",
    "1433": "U21 Fighting Men -85 kg",
    "1434": "U21 Fighting Men -94 kg",
    "1435": "U21 Fighting Men +94 kg",
    "1497": "U21 Show Men",
    "1498": "U21 Show Mixed",
    "1496": "U21 Show Women",
    "1491": "Adults Duo Men",
    "1492": "Adults Duo Mixed",
    "1490": "Adults Duo Women",
    "1444": "Adults Fighting Men -56 kg",
    "1451": "Adults Fighting Men -62 kg",
    "1446": "Adults Fighting Men -69 kg",
    "1447": "Adults Fighting Men -77 kg",
    "1448": "Adults Fighting Men -85 kg",
    "1449": "Adults Fighting Men -94 kg",
    "1450": "Adults Fighting Men +94 kg",
    "1452": "Adults Fighting Women -45 kg",
    "1453": "Adults Fighting Women -48 kg",
    "1454": "Adults Fighting Women -52 kg",
    "1455": "Adults Fighting Women -57 kg",
    "1456": "Adults Fighting Women -63 kg",
    "1457": "Adults Fighting Women -70 kg",
    "1458": "Adults Fighting Women +70 kg",
    "1473": "Adults Jiu-Jitsu Men -56 kg",
    "1474": "Adults Jiu-Jitsu Men -62 kg",
    "1475": "Adults Jiu-Jitsu Men -69 kg",
    "1476": "Adults Jiu-Jitsu Men -77 kg",
    "1477": "Adults Jiu-Jitsu Men -85 kg",
    "1478": "Adults Jiu-Jitsu Men -94 kg",
    "1479": "Adults Jiu-Jitsu Men +94 kg",
    "1480": "Adults Jiu-Jitsu Women -45 kg",
    "1481": "Adults Jiu-Jitsu Women -48 kg",
    "1482": "Adults Jiu-Jitsu Women -52 kg",
    "1483": "Adults Jiu-Jitsu Women -57 kg",
    "1484": "Adults Jiu-Jitsu Women -63 kg",
    "1485": "Adults Jiu-Jitsu Women -70 kg",
    "1486": "Adults Jiu-Jitsu Women +70 kg",
    "1494": "Adults Show Men",
    "1495": "Adults Show Mixed",
    "1493": "Adults Show Women"
    }


# since teams categories have no country I use this quick and dirty workaround
# to map clubnames in sportdata api to country codes...
CLUBNAME_COUNTRY_MAP = {"Belgian Ju-Jitsu Federation": 'BEL',
                        "Deutscher Ju-Jitsu Verband e.V.": 'GER',
                        "Federazione Ju Jitsu Italia": 'ITA',
                        "Romanian Martial Arts Federation": 'ROU',
                        "Österreichischer Jiu Jitsu Verband": 'AUT',
                        "Taiwan Ju Jitsu Federation": 'TPE'
                        }


def get_athletes_cat(eventid, cat_id, user, password):
    """
    get the athletes form sportdata per category & export to a nice data frame

    Parameters
    ----------
    eventid
        sportdata event_id (from database) [int]
    cat_id
        sportdata category_id (from database) [int]
     user
        api user name
    password
        api user password
    """

    # URI of the rest API
    uri = str(BASEURI)+'event/categories/'+str(eventid)+'/'+str(cat_id)+'/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)
    d_in = response.json()
    df_out = json_normalize(d_in["members"])

    if not df_out.empty:
        # first idivdual categories
        if df_out['type'].str.contains('athlete').any():
            #  match to name format of Duo categories
            df_out['last'] = df_out['last'].str.rstrip()
            df_out['first'] = df_out['first'].str.rstrip()
            df_out['name'] = df_out['first'] + " " + df_out['last']
            df_ath = df_out[['name', 'country_code']]
            # add the origial category id
            df_ath['cat_id'] = cat_id
            df_ath['cat_name'] = df_ath['cat_id'].replace(key_map)
            df_ath = df_ath.astype(str)
        else:
            # for an unclear reason teams to no have a country code...
            # convert club name to country using dict...
            df_out['country_code'] = df_out['club_name'].replace(CLUBNAME_COUNTRY_MAP)
            df_out['name'].replace(",", "/", regex=True, inplace=True)
            df_out['name'] = df_out['name'].str.rstrip()
            df_ath = df_out[['name', 'country_code']]
            df_ath['cat_id'] = cat_id
            df_ath['cat_name'] = df_ath['cat_id'].replace(key_map)
            df_ath = df_ath.astype(str)
    else:
        # just return empty datafram
        df_ath = pd.DataFrame()
    return df_ath


def get_event_name(eventid, user, password):
    """
    get the event name from sportdarta as string

    Parameters
    ----------
    eventid
        sportdata event_id (from database) [int]
     user
        api user name
    password
        api user password
    """

    # URI of the rest API
    uri = str(BASEURI)+'/event/'+str(eventid)+'/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)
    d_in = response.json()
    df_out = json_normalize(d_in)
    event_name = df_out['name'].astype(str)
    return event_name


def get_ranking_cat(user, password):
    """
    ranking has differnet category ids...
    so get a dict with them
    """

    # URI of the rest API
    uri = str(BASEURI)+'/ranking/categories/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)
    d_in = response.json()
    df_rankcats = json_normalize(d_in)
    df_rankcats = df_rankcats.drop(['cat_sex', 'cat_isteam'], axis=1)
    df_rankcats = df_rankcats.set_index('cat_id')
    my_series = df_rankcats['cat_title'].squeeze()
    dict_ranking = my_series.to_dict()
    return dict_ranking


def get_ranking(rank_cat_id, max_rank_pos, user, password):
    """
    get the athletes form sportdata per category & export to a nice data frame

    Parameters
    ----------
    rank_cat_id
        sportdata category_id (from ranking) [int]
    MAX_RANK_pos
        seeding will stop at this number [int]
    user
        api user name
    password
        api user password
    """

    # URI of the rest API
    uri = str(BASEURI)+'/ranking/category/'+str(rank_cat_id)+'/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)
    d_in = response.json()
    df_out = json_normalize(d_in)

    if not df_out.empty:
        df_out['name'] = df_out['name'].str.split('(').str[0]
        df_out['name'] = df_out['name'].str.rstrip()
        df_rank = df_out[['name', 'countrycode', 'rank', 'cat_id', 'totalpoints', 'cat_title']]
        # rename rank to ranking since df.rank is a function name
        df_rank['ranking'] = df_rank['rank'].astype(int)
        df_rank = df_rank[df_rank['ranking'] < int(max_rank_pos)]
        df_rank['ranking'] = df_rank['ranking'].astype(str)
    else:
        # just return empty datafram
        df_rank = pd.DataFrame()
    return df_rank


def rev_look(val_in, dict_in):
    ''' revese lookup of key.
    Returns first matching key
    Parameters
    ----------
    val
        value to be looked up
    dict
        dict that contains the keys and value

    '''
    key_out = next(key for key, value in dict_in.items() if value == val_in)

    return key_out


def draw_as_table(df_in):
    ''' draws a dataframe as a table and then as a fig.
    Parameters
    ----------
    val
        value to be looked up
    dict
        dict that contains the keys and value

    '''

    header_color = 'grey'
    row_even_color = 'lightgrey'
    row_odd_color = 'white'

    fig_out = go.Figure(data=[go.Table(
                        columnwidth=[15, 50, 24, 25, 25],
                        header=dict(values=["Position", "Name", "Country", "Ranking Postion", "Ranking Points"],
                                    fill_color=header_color,
                                    font=dict(family="Arial", color='white', size=12),
                                    align='left'),
                        cells=dict(values=[df_in.position, df_in.name, df_in.country_code, df_in.ranking, df_in.totalpoints],
                                   line_color='darkslategray',
                                   # 2-D list of colors for alternating rows
                                   fill_color=[[row_odd_color, row_even_color, row_odd_color, row_even_color, row_odd_color]*5],
                                   align=['left', 'left', 'left', 'left', 'left'],
                                   font=dict(family="Arial", color='black', size=10)
                                   ))
                        ])

    numb_row = len(df_in.index)

    fig_out.update_layout(
        autosize=False,
        width=750,
        height=(numb_row+1) * 30,
        margin=dict(
            l=20,
            r=50,
            b=0,
            t=0,
            pad=4
            ),
        )

    return fig_out


# main progreamm starts here
sd_key = st.number_input("Enter the Sportdata event number",
                         help='the number behind vernr= in the URL', value=325)

tourname = get_event_name(str(sd_key), st.secrets['user'], st.secrets['password'])

st.title('Seeding for ' + str(tourname))

st.sidebar.image("https://i0.wp.com/jjeu.eu/wp-content/uploads/2018/08/jjif-logo-170.png?fit=222%2C160&ssl=1",
                 use_column_width='always')

mode = st.sidebar.selectbox('Select mode', ['Top10', 'Top20'])

if mode == 'Top10':
    MAX_RANK = 10
else:
    MAX_RANK = 20

# ID_TO_NAME = read_in_catkey()
catID_to_rankID = read_in_cat_rankid()

# create empty temporary list for catgories to merge into team categories
list_df_athletes = []
list_df_ranking = []

dict_ranking_ids = get_ranking_cat(st.secrets['user'], st.secrets['password'])

my_bar = st.progress(0)
with st.spinner('Read in data'):
    list_df_ath = []
    for i, key_ath in enumerate(key_map):
        athletes_cat = get_athletes_cat(str(sd_key),
                                        str(key_ath),
                                        st.secrets['user'],
                                        st.secrets['password'])
        list_df_athletes.append(athletes_cat)

        my_bar.progress(((i+1)/len(key_map))/2)
    df_athletes = pd.concat(list_df_athletes)

    df_athletes['cat_id'] = df_athletes['cat_id'].astype(int)
    df_athletes['rank_id'] = df_athletes['cat_id'].replace(catID_to_rankID)
    df_athletes = df_athletes.astype(str)

    for j, key in enumerate(dict_ranking_ids):
        ranking_cat = get_ranking(str(key),
                                  MAX_RANK,
                                  st.secrets['user'],
                                  st.secrets['password'])
        list_df_ranking.append(ranking_cat)
        my_bar.progress(0.5+((j+1)/len(dict_ranking_ids))/2)
    df_ranking = pd.concat(list_df_ranking)
    df_ranking['rank_id'] = df_ranking['cat_id']


# duo cat have no matching on name...need to fix this
# st.write(df_ranking[df_ranking['cat_title'].str.contains("DUO")])
# st.write(df_athletes[df_athletes['cat_name'].str.contains("Duo")])

# get all the categories that are registered
cat_list = df_athletes['cat_name'].unique()

df_all = pd.merge(df_athletes, df_ranking, on=['rank_id', 'name'])

# new pdf in landscape
pdf = PDF('L')

for k in cat_list:
    pdf.add_page()
    pdf.alias_nb_pages()
    pdf.set_font("Arial", size=20)
    pdf.cell(200, 20, txt="Seeding for Category " + k, ln=1, align='C')

    names_seeding = df_all[['name', 'country_code', 'ranking', 'totalpoints']][(df_all['cat_name'] == str(k))]
    names_seeding['ranking'] = names_seeding['ranking'].astype(int)
    names_seeding = names_seeding.sort_values(by=['ranking'], ascending=True)
    names_seeding['position'] = list(range(1, len(names_seeding.index)+1))
    names_seeding = names_seeding.astype(str)

    st.header(k)
    if len(names_seeding) > 0:
        st.write(names_seeding)
        fig = draw_as_table(names_seeding)
        PNG_NAME = str(k) + ".png"
        fig.write_image(PNG_NAME)
        pdf.image(PNG_NAME)
        os.remove(PNG_NAME)
    else:
        st.write("No one in Seeding")
        pdf.set_font("Arial", size=15)
        pdf.cell(200, 20, txt="No one in Seeding", ln=1, align='L')


pdf.output("dummy2.pdf")
with open("dummy2.pdf", "rb") as pdf_file:
    PDFbyte2 = pdf_file.read()

st.download_button(label="Download Seeding",
                   data=PDFbyte2,
                   file_name='Download Seeding.pdf')
os.remove("dummy2.pdf")


st.sidebar.markdown('<a href="mailto:sportdirector@jjif.org">Contact for problems</a>', unsafe_allow_html=True)

LINK = '[Click here for the source code](https://github.com/ClaudiaBehnke86/JJIFseeding)'
st.markdown(LINK, unsafe_allow_html=True)
