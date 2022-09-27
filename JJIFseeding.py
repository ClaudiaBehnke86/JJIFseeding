'''

Reads in the registrations and makes a seeding based on the ranking list

'''
import random
import streamlit as st
import plotly.graph_objects as go

import requests
import json
import plotly.express as px
from requests.auth import HTTPBasicAuth
import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'

from datetime import datetime
from datetime import timedelta

from pandas import json_normalize
from fpdf import FPDF
from fpdf import Template

import base64

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class PDF(FPDF):
    def header(self):
        # Logo
        self.image('Logo_real.png', 8, 8, 30)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(70)
        # Title
        self.cell(30, 10, 'Seeding','C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Printed ' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ' Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')


# def read_in_catkey():
#     ''' Read in file
#      - HELPER FUNCTION
#      Reads in a csv  and convert catergory ids to catergoy names

#     '''
#     inp_file = pd.read_csv("catID_name.csv", sep=';')
#     key_map = inp_file[
#         ['cat_id', 'name']
#     ].set_index('cat_id').to_dict()['name']

#     return key_map

def read_in_cat_rankID():
    ''' Read in file
     - HELPER FUNCTION
     Reads in a csv  and convert catergory ids to catergoy names

    '''
    inp_file = pd.read_csv("catID_rankID.csv", sep=',')
    rank_cat_id_out = inp_file[
        ['cat_id', 'ranking_id']].set_index('cat_id').to_dict()['ranking_id']

    return rank_cat_id_out

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


#  since teams categories have no country I use this quick and dirty workaround
#  to map clubnames in sportdata api to country codes... 
CLUBNAME_COUNTRY_MAP = {"Belgian Ju-Jitsu Federation": 'BEL',
                        "Deutscher Ju-Jitsu Verband e.V.": 'GER',
                        "Federazione Ju Jitsu Italia": 'ITA',
                        "Romanian Martial Arts Federation": 'ROU'}


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

    #URI of the rest API

    uri = 'https://www.sportdata.org/ju-jitsu/rest/event/categories/'+str(eventid)+'/'+str(cat_id)+'/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password))
    d = response.json()
    df_out = json_normalize(d["members"])

    if not df_out.empty:
        #first idivdual categories
        if df_out['type'].str.contains('athlete').any():
            #  match to name format of Duo categories
            df_out['name'] = df_out['first'] + " " + df_out['last']
            df = df_out[['name' , 'country_code']]
            # add the origial category id
            df['cat_id'] = cat_id
            df['cat_name'] = df['cat_id'].replace(key_map)
            df = df.astype(str)
        else:
            # for an unclear reason teams to no have a country code...
            # convert club name to country using dict...
            df_out['country_code'] = df_out['club_name'].replace(CLUBNAME_COUNTRY_MAP)
            df_out['name'].replace(",", "/", regex=True, inplace=True)
            df = df_out[['name', 'country_code']]
            df['cat_id'] = cat_id
            df['cat_name'] = df['cat_id'].replace(key_map)
            df = df.astype(str)
    else:
        # just return empty datafram
        df =pd.DataFrame()
    return df


def get_ranking_cat(user, password):
    """
    ranking has differnet category ids... 
   
    """

    #URI of the rest API
    uri = 'https://www.sportdata.org/ju-jitsu/rest/ranking/categories/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password))
    d = response.json()
    df = json_normalize(d)
    
    df = df.drop(['cat_sex', 'cat_isteam'], axis=1)
    df = df.set_index('cat_id')
    my_series = df['cat_title'].squeeze()
    dict_ranking = my_series.to_dict()
    return dict_ranking


def get_ranking(rank_cat_id, max_rank, user, password):
    """
    get the athletes form sportdata per category & export to a nice data frame

    Parameters
    ----------
    rank_cat_id
        sportdata category_id (from ranking) [int]
    max
        seeding will stop at this number [int]
    user
        api user name
    password
        api user password    
    """

    #URI of the rest API
    uri = 'https://www.sportdata.org/ju-jitsu/rest/ranking/category/'+ str(rank_cat_id)+'/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password))
    d = response.json()
    df_out = json_normalize(d)

    if not df_out.empty:
        df = df_out[['name', 'countrycode','rank', 'cat_id', 'id','totalpoints','cat_title']]
        df['rank'] = df['rank'].astype(int)
        df = df[df['rank'] < int(max_rank)]
        df['rank'] = df['rank'].astype(str)
    else:
        # just return empty datafram
        df =pd.DataFrame()
    return df

def rev_look(val, dict):
    ''' revese lookup of key.
    Returns first matching key
    Parameters
    ----------
    val
        value to be looked up
    dict
        dict that contains the keys and value

    '''
    key = next(key for key, value in dict.items() if value == val)

    return key

def draw_as_table(df):

    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'
    
    fig = go.Figure(data=[go.Table(
                    columnwidth = [50,25,25,25],
                    header=dict(values=["Name", "Country", "Ranking Postion", "Ranking_Points"],
                    fill_color=headerColor,
                    font = dict(family= "Arial", color = 'white', size = 12),
                    align='left'),
                    cells=dict(values=[df.name, df.country_code, df.rank, df.totalpoints],
                        line_color='darkslategray',
                        # 2-D list of colors for alternating rows
                        fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]*5],
                        align = ['left', 'left','left', 'left'],
                        font = dict(family= "Arial", color = 'black', size = 10)
                        ))
                    ])

    numb_row = len(df.index)

    fig.update_layout(
        autosize=False,
        width=600,
        height=(numb_row+1) *25,
        margin=dict(
            l=20,
            r=50,
            b=0,
            t=0,
            pad=4
            ),
        )
    
    
    return fig

st.title('Seeding')
st.sidebar.image("https://i0.wp.com/jjeu.eu/wp-content/uploads/2018/08/jjif-logo-170.png?fit=222%2C160&ssl=1",
                 use_column_width='always')

mode = st.sidebar.selectbox('Select mode',['Top10','Top20']) 

if mode == 'Top10':
    max_rank = 10
else:  
    max_rank = 20


#ID_TO_NAME = read_in_catkey()
catID_to_rankID = read_in_cat_rankID()

sd_key = 325     
sd_key = st.number_input("Enter the number of Sportdata event number",
                                 help='is the number behind vernr= in the URL', value=325)
# create empty temporary list for catgories to merge into team categories
list_df_athletes = []
list_df_ranking = []

dict_ranking_ids = get_ranking_cat(st.secrets['user'],st.secrets['password'])

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
                                  max_rank,
                                  st.secrets['user'],
                                  st.secrets['password'])
        list_df_ranking.append(ranking_cat)
        my_bar.progress(0.5+((j+1)/len(dict_ranking_ids))/2)
    df_ranking = pd.concat(list_df_ranking)
    df_ranking['rank_id'] = df_ranking['cat_id']
   
df_all = pd.merge(df_athletes, df_ranking, on=['rank_id','name'])

cat_list = df_all['cat_name'].to_list()
pdf = PDF()

for k in cat_list:

    pdf.add_page()
    pdf.set_font("Arial", size = 25)
    pdf.cell(200, 20, txt = "Seeding for Category " + k,
          ln = 1, align = 'C')
    pdf.alias_nb_pages()

    names_seeding = df_all[['name','country_code','rank', 'totalpoints']][(df_all['cat_name'] == str(k))]
    names_seeding.sort_values(by=['rank'])
    st.write(k)
    st.write(names_seeding)

    pdf.cell(200, 10, txt = k,
          ln = 2, align = 'C')

    # if(len(names_seeding)>0):
    #     fig = draw_as_table(names_seeding)
    #     png_name = str(k) + ".png"
    #     st.write(png_name)
    #     fig.write_image(png_name)
    #     pdf.image(png_name) 

   
pdf.output("dummy2.pdf")  
with open("dummy2.pdf", "rb") as pdf_file:
    PDFbyte2 = pdf_file.read()

st.download_button(label="Download Seeding",
                   data=PDFbyte2,
                   file_name='Download Seeding.pdf')        


st.sidebar.markdown('<a href="mailto:sportdirector@jjif.org">Contact for problems</a>', unsafe_allow_html=True)

LINK = '[Click here for the source code](https://github.com/ClaudiaBehnke86/JJIFseeding)'
st.markdown(LINK, unsafe_allow_html=True)

