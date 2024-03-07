'''

Reads in the registrations and makes a seeding based on the ranking list

'''
import numpy as np
from datetime import datetime
import os
import re
from pandas import json_normalize
from fpdf import FPDF

import streamlit as st
import plotly.graph_objects as go

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# for the name matching
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct  # Leading Juice for us

class PDF(FPDF):
    '''
    overwrites the pdf settings
    '''

    def __init__(self, orientation, tourname):
        # initialize attributes of parent class
        super().__init__(orientation)
        # initialize class attributes
        self.tourname = tourname

    def header(self):
        # Logo
        self.image('Logo_real.png', 8, 8, 30)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(70)
        # Title
        self.cell(30, 10, 'Seeding ' + self.tourname, 'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number & printing date
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cell(0, 10, 'Printed ' + str(now) + ' Page ' +
                  str(self.page_no()) + '/{nb}', 0, 0, 'C')


def read_in_cat_rankid():
    ''' Read in file
     - HELPER FUNCTION
     Reads in a csv  and convert category ids to category names

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
    "1495": "Adults Show Women",
    "1493": "Adults Show Mixed"
    }


# since teams categories have no country I use this quick and dirty workaround
# to map clubnames in sportdata api to country codes...
CLUBNAME_COUNTRY_MAP = {"Belgian Ju-Jitsu Federation": 'BEL',
                        "Deutscher Ju-Jitsu Verband e.V.": 'GER',
                        "Federazione Ju Jitsu Italia": 'ITA',
                        "Romanian Martial Arts Federation": 'ROU',
                        "Österreichischer Jiu Jitsu Verband": "AUT",
                        "Taiwan Ju Jitsu Federation": "TPE",
                        "Royal Spain Ju Jutsi Federation": 'ESP',
                        "Federation Française de Judo, Jujitsu, Kendo et DA": 'FRA',
                        "Pakistan Ju-Jitsu Federation": 'PAK',
                        "Vietnam Jujitsu Federation": 'VIE',
                        "Ju-Jitsu Federation of Slovenia": 'SLO',
                        "Hellenic Ju-Jitsu Federation": 'GRE',
                        "Ju Jitsu Association of Thailand": 'THA',
                        "FÉDÉRATION ROYALE MAROCAINE DE JU-JITSU ": 'MAR',
                        "Ju-Jitsu Federation of Montenegro": 'MNE',
                        "Swiss Judo & Ju-Jitsu Federation": 'SUI',
                        "Kazakhstan Jiu Jitsu Association": 'KAZ',
                        "Mongolian Jiu-Jitsu Association": 'MGL',
                        "Algerian Federation of Ju Jitsu": 'ALG',
                        "Federacion Colombiana de Jiu-Jitsu": 'COL',
                        "Federaciòn Costarricense de Jiu Jitsu": 'CRC',
                        "Federaciòn Uruguaya de Jiu Jitsu": 'URU',
                        "Asociatiòn Argentina de Jiu Jitsu": 'ARG',
                        }


def conv_to_type(df_in, type_name, type_list):
    '''
    checks strings in data frames and
    replaces them with types based on the _INP lists (see line 28 - 49)

    Parameters
    ----------
    df_in
        data frame to check [str]
    type_name
        the name of the _INP list to check [str]
    type_list
        of the _INP list to check [list]
    '''
    for inp_str in type_list:
        df_in[type_name].where(~(df_in[type_name].str.contains(inp_str)),
                               other=inp_str, inplace=True)

    return df_in[type_name]


def ngrams(string, n_gr=3):
    '''
    Function from name comparison
    'https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e'

    used to check for similar names
    Parameters
    ----------
    string
        input string
    n_gr
        ?

    '''
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams_in = zip(*[string[i:] for i in range(n_gr)])
    return [''.join(ngram_in) for ngram_in in ngrams_in]


def awesome_cossim_top(A, B, ntop, lower_bound=0):
    '''
    Function from name comparison
    'https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e'

    force A and B as a CSR matrix.
    If they have already been CSR, there is no overhead'''
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M*ntop

    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)
    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)
    return csr_matrix((data, indices, indptr), shape=(M, N))


def get_matches_df(sparse_matrix, name_vector, top=100):
    '''
    Function from name comparison
    'https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e'

    unpacks the resulting sparse matrix
    '''
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if sparsecols.size > top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similarity_in = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similarity_in[index] = sparse_matrix.data[index]

    return pd.DataFrame({'left_side': left_side,
                         'right_side': right_side,
                         'similarity': similarity_in})


@st.cache_data
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
    if len(d_in) > 0:
        df_out = json_normalize(d_in["members"])
    else:
        df_out = pd.DataFrame()

    if not df_out.empty:
        # first individual categories
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
            df_out['name'] = df_out['name'].str.split('(').str[1]
            df_out['name'] = df_out['name'].str.split(')').str[0]
            df_out['name'].replace(",", " ", regex=True, inplace=True)
            df_out['name'].replace("_", " ", regex=True, inplace=True)
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
    get the event name from sportdata as string

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
    event_name = df_out['name'].astype(str)[0]
    return event_name


def get_ranking_cat(user, password):
    """
    ranking has different category ids...
    so get a dict with them
    """

    # URI of the rest API
    uri = str(BASEURI)+'/ranking/categories/'

    response = requests.get(uri, auth=HTTPBasicAuth(user, password), timeout=5)
    d_in = response.json()
    df_rankcats = json_normalize(d_in)
    df_rankcats = df_rankcats.drop(['cat_sex', 'cat_isteam'], axis=1)
    st.write(df_rankcats)
    df_rankcats = df_rankcats.set_index('cat_id')
    my_series = df_rankcats['cat_title'].squeeze()
    dict_ranking = my_series.to_dict()
    return dict_ranking


@st.cache_data
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
        df_rank = df_rank.rename(columns={"countrycode": 'country_code'})
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
                        columnwidth=[15, 40, 20, 25, 25, 20],
                        header=dict(values=["Position", "Name", "Country", "Ranking Position", "Ranking Points", "Similarity"],
                                    fill_color=header_color,
                                    font=dict(family="Arial", color='white', size=12),
                                    align='left'),
                        cells=dict(values=[df_in.position, df_in.name, df_in.country_code, df_in.ranking, df_in.totalpoints, df_in.similarity],
                                   line_color='darkslategray',
                                   # 2-D list of colors for alternating rows
                                   fill_color=[[row_odd_color, row_even_color]*2],
                                   align=['left', 'left', 'left', 'left', 'left'],
                                   font=dict(family="Arial", color='black', size=10)
                                   ))
                        ])

    numb_row = len(df_in.index)

    fig_out.update_layout(
        autosize=False,
        width=750,
        height=(numb_row+1) * 35,
        margin=dict(
            l=20,
            r=50,
            b=0,
            t=0,
            pad=4
            ),
        )

    return fig_out


# main program starts here
st.sidebar.image("https://i0.wp.com/jjeu.eu/wp-content/uploads/2018/08/jjif-logo-170.png?fit=222%2C160&ssl=1",
                 use_column_width='always')

# get upcoming events
uri_upc = "https://www.sportdata.org/ju-jitsu/rest/events/upcoming/"

response = requests.get(uri_upc, auth=HTTPBasicAuth(st.secrets['user'], st.secrets['password']), timeout=5)
d_upc = response.json()
df_upc = json_normalize(d_upc)

offmail = ["sportdata@jjif.org", "rick.frowyn@jjeu.eu", "office@jjau.org", "mail@jjif.org", "jjif@sportdata.org", "fjjitalia@gmail.com"]
evts = df_upc['name'][df_upc['contactemail'].isin(offmail)].tolist()
evts.append('Other')
option = st.sidebar.selectbox("Choose your event", evts,
                              help='if the event is not listed choose Other')

if option == 'Other':
    sd_key = st.sidebar.number_input("Enter the Sportdata event number",
                                     help='the number behind vernr= in the URL', value=325)
else:
    sd_key = int(df_upc['id'][df_upc['name'] == option])

tourname = get_event_name(str(sd_key), st.secrets['user'], st.secrets['password'])

st.title('Seeding for ' + tourname)

mode = st.sidebar.selectbox('Select mode', ['Top10', 'Top20'])

if mode == 'Top10':
    MAX_RANK = 10
else:
    MAX_RANK = 20

# preselected age_divisions
AGE_SEL = ["U16", "U18", "U21", "Adults"]

age_select = st.sidebar.multiselect('Select the age divisions for seeding',
                                    AGE_SEL,
                                    ["U21", "Adults"])

# ID_TO_NAME = read_in_catkey()
catID_to_rankID = read_in_cat_rankid()

# create empty temporary list for categories to merge into team categories
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

    if len(df_athletes) > 0:
        df_athletes['cat_id'] = df_athletes['cat_id'].astype(int)
        df_athletes['rank_id'] = df_athletes['cat_id'].replace(catID_to_rankID)
        df_athletes = df_athletes.astype(str)

        # select the age divisions you want to seed
        df_athletes['age_division'] = df_athletes['cat_name']
        df_athletes['age_division'] = conv_to_type(df_athletes, 'age_division', AGE_SEL)

        # remove what is not selected
        df_athletes = df_athletes[df_athletes['age_division'].isin(age_select)]

        # only read in rankings associated to the df_athletes
        cat_list = df_athletes['rank_id'].unique()

        for j, key in enumerate(cat_list):
            ranking_cat = get_ranking(str(key),
                                      MAX_RANK,
                                      st.secrets['user'],
                                      st.secrets['password'])
            list_df_ranking.append(ranking_cat)
            my_bar.progress(0.5+((j+1)/len(cat_list))/2)
        df_ranking = pd.concat(list_df_ranking)
        df_ranking['rank_id'] = df_ranking['cat_id']

        vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)

        with st.expander('Details on name matching', expanded=False):
            st.write('Similar names were matched to avoid missing mapping. This is based on:')
            st.write('https://towardsdatascience.com/surprisingly-effective-way-to-name-matching-in-python-1a67328e670e')

        # create dummy column for similarity score
        df_ranking['similarity'] = df_ranking['name']
        df_ranking['original_name'] = ''

        # loop over all ranks to match
        for cat in cat_list:

            # get the names of from leading dataframe (athletes) and ranking frame
            names_athletes = df_athletes[df_athletes['rank_id'] == cat]['name']
            names_ranking = df_ranking[df_ranking['rank_id'] == cat]['name']

            # skip category if empty
            if len(names_athletes) < 1:
                df_ranking.loc[
                    df_ranking['rank_id'] == cat,
                    'similarity'
                ] = None
                continue

            # combine the two lists of names into one list
            all_names = pd.concat([names_athletes, names_ranking]).values
            # perform matching over combined name list
            matrix = vectorizer.fit_transform(all_names)
            if len(all_names) > 4:
                matrix = awesome_cossim_top(matrix, matrix.transpose(), 10, .4)
            else:
                matrix = awesome_cossim_top(matrix, matrix.transpose(), 4, .4)
            # create a dataframe with the matches
            df_matches = get_matches_df(matrix, all_names)

            # Duo names have much lower similarity
            name_cat = df_athletes[df_athletes['rank_id'] == cat]['cat_name'].astype(str)[0]

            if "Duo" in name_cat:
                min_value = 0.35
            else:
                min_value = 0.55

            # remove self-mapping of names (exact matches)
            df_matches = df_matches[
                (df_matches["similarity"] < .99999999) & (
                    df_matches["similarity"] > min_value)
            ]
            # create mapping dictionary for names
            dict_map = dict(zip(df_matches.left_side, df_matches.right_side))

            # if approximate matches are found, replace the names
            if len(dict_map) > 0:
                # drop all keys in the mapping dict that do not have a value
                # that appears in the athletes names. by doing this, you
                # avoid replacing names in the ranking list with other names
                # in the ranking list, which would potentially lead to mis-
                # matches with athlete names
                slim_dict = {key: value for key,
                    value in dict_map.items() if value in df_athletes[
                        df_athletes['rank_id'] == cat
                    ]['name'].values
                }
                similarity_dict = {
                    key: df_matches[
                        df_matches["right_side"] == key
                    ]['similarity'].values[0] for key, value in slim_dict.items()
                }
                # replace names in the ranking data
                df_ranking.loc[
                    df_ranking['rank_id'] == cat,
                    'similarity'
                ] = df_ranking[df_ranking['rank_id'] == cat]['similarity'].apply(lambda x:  similarity_dict.get(x))
                df_ranking.loc[
                    df_ranking['rank_id'] == cat,
                    'original_name'
                ] = df_ranking[df_ranking['rank_id'] == cat]['name'].apply(lambda x:  x if slim_dict.get(x) else None)
                df_ranking.loc[
                    df_ranking['rank_id'] == cat,
                    'name'
                ] = df_ranking[df_ranking['rank_id'] == cat]['name'].replace(slim_dict)
            else:
                df_ranking.loc[
                    df_ranking['rank_id'] == cat,
                    'similarity'
                ] = None

        df_ranking['similarity'] = df_ranking['similarity'].astype(float).fillna(1)
        df_all = pd.merge(df_athletes, df_ranking, on=['rank_id', 'name', 'country_code'])
        # new pdf in landscape

        pdf = PDF('L', tourname)

        cat_list_str = df_athletes['cat_name'].unique()

        for k in cat_list_str:
            pdf.add_page()
            pdf.alias_nb_pages()
            pdf.set_font("Arial", size=20)
            pdf.cell(200, 20, txt="Seeding for Category " + k, ln=1, align='C')

            st.header(k)
            if len(df_all[(df_all['cat_name'] == str(k))]) <= 0:
                st.write("Category is empty")
            else:
                names_seeding = df_all[['name', 'country_code', 'ranking', 'totalpoints', 'similarity', 'original_name']][(df_all['cat_name'] == str(k))]
                names_seeding['ranking'] = names_seeding['ranking'].astype(int)
                names_seeding = names_seeding.sort_values(by=['ranking'], ascending=True)
                names_seeding['position'] = list(range(1, len(names_seeding.index)+1))

                # move positions to first column
                cols = names_seeding.columns.tolist()
                cols = cols[-1:] + cols[:-1]
                names_seeding = names_seeding[cols]

                # remove more than 4 seeded people
                names_seeding = names_seeding[names_seeding['position'] < 5]
                names_seeding = names_seeding.astype(str)

                if len(names_seeding) > 0:
                    if names_seeding[names_seeding["similarity"].astype(float) < 1.0].empty:
                        st.write(names_seeding[['name', 'country_code', 'ranking', 'totalpoints']])
                    else:
                        st.warning('There are non exact matches, check names and original_name', icon="⚠️")
                        names_seeding["similarity"] = names_seeding["similarity"].astype(float).round(2)
                        st.dataframe(names_seeding.style.highlight_between(subset=['similarity'], left=0.1, right=0.99, color="#F31C2B"))
                        names_seeding = names_seeding.astype(str)
                        pdf.cell(200, 20, txt='!!! There are non exact matches, check names in event and ranking', ln=1, align='C')
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

    else:
        st.error("The event has no categories, use a different event")

st.sidebar.markdown('<a href="mailto:sportdirector@jjif.org">Contact for problems</a>', unsafe_allow_html=True)

LINK = '[Click here for the source code](https://github.com/ClaudiaBehnke86/JJIFseeding)'
st.markdown(LINK, unsafe_allow_html=True)
