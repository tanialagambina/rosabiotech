"""
********************************************************************************
                            Data Processing Package
********************************************************************************
This package contains all of the functions that deal with parsing and processing
the data that comes from the database. The module also deals with
standarising the data so it is output ready to be fed into the train_model.py
module.

Functions & classes included within this package:

- process_and_calc_median
- min_max_scale
- connect_to_db
- pull_data_from_db
- analyte_data_from_db
- experiment_data_from_db
- plate_data_from_db
- live_upload_to_db
- standardise_data
- array_processing
- gaussian_plate_test
- human_serum_quality_check
- apply_median_per_analyte

AUTHOR:
    Tania LaGambina
********************************************************************************
"""
import pandas as pd
import numpy as np
import seaborn as sns
import copy
import os

def process_and_calc_median(queried_data):
    """
    This function pivots the raw data that is directly returned from the
    database into a useable format. As well as doing this, it also calculates
    the median for each individual feature. It is a rework of process_and_calc_median.
    It no longer treats different dyes seperately, but instead treats every
    peptide-dye combination separately.

    INPUTS:
    - queried_data: Data in the exact form that it as come off the database

    OUTPUTS:
    - processed_data: Data in a feature x readings format, ready to be scaled

    AUTHOR:
        Tania LaGambina
    """
    data = copy.deepcopy(queried_data)

    data['aHB'] = data['PEPTIDE_ID']+' '+data['DYE_ID']
    data.drop(columns=['PEPTIDE_ID', 'DYE_ID'], inplace=True, axis=1)
    overflow_measurements = copy.deepcopy(data.loc[data.FLUORESCENCE==260000])
    if not overflow_measurements.empty:
        print('Overflow measurements found for:')
        overflow_measurements['name'] = 'Plate: '+overflow_measurements['EXPERIMENT_ID']+', Analyte: '+overflow_measurements['ANALYTE_ID']+', Peptide: '+overflow_measurements['aHB']
        print(overflow_measurements['name'].values)
        data.replace({260000:np.nan}, inplace=True)
    processed_data = data.pivot_table(
        index=['EXPERIMENT_ID', 'ANALYTE_ID'],
        columns='aHB',
        values='FLUORESCENCE',
        aggfunc=np.median
        )
    processed_data.reset_index(inplace=True)
    processed_data.columns.name=None

    return processed_data

def min_max_scale(processed_data):
    """
    This function takes the processed data and min max scales the data across
    the reading. This allows us to look at the dye displacement in a clearer
    more relative way. This is a reworked version of min_max_scale. It disregards
    the need for the scaling to be done seperately per dye. It instead takes the aHB
    input from process_and_calc_median_2.

    INPUTS:
    - processed_data: Data in a feature x readings format, which is produced
        from the process_and_calc_median function

    OUTPUTS:
    - scaled_processed_data: The processed data, but min_max_scaled.

    AUTHOR:
        Tania LaGambina
    """
    full_scaled_data = {}
    experiments = [experiment for experiment in list(processed_data.EXPERIMENT_ID.unique())]
    for experiment in experiments:
        print('Processing plate: {}'.format(experiment))
        experiment_data = copy.deepcopy(
            processed_data[processed_data['EXPERIMENT_ID']==experiment])
        experiment_data.dropna(axis=1, inplace=True)
        # print(experiment_data)

        try:
            experiment_blank_data = copy.deepcopy(
                experiment_data[experiment_data['ANALYTE_ID']=='Blank'])
        except KeyError:
            raise PlateLayoutError('No blank readings included on plate')
        scaled_plate = {}
        analytes = [analyte for analyte in list(experiment_data['ANALYTE_ID']) if analyte != 'Blank']
        # min_max_data = pd.DataFrame(columns=['EXPERIMENT_ID', 'ANALYTE_ID'])

        for analyte in analytes:
            min_max_data = pd.DataFrame(columns=['EXPERIMENT_ID', 'ANALYTE_ID'])
            analyte_experiment_data = copy.deepcopy(
                experiment_data[experiment_data['ANALYTE_ID']==analyte])

            for column in (list(analyte_experiment_data.columns)):
                if (not column in ['ANALYTE_ID', 'EXPERIMENT_ID', 'No Pep None']
                    ):

                    dye = column.split(' ')[-1]
                    min_fluor_an = analyte_experiment_data['No Pep'+' '+dye].iloc[0]
                    # print(column)
                    max_fluor = experiment_blank_data[column].iloc[0]
                    min_fluor_c = experiment_blank_data['No Pep'+' '+dye].iloc[0]
                    val = analyte_experiment_data[column].astype('float').values[0]
                    if (max_fluor - min_fluor_c) != 0:
                        scaled_val = (val-min_fluor_an) / (max_fluor-min_fluor_c)
                    elif val is np.nan:
                        scaled_val = np.nan
                    else:
                        if column == 'No Pep'+' '+dye:
                            # print(val)
                            # print(min_fluor_an)
                            # print(max_fluor)
                            scaled_val = (val-min_fluor_an) / (max_fluor-min_fluor_an)
                            # print(scaled_val)
                        else:
                            scaled_val = np.nan
                            print('Warning - min and max fluor readings for analyte {},'
                                ' peptide {} and dye {} on plate are the same'.format(analyte, column, dye))
                    # print(column)
                    if len(min_max_data) == 0:
                        min_max_data = pd.DataFrame(
                        {'EXPERIMENT_ID':experiment, 'ANALYTE_ID':analyte, column:scaled_val}, index=[0])
                        # print(min_max_data)
                    else:
                        min_max_data = min_max_data.merge(pd.DataFrame(
                            {'EXPERIMENT_ID':experiment, 'ANALYTE_ID':analyte, column:scaled_val}, index=[0]),
                                                           on=['ANALYTE_ID', 'EXPERIMENT_ID'])
            # print(min_max_data)
            scaled_plate[analyte] = min_max_data

        try:
            scaled_data_dyes_combined = pd.concat(scaled_plate)
            # print(scaled_data_dyes_combined)
        except:
            scaled_data_dyes_combined = pd.DataFrame.from_dict(scaled_plate)
        scaled_data_dyes_combined.reset_index(drop=True, inplace=True)
        scaled_data_dyes_combined.columns.name = None
        full_scaled_data[experiment]=scaled_data_dyes_combined

    scaled_processed_data = pd.concat(full_scaled_data)
    scaled_processed_data.reset_index(inplace=True, drop=True)
    scaled_processed_data.columns.name = None

    return scaled_processed_data

def connect_to_db(username):
    """
    This function creates the connection to the Oracle Cloud ATP database. The
    type of connection is determined by the username and password, as different
    users have different permissions. The conn object is returned which is used
    later to perform queries to access data from the database.

    INPUTS:
    - username: The user that wants to access the database account, i.e. 'lab'
    - password: The password associated with that particular user account

    OUTPUTS:
    - conn: The connection object which is needed to perform queries on the
        database
    - db_connection: This is the connection object that is needed to use the
        'engine' that can upload data to the database, instead of just querying
        it

    AUTHOR:
        Tania LaGambina
    """

    import cx_Oracle
    from sqlalchemy import create_engine
    import getpass

    print('Connecting to database...')
    # Making the connection to the database
    user = username
    password = getpass.getpass('Password: ')
    sid = 'rosadb_low'
    ORACLE_HOME = os.getenv('ORACLE_HOME')
    try:
        cx_Oracle.init_oracle_client(lib_dir=ORACLE_HOME)
    except Exception as err:
        # print("Whoops!")
        print(err)
    os.environ['TNS_ADMIN'] = ORACLE_HOME+'/network/admin'
    conn = cx_Oracle.connect(user, password, sid)
    cstr = 'oracle+cx_oracle://{user}:{password}@{sid}'.format(
        user=user,
        password=password,
        sid=sid
        )
    engine = create_engine(
        cstr,
        convert_unicode=False,
        pool_recycle=10,
        pool_size=50,
        echo=True
        )
    db_connection = engine.connect()
    print('Connection made!')

    return conn, db_connection

def pull_data_from_db(conn, project):
    """
    This function performs a query to pull data from the database into the format
    which we need it, using joins between the various related tables. At this
    time, the data to be downloaded can be defined by the PROJECT_ID
    associated with it.

    * This function was used to extract data for the paper. However, the data is
    provided instead for external use instead of an active database connection.
    This function is here for reference only!

    INPUTS:
    - conn: The database connection object. This can be obtained from the
        connect_to_db function
    - project: The string containing the project name for the data which the
        user wishes to access. For example, 'Urine'

    OUTPUTS:
    - queried_data: The raw data pulled from the database, ready to be processed.

    AUTHOR:
        Tania LaGambina
    """
    query = """
        SELECT
            ADMIN.PLATE.PLATE_ID,
            ADMIN.PLATE.FLUORESCENCE,
            ADMIN.PEPTIDE_LAYOUT.PEPTIDE_ID,
            ADMIN.DYE_LAYOUT.DYE_ID,
            ANALYTE_ASSIGNMENTS.ANALYTE_ID,
            ADMIN.PLATE.ROW_LOC,
            ADMIN.PLATE.COLUMN_LOC
        FROM ADMIN.PLATE
        JOIN ADMIN.EXPERIMENT
            ON ADMIN.PLATE.PLATE_ID=ADMIN.EXPERIMENT.PLATE_ID
        JOIN ADMIN.PEPTIDE_LAYOUT
            ON ADMIN.PLATE.ROW_LOC=ADMIN.PEPTIDE_LAYOUT.ROW_LOC
            AND ADMIN.PLATE.COLUMN_LOC=ADMIN.PEPTIDE_LAYOUT.COLUMN_LOC
            AND ADMIN.EXPERIMENT.PEPTIDE_LAYOUT_ID=ADMIN.PEPTIDE_LAYOUT.PEPTIDE_LAYOUT_ID
        JOIN ADMIN.DYE_LAYOUT
            ON ADMIN.PLATE.ROW_LOC=ADMIN.DYE_LAYOUT.ROW_LOC
            AND ADMIN.PLATE.COLUMN_LOC=ADMIN.DYE_LAYOUT.COLUMN_LOC
            AND ADMIN.EXPERIMENT.DYE_LAYOUT_ID=ADMIN.DYE_LAYOUT.DYE_LAYOUT_ID
        JOIN ADMIN.ANALYTE_LAYOUT
            ON ADMIN.PLATE.ROW_LOC=ADMIN.ANALYTE_LAYOUT.ROW_LOC
            AND ADMIN.PLATE.COLUMN_LOC=ADMIN.ANALYTE_LAYOUT.COLUMN_LOC
            AND ADMIN.EXPERIMENT.ANALYTE_LAYOUT_ID=ADMIN.ANALYTE_LAYOUT.ANALYTE_LAYOUT_ID
        JOIN (
            select plate_id, analyte_1 as ANALYTE_ID, 'Analyte 1' as value from admin.experiment
                where analyte_1 is not null
            union all
            select plate_id, analyte_2 as ANALYTE_ID, 'Analyte 2' as value from admin.experiment
                where analyte_2 is not null
            union all
            select plate_id, analyte_3 as ANALYTE_ID, 'Analyte 3' as value from admin.experiment
                where analyte_3 is not null
            union all
            select plate_id, analyte_4 as ANALYTE_ID, 'Analyte 4' as value from admin.experiment
                where analyte_4 is not null
            union all
            select plate_id, 'Blank' as ANALYTE_ID, 'Control (H2O)' as value from admin.experiment
            ) analyte_assignments

            ON admin.analyte_layout.analyte_placeholder = analyte_assignments.value
            AND admin.experiment.plate_id = analyte_assignments.plate_id

        WHERE ADMIN.EXPERIMENT.PROJECT_ID = '{}'
    """.format(project)

    queried_data = pd.read_sql(query, conn)

    return queried_data

def analyte_data_from_db(conn, df, info):
    """
    This function extracts data from the analyte_inventory table in the database
    which contains the meta data for samples using the analyte_id column

    INPUTS:
    - conn: The database connection object. This can be obtained from the
        connect_to_db function
    - df: The dataframe which you wish to perform the join. Note, it must
        contain analyte_id as a column
    - info: The info which the user wishes to extract from the analyte
        inventory. E.g. diagnosis

    OUTPUTS:
    - queried_data: The raw data pulled from the database, ready to be processed.

    AUTHOR:
        Tania LaGambina
    """
    if info.upper() in df.columns:
        print("There is already a column with this name in the dataframe")
        joined_df = df
    else:
        query = """
            select
                admin.analyte_inventory.analyte_id,
                admin.analyte_inventory.{}
            from admin.analyte_inventory
        """.format(info)
        analyte_data = pd.read_sql(query, conn)
        joined_df = df.join(analyte_data.set_index('ANALYTE_ID'), on='ANALYTE_ID')

    return joined_df

def experiment_data_from_db(conn, df, info, plate_id_column):
    """
    This function extracts data from the experiment table in the database.
    This table contains most of the information from the plate_inventory excel
    sheet relating to the experiment itself (not the plate making)

    INPUTS:
    - conn: The database connection object. This can be obtained from the
        connect_to_db function
    - df: The dataframe which you wish to perform the join. Note, it must
        contain plate_id as a column
    - info: The info which the user wishes to extract from the plate
        inventory. E.g. Plate_batch

    OUTPUTS:
    - joined_df: The raw data pulled from the database, ready to be processed.

    AUTHOR:
        Tania LaGambina
    """
    if info.upper() in df.columns:
        print("There is already a column with this name in the dataframe")
        joined_df = df
    else:
        query = """
            select
                admin.experiment.plate_id,
                admin.experiment.{}
            from admin.experiment
        """.format(info)
        experiment_data = pd.read_sql(query, conn)
        joined_df = df.join(experiment_data.set_index('PLATE_ID'), on=plate_id_column)

    return joined_df

def plate_data_from_db(conn, df, info, plate_id_column):
    """
    This function extracts data from the plate_inventory table in the database.
    This table contains information from the plate_inventory excel sheet that is
    related to the specific plates and plate making

    INPUTS:
    - conn: The database connection object. This can be obtained from the
        connect_to_db function
    - df: The dataframe which you wish to perform the join. Note, it must
        contain plate_id as a column
    - info: The info which the user wishes to extract from the plate
        inventory. E.g. Plate_batch

    OUTPUTS:
    - queried_data: The raw data pulled from the database, ready to be processed.

    AUTHOR:
        Tania LaGambina
    """
    if info.upper() in df.columns:
        print("There is already a column with this name in the dataframe")
        joined_df = df
    else:
        query = """
            select
                admin.plate_inventory.plate_id,
                admin.plate_inventory.{}
            from admin.plate_inventory
        """.format(info)
        experiment_data = pd.read_sql(query, conn)
        # print(analyte_data)
        joined_df = df.join(experiment_data.set_index('PLATE_ID'), on=plate_id_column)

    return joined_df

def array_processing(data):
    """
    This function is a rework of the function array_2_processing. It uses the
    new min_max_scale_2 function and process_and_calc_median_2 function to
    process the data in a more streamlined way, and can be applied to all
    arrays of any types. It does this by treating each peptide-dye
    combination as one entity (so the previous limitations of using different
    dyes is no longer an issue)

    INPUTS:
        data: The raw data in the format straight from the database.
            A dataframe with the columns: PLATE_ID, FLUORESCENCE, ANALYTE_ID,
            ROW_LOC, COLUMN_LOC, PEPTIDE_ID, DYE_ID.
    OUTPUTS:
        parsed_data: The processed, min max scaled and transformed data in a
            dataframe format suitable for data analysis

    AUTHOR:
        Tania LaGambina
    """
    try:
        data.rename(columns={'PLATE_ID':'EXPERIMENT_ID'}, inplace=True)
    except:
        pass

    data.replace({
         'Buffer':'No Pep',
         'FRET C7':'DPH+C7',
         'FRET NR':'DPH+NR',
         'DPH + C7':'DPH+C7',
         'DPH + NR':'DPH+NR',
         'Sq III':'Sq-III'
        }, inplace=True)
    data.drop_duplicates(
        ['ROW_LOC', 'COLUMN_LOC', 'EXPERIMENT_ID'],
        keep='last',
        inplace=True
        )

    processed_data = process_and_calc_median(data)
    parsed_data = min_max_scale(processed_data)

    return parsed_data

def gaussian_plate_test(queried_data):
    """
    This function tests for systematic errors across a plate by testing for a
    gaussian distribution in individual 'instances' - these being technical
    repeats of peptide-dye-analyte combinations. It works on the assumption that
    a plate without systematic errors would have a gaussian distribution for
    these individual instances.
    Two different tests are used to test for a gaussian distribution. These are
    the D'Agostino's k^2 Test and the Shapiro-Wilk test. Note, if the number
    of technical repeats across the plate are less than 8, only the Shapiro-
    Wilk test will be performed.

    INPUTS:
        - queried_data: The unprocessed data, extracted straight from the
                    database

    AUTHOR:
        Tania LaGambina
    """
    from scipy.stats import shapiro, normaltest
    from collections import Counter

    non_blank_queried_data = queried_data.loc[~queried_data['ANALYTE_ID'].isin(['Blank'])]
    testing_queried_data = copy.deepcopy(
        non_blank_queried_data[['EXPERIMENT_ID', 'FLUORESCENCE', 'PEPTIDE_ID', 'DYE_ID', 'ANALYTE_ID']]
    )
    testing_queried_data['INSTANCE'] = testing_queried_data['EXPERIMENT_ID']+'_'+testing_queried_data['PEPTIDE_ID']+'_'+testing_queried_data['DYE_ID']+'_'+testing_queried_data['ANALYTE_ID']
    dictionary = {}
    for i in testing_queried_data['INSTANCE'].unique():
        instance_testing_queried_data = testing_queried_data.loc[testing_queried_data['INSTANCE'].isin([i])]
        plate = instance_testing_queried_data['EXPERIMENT_ID']
        instance_fluorescence = instance_testing_queried_data['FLUORESCENCE'].tolist()
        if len(instance_fluorescence) < 8:
            stat2, p2 = shapiro(instance_fluorescence)
            if p2<0.01:
                dictionary[i]=plate.unique()[0]
            else:
                pass
        else:
            stat1, p1 = normaltest(instance_fluorescence)
            stat2, p2 = shapiro(instance_fluorescence)
            if p1<0.01 and p2<0.01:
                dictionary[i]=plate.unique()[0]
            else:
                pass
    counts = Counter(dictionary.values())
    for key, v in counts.items():
        if v > 4:
            print('Half or more of plate {} is not Gaussian'.format(key))


def human_serum_quality_check(queried_data):
    """
    This function performas a quality control check on human serum data. It
    checks that water blanks are within an accepted limit determined by the
    human serum data, as well as the human serum wells being within the
    accepted limit. The accepted limit is determined as lower limit = q1-1.5*iqr
    and the upper limit = q3+1.5*iqr (from the poc human serum data).
    Note, it only works when using human serum data on plates with peptide-dye
    layout H.

    AUTHOR:
        Tania LaGambina
    """

    queried_data['aHB'] = queried_data['PEPTIDE_ID']+' '+queried_data['DYE_ID']
    queried_data['POSITION'] = queried_data['ROW_LOC'].astype('str')+' '+queried_data['COLUMN_LOC'].astype('str')

    u_lim = pd.read_csv('/Users/tanialagambina/rosaBiotech/layout_limits/upper_limit_layout_H.csv')
    u_lim.columns = ['POSITION', 'UPPER_LIMIT']

    l_lim = pd.read_csv('/Users/tanialagambina/rosaBiotech/layout_limits/lower_limit_layout_H.csv')
    l_lim.columns = ['POSITION', 'LOWER_LIMIT']

    queried_data_ul = queried_data.join(u_lim.set_index('POSITION'), on='POSITION')
    queried_data_ulll = queried_data_ul.join(l_lim.set_index('POSITION'), on='POSITION')

    outliers = queried_data_ulll.loc[~queried_data_ulll.FLUORESCENCE.between(
        queried_data_ulll.LOWER_LIMIT,
        queried_data_ulll.UPPER_LIMIT
    )]

    outlier_counts = pd.DataFrame(outliers['PLATE_ID'].value_counts())

    # outlier_plates = outlier_counts.loc[outlier_counts.PLATE_ID>80]
    # outlier_plates

    print(outlier_counts)

    return outlier_counts

def apply_median_per_analyte(parsed_data):
    """
    Function to take the median of a parsed_data dataframe without the need
    to drop meta data to do so, and retain this information afterwards. The
    median is taken as a median across the individual analytes, so only one
    row in the dataframe per analyte (instead of the number of plate repeats)

    AUTHOR:
        Tania LaGambina
    """
    array_cols = parsed_data.filter(regex='NR|DPH|C7|SR|DiOC|5-DAF|Sq-III|Flipper').columns
    data_to_median = parsed_data[[*array_cols, 'ANALYTE_ID']]
    data_to_keep = parsed_data[parsed_data.columns.difference([*array_cols, 'EXPERIMENT_ID'])]
    data_to_keep.drop_duplicates(inplace=True)

    median_data = data_to_median.groupby('ANALYTE_ID').median()
    median_parsed_data = median_data.join(data_to_keep.set_index('ANALYTE_ID'), on='ANALYTE_ID')
    median_parsed_data['ANALYTE_ID'] = median_parsed_data.index
    median_parsed_data.reset_index(drop=True, inplace=True)

    return median_parsed_data
