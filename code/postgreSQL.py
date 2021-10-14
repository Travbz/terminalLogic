import os
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras as extras
from sqlalchemy import create_engine
from config import user, password, host, database, param_dic
from typing import Optional, Dict, Any, List
from ciso8601 import parse_datetime
from requests import Request, Session, Response
import sys
import hmac
import urllib.parse
import time

class PostgreSQL:
    """Use for writing api data to db"""
    nb_start = time.time() # measures number of seconds since 1970

    # print time record status
    print(f'Process start time recorded')
    print(f'---')

        # create connector for pandas.to_sql() method for loading data to database
    engine = create_engine(f'postgresql://{user}:{password}@{host}:5432/{database}')

    # print connector status
    print(f'Database connector created')

    conn = connect(param_dic)

    # print partition from connection status
    print(f'-')


    def connect(params):
        conn = None
    try:
        print(f'Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        # create cursor to run query
        cur = conn.cursor()
        # return database version
        cur.execute('SELECT version();')
        # fetch one result
        record = cur.fetchone()
        print('You are connected to -', record)
        # close cursor
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print('Error while connecting to PostgreSQL database', error)
    # close connection
    finally:
        if (conn):
            cur.close()
            conn.rollback()
            return conn



    def create_schema(sql_file):
        cursor = conn.cursor()
        try:
            print(f'Creating schema in database using {sql_file}...')
            cursor.execute(open(sql_file, 'r').read())
            print('Schema successfully created!')
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print('Error while creating schema in database', error)
            conn.rollback()
        # close cursor
        cursor.close()

    create_schema('queries/create_schema.sql')

    def load_data(conn, df, table):
        start = time.time()
        print(f'Staging {table}...')
        print(f'-')
        # create list of tupples from dataframe values
        tuples = [tuple(x) for x in df.to_numpy()]
        # comma-separated dataframe columns
        cols = ','.join(list(df.columns))
        # SQL query to execute insert
        query  = 'INSERT INTO %s(%s) VALUES %%s' % (table, cols)
        cursor = conn.cursor()
        try:
            print(f'Loading {table} to the PostgreSQL database...')
            extras.execute_values(cursor, query, tuples)
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print(f'Error: %s' % error)
            print(f'-')
            print(f'Load failed.')
            print(f'---')
            conn.rollback()
            cursor.close()
            return 1
        end = time.time()
        seconds = round((end - start) % 60)
        minutes = round(((end - start) - seconds) / 60)
        print(f'Load successful!  Run Time: {minutes}min {seconds}sec')
        print(f'---')
        cursor.close()


    # to sql function to use pandas to_sql function with status print
def to_sql(df, table):
    start = time.time()
    print(f'Staging {table}...')
    print(f'-')
    try:
        print(f'Loading {table} to the PostgreSQL database...')
        # use pd.to_sql() to set load details
        df.to_sql(
            table, 
            con = engine, 
            schema = 'public', 
            if_exists = 'append',
            index = False
        )
    except (Exception, psycopg2.DatabaseError) as error:
        print(f'Error: %s' % error)
        print(f'-')
        print(f'Load failed.')
        print(f'---')
        return 1
    end = time.time()
    seconds = round((end - start) % 60)
    minutes = round(((end - start) - seconds) / 60)
    print(f'Load successful!  Run Time: {minutes}min {seconds}sec')
    print(f'---')


    

    

    


    