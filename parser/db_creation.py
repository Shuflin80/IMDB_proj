import psycopg2
import os
import psycopg2.extras
from dotenv import load_dotenv
from typing import Iterable


load_dotenv()

db_host = os.environ.get('host')
db_database = os.environ.get('database', 'postgres')
db_user = os.environ.get('user', 'postgres')
db_password = os.environ.get('password')

connection_dict = {'dbname': db_database, 'user': db_user, 'host': db_host, 'password': db_password}


def sql_create_tables(close_conn: bool = True):
    conn = None
    try:
        conn = psycopg2.connect(**connection_dict)
        cur = conn.cursor()

        cur.execute('''CREATE TABLE IF NOT EXISTS actors (act_id VARCHAR(20) PRIMARY KEY, 
                                            name VARCHAR(80));''')

        cur.execute('''CREATE TABLE IF NOT EXISTS movies  (mov_id VARCHAR(20) PRIMARY KEY, 
                                            title VARCHAR(80),
                                            description VARCHAR(2000));''')

        cur.execute("""CREATE TABLE IF NOT EXISTS relations (mov_id VARCHAR(20), 
                                            act_id VARCHAR(20),
                                            roles VARCHAR(1000),
                                            CONSTRAINT act
                                                FOREIGN KEY(act_id)
                                                    REFERENCES actors(act_id),
                                            CONSTRAINT mov
                                                FOREIGN KEY(mov_id)
                                                    REFERENCES movies(mov_id));""")

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    finally:
        if conn is not None:
            conn.commit()
            if close_conn:
                conn.close()


def dump_table(vals_list: Iterable, table_name: str, conn=None, close_conn: bool = False):
    if not conn:
        close_conn = True
        conn = psycopg2.connect(**connection_dict)
    with conn.cursor() as cursor:
        if table_name == 'actors':
            sql_line = f"""INSERT INTO {table_name} VALUES (%s, %s);"""
        else:
            sql_line = f"""INSERT INTO {table_name} VALUES (%s, %s, %s);"""
        psycopg2.extras.execute_batch(cursor, sql_line, vals_list)
    conn.commit()
    if close_conn:
        conn.close()

    return set()


def truncate_tables(conn=None, close_conn: bool = False):
    if not conn:
        close_conn = True
        conn = psycopg2.connect(**connection_dict)
    with conn.cursor() as cursor:
        cursor.execute('''TRUNCATE TABLE movies CASCADE;''')
        cursor.execute('''TRUNCATE TABLE actors CASCADE;''')
        cursor.execute('''TRUNCATE TABLE relations;''')

    conn.commit()
    if close_conn:
        conn.close()

