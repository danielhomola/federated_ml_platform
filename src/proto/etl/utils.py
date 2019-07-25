import os
import logging

import psycopg2
import sshtunnel
import pandas as pd

logger = logging.getLogger(__name__)


def connect_to_db_via_ssh(ssh_info, db_info):
    """
    Connects to a remote PostgreSQL db, via SSH tunnel.

    Args:
        ssh_info (obj): All ssh connection info.
        db_info (obj): All db related connection info.

    Returns:
        :class:`psycopg2.extensions.connection`: Live connection suitable for queries.
    """
    tunnel = sshtunnel.SSHTunnelForwarder(
         ssh_info.host,
         ssh_private_key=ssh_info.ssh_private_key,
         ssh_username=ssh_info.ssh_username,
         remote_bind_address=ssh_info.remote_bind_address
    )

    tunnel.start()
    logger.info("SSH tunnel connected.")

    conn = psycopg2.connect(
        database=db_info.db_name,
        user=db_info.db_user,
        password=db_info.db_password,
        host=tunnel.local_bind_host,
        port=tunnel.local_bind_port
    )
    logger.info("Postgres database %s connected" % db_info.db_name)
    return conn


def run_eicu_query(query, conn):
    """
    Runs a SQL query, with the appropriate search path for the eICU database.
    Returns a pandas DataFrame with the results.

    Args:
        query (str): SQL query to be executed.
        conn (:class:`psycopg2.extensions.connection`): Established psycopg2 connection.

    Returns:
        :class:`pandas.DataFrame`: DataFrame with the results of the query.
    """
    query_schema = "set search_path to eicu_crd;"
    query = query_schema + query
    return pd.read_sql_query(query, conn)


def get_column_completeness(table_name, column_names, conn, verbose=True):
    """
    Takes a PostgreSQL table and a list of column names and returns a pandas Series
    with the percentage completeness of all columns.

    Args:
        table_name (str): Name of the table to use.
        column_names (list<str>): Column names to use.
        conn (:class:`psycopg2.extensions.connection`): Established psycopg2 connection.
        verbose (bool): If False, we don't print out anything during the computation.

    Returns:
        :class:`pandas.Series`: Series with the percentage of non missing for each col.
    """
    percentages = []
    for column_name in column_names:
        query ="""select 100.0 * count({col}) / count(1) as percentnotnull
        from {table};""".format(col=column_name, table=table_name)
        df = run_eicu_query(query, conn)
        percentage = df['percentnotnull'].values[0]
        percentages.append(percentage)
        if verbose:
            logger.debug('%s: %f' % (column_name, percentage))
    return pd.Series(percentages, index=column_names)


def load_schema_for_modelling():
    """
    Loads the cleaned and edited schema (as a csv) of the tables used for modelling.

    Returns:
        :class:`pandas.DataFrame`: DF with the schema columns (indexed by table names).
    """
    filename = "modelling_schema.csv"
    folder = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(folder, filename)
    return pd.read_csv(path).set_index('table_name')
