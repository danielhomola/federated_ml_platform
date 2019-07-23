import logging

import psycopg2
from sshtunnel import SSHTunnelForwarder

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
    tunnel = SSHTunnelForwarder(
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

