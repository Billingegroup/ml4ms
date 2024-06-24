"""Helps manage mongodb setup and connections."""

import os
from contextlib import contextmanager

from ml4ms.fsclient import FileSystemClient


def load_local_database(db_info, client, rc):
    """Loads a local database"""
    # make sure that we expand user stuff
    db_info["url"] = os.path.expanduser(db_info["url"])
    # import all of the data
    client.load_database(db_info)


def load_mongo_database(db, client):
    """Load a mongo database."""
    client.load_database(db)


def load_database(db_info, client, rc):
    """Loads a database"""
    if rc.client in ("mongo", "mongodb"):
        load_mongo_database(db_info, client)
        return
    url = db_info["url"]
    if os.path.exists(os.path.expanduser(url)):
        load_local_database(db_info, client, rc)
    else:
        raise ValueError("Do not know how to load this kind of database: " "{}".format(db_info))


def dump_local_database(db, client, rc):
    """Dumps a local database"""
    # dump all of the data
    client.dump_database(db)
    return


def dump_database(db_info, client, rc):
    """Dumps a database"""
    # do not dump mongo db
    if rc.client in ("mongo", "mongodb"):
        return
    url = db_info["url"]
    if os.path.exists(url):
        dump_local_database(db_info, client, rc)
    else:
        raise ValueError("Do not know how to dump this kind of database")


def open_dbs(rc, colls=None):
    """Open the databases

    Parameters
    ----------
    rc : RunControl instance
        The rc which has links to the dbs
    colls: set or None, optional
        The collections to load. If None load all, defaults to None

    Returns
    -------
    client : {FileSystemClient, MongoClient}
        The database client containing connected dbs
    """
    if rc.client == "fs":
        client = FileSystemClient(rc)
    else:  # we only have one client atm...but may want to change to mongo later
        client = FileSystemClient(rc)
    client.open()
    load_database(rc.database_info, client, rc)
    return client


@contextmanager
def connect(rc, colls=None):
    """Context manager for ensuring that database is properly setup and torn
    down"""
    client = open_dbs(rc, colls=colls)
    yield client
    dump_database(rc.database_info, client, rc)
    client.close()
