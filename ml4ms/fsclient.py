"""Contains a client database backed by the file system."""

import datetime
import json
import logging
import os
import signal
import sys
from collections import defaultdict
from copy import deepcopy
from glob import iglob
from pathlib import Path

import ruamel.yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq


def dbpathname(db_info=None):
    dbdir = Path(db_info["url"])
    dbpath = dbdir / db_info["path"]
    return dbpath


class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug("SIGINT received. Delaying KeyboardInterrupt.")

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


YAML_BASE_MAP = {CommentedMap: dict, CommentedSeq: list}


def _rec_re_type(i):
    """Destroy this when ruamel.yaml supports basetypes again"""
    if type(i) in YAML_BASE_MAP:
        base = YAML_BASE_MAP[type(i)]()
        if isinstance(base, dict):
            for k, v in i.items():
                base[_rec_re_type(k)] = _rec_re_type(v)
        elif isinstance(base, list):
            for j in i:
                base.append(_rec_re_type(j))
    else:
        base = i
    return base


def _id_key(doc):
    return doc.get("_id")


def load_json_collection(filename):
    """Loads a collection in hte form of a set of json objects, one per line in the
     file and returns a dict of its documents.

    Expects one document per line in the file with the form:
    {'_id': '<id>', 'field1':'value1', 'field2':'value2'}
    """
    with open(filename, encoding="utf-8") as fh:
        lines = fh.readlines()
    docs = {}
    for line in lines:
        doc = json.loads(line)
        try:
            doc["date"] = datetime.date.fromisoformat(doc["date"])
        except KeyError:
            pass
        try:
            doc["datetime"] = datetime.datetime.fromisoformat(doc["datetime"])
        except KeyError:
            pass
        docs[doc["_id"]] = doc
    return docs


def date_encoder(obj):
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    else:
        return obj


def dump_json_collection(filename, docs, date_handler=None):
    """Dumps a dict of documents into a file as a list of json objects"""
    if date_handler is None:
        date_handler = date_encoder
    docs = sorted(docs.values(), key=_id_key)
    lines = [json.dumps(doc, sort_keys=True, default=date_handler) for doc in docs]
    s = "\n".join(lines)
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(s)


def load_yaml(filename, return_inst=False, loader=None):
    """Loads a YAML file and returns a dict of its documents."""
    if loader is None:
        inst = YAML()
    else:
        inst = loader
    with open(filename, encoding="utf-8") as fh:
        docs = inst.load(fh)
        docs = _rec_re_type(docs)
    for _id, doc in docs.items():
        doc["_id"] = _id
    return (docs, inst) if return_inst else docs


def dump_yaml(filename, docs, inst=None):
    """Dumps a dict of documents into a file."""
    inst = YAML() if inst is None else inst
    inst.representer.ignore_aliases = lambda *data: True
    inst.indent(mapping=2, sequence=4, offset=2)
    sorted_dict = ruamel.yaml.comments.CommentedMap()
    for k in sorted(docs):
        doc = docs[k]
        _ = doc.pop("_id")
        sorted_dict[k] = ruamel.yaml.comments.CommentedMap()
        for kk in sorted(doc.keys()):
            sorted_dict[k][kk] = doc[kk]
    with open(filename, "w", encoding="utf-8") as fh:
        with DelayedKeyboardInterrupt():
            inst.dump(sorted_dict, stream=fh)


def json_to_yaml(inp, out):
    """Converts a JSON collection file to a YAML one."""
    docs = load_json_collection(inp)
    dump_yaml(out, docs)


def yaml_to_json(inp, out, loader=None):
    """Converts a YAML file to a JSON one."""
    docs = load_yaml(inp, loader=loader)
    dump_json_collection(out, docs)


class FileSystemClient:
    """A client database backed by the file system."""

    def __init__(self, rc):
        self.db = None
        self.rc = rc
        self.closed = True
        self.open()
        self._collfiletypes = {}
        self._collexts = {}
        self._yamlinsts = {}

    def is_alive(self):
        return not self.closed

    def open(self):
        if self.closed:
            self.db = defaultdict(lambda: defaultdict(dict))
            self.chained_db = {}
            self.closed = False

    def load_json(self, db_info=None):
        """Loads the JSON collection part of a database.

        If no db is passed, take the first database in rc.databases
        """
        if db_info is None:
            db_info = self.rc.database_info.get("name")
        dbpath = dbpathname(db_info)
        for f in [
            file
            for file in iglob(os.path.join(dbpath, "*.json"))
            if file not in db_info.get("blacklist", [])
            and len(db_info.get("whitelist", [])) == 0
            or os.path.basename(file).split(".")[0] in db_info["whitelist"]
        ]:
            collfilename = os.path.split(f)[-1]
            base, ext = os.path.splitext(collfilename)
            self._collfiletypes[base] = "json"
            print("loading " + f + "...", file=sys.stderr)
            self.db[base] = load_json_collection(f)

    def load_yaml(self, db_info=None):
        """Loads the YAML part of a database.

        If no db is passed, take the first database in rc.databases
        """
        if db_info is None:
            db_info = self.rc.database_info.get("name")
        dbpath = dbpathname(db_info)
        for f in [
            file
            for file in iglob(os.path.join(dbpath, "*.y*ml"))
            if file not in self.db.get("blacklist", [])
            and len(self.db.get("whitelist", [])) == 0
            or os.path.basename(file).split(".")[0] in db_info["whitelist"]
        ]:
            collfilename = os.path.split(f)[-1]
            base, ext = os.path.splitext(collfilename)
            self._collexts[base] = ext
            self._collfiletypes[base] = "yaml"
            # print("loading " + f + "...", file=sys.stderr)
            coll, inst = load_yaml(f, return_inst=True)
            self.db[base] = coll
            self._yamlinsts[dbpath, base] = inst

    def load_database(self, db_info):
        """Loads a database.

        If no db is passed, take the first database in rc.databases
        """
        self.load_json(db_info=db_info)
        self.load_yaml(db_info=db_info)

    def dump_json(self, docs, collname, db=None):
        """Dumps json docs and returns filename

        If no db is passed, take the first database in rc.databases
        """
        if db is None:
            db = self.rc.database_info.get("name")
        dbpath = dbpathname(db)
        f = os.path.join(dbpath, collname + ".json")
        dump_json_collection(f, docs)
        filename = os.path.split(f)[-1]
        return filename

    def dump_yaml(self, docs, collname, db=None):
        """Dumps json docs and returns filename

        If no db is passed, take the first database in rc.databases
        """
        if db is None:
            db = self.rc.database_info.get("name")
        dbpath = dbpathname(db)
        f = os.path.join(dbpath, collname + self._collexts.get(collname, ".yaml"))
        inst = self._yamlinsts.get((dbpath, collname), None)
        dump_yaml(f, docs, inst=inst)
        filename = os.path.split(f)[-1]
        return filename

    def dump_database(self, db=None):
        """Dumps a database back to the filesystem.

        If no db is passed, take the first database in rc.databases
        """
        if db is None:
            db = self.rc.database_info.get("name")
        dbpath = dbpathname(db)
        os.makedirs(dbpath, exist_ok=True)
        to_add = []
        for collname, collection in self.db.items():
            # print("dumping " + collname + "...", file=sys.stderr)
            filetype = self._collfiletypes.get(collname, "json")
            if filetype == "json":
                filename = self.dump_json(collection, collname, db)
            elif filetype == "yaml":
                filename = self.dump_yaml(collection, collname, db)
            else:
                raise ValueError("did not recognize file type for regolith")
            to_add.append(os.path.join(db["path"], filename))
        return to_add

    def close(self):
        self.db = None
        self.closed = True

    def keys(self):
        return self.db.keys()

    def __getitem__(self, key):
        return self.db[key]

    def collection_names(self, db=None, include_system_collections=True):
        """Returns the collection names for a database.

        If no db is passed, take the database in rc.client
        """
        if db is None:
            db = self.rc.client.db
        return set(db.keys())

    def all_documents(self, collname, copy=True):
        """Returns an iteratable over all documents in a collection."""
        if copy:
            return deepcopy(self.client.db.get(collname, {})).values()
        return self.client.db.get(collname, {}).values()

    def insert_one(self, collname, doc, db=None):
        """Inserts one document to a database/collection.

        If no db is passed, take the database in rc.client
        """
        if db is None:
            db = self.rc.client.db
        coll = db[collname]
        coll[doc["_id"]] = doc

    def insert_many(self, collname, docs, db=None):
        """Inserts many documents into a database/collection.

        If no db is passed, take the database in rc.client
        """
        if db is None:
            db = self.rc.client.db
        coll = db[collname]
        for doc in docs:
            coll[doc["_id"]] = doc

    def delete_one(self, collname, doc, db=None):
        """Removes a single document from a collection

        If no db is passed, take the database in rc.client
        """
        if db is None:
            db = self.rc.client.db
        coll = db[collname]
        del coll[doc["_id"]]

    def find_one(self, collname, filter, db=None):
        """Finds the first document matching filter.

        If no db is passed, take the database in rc.client
        """
        if db is None:
            db = self.rc.client.db
        coll = db[collname]
        for doc in coll.values():
            matches = True
            for key, value in filter.items():
                if key not in doc or doc[key] != value:
                    matches = False
                    break
            if matches:
                return doc

    def find(self, collname, filter, db=None):
        """Finds and returns all the documents matching filter.

        If no db is passed, take the database in rc.client
        """
        if db is None:
            db = self.rc.db
        coll = db[collname]
        matches = []
        for doc in coll.values():
            if filter == {}:
                matches.append(doc)
            else:
                for key, value in filter.items():
                    if key in doc and doc[key] == value:
                        matches.append(doc)
        return matches

    def update_one(self, collname, filter, update, db=None, **kwargs):
        """Updates one document.

        If no db is passed, take the database in rc.client
        """
        if db is None:
            db = self.rc.client.db
        coll = db[collname]
        doc = self.find_one(collname, filter, db=self.db)
        newdoc = dict(filter if doc is None else doc)
        newdoc.update(update)
        coll[newdoc["_id"]] = newdoc
