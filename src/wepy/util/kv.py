"""Implement a key-value store on top of sqlite3 database."""

# Copyright (c) 2012, Alex Morega
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Software copied and modified heavily from this source


import os
import os.path as osp
import logging
import sqlite3
from collections.abc import MutableMapping
from contextlib import contextmanager

# mapping of the modes we support and the modes that SQLite provides
# KV mode -> sqlite3 mode
MODE_MAPPING = (
    ('r', 'ro'),
    ('r+', 'rw'),
    ('a', 'rwc'),
    ('x', None),
    ('w', None),
    ('w-', None),
)

# modes that sqlite3 itself doesn't support and have to manual
# processing for
TRUNCATE_MODES = ('w',)
FAIL_IF_EXISTS_CREATE_MODES = ('w-', 'x')

SQLITE3_URI_TEMPLATE = "{protocol}:{url}"
SQLITE3_QUERY_URI_TEMPLATE = "{protocol}:{url}?{query}"

SQLITE3_QUERY_JOIN_CHAR = "&"

SQLITE3_INMEMORY_URI = "file::memory:?cache=shared"


# the default types for the values that is checked, is bytes
DEFAULT_VALUE_TYPES = (bytes, bytearray)

def gen_uri(db_url, mode_spec):

    # if the db url is the in memory special string or None or the
    # :memory: identifier, use the full in-memory URI
    if (db_url == SQLITE3_INMEMORY_URI or
        db_url is None or
        db_url == ":memory:"):

        db_uri = SQLITE3_INMEMORY_URI

        return db_uri

    # check for and split the URI into the components: protocol, URL, query
    if len(db_url.split(':')) > 1:
        # for a query
        if len(db_url.split('?')) > 1:

            # protocol and queries
            ((protocol,), (url, query)) = [comp.split('?') for comp in
                                           db_url.split(':')]

        else:
            # no query
            protocol, url = db_url.split(':')
            query = ""
    else:
        protocol = ""
        if len(db_url.split('?')) > 1:
            url, query = db_url.split('?')
        else:
            query = ""
            url = db_url

    # process and build the URI given this information

    # if the protocol is not given set it to the default
    if len(protocol) == 0:
        protocol = 'file'

    # split the query up into sections if it has them
    if len(query.split('?')) > 1:
        queries = query.split('?')

        # split the "key=value" pairs in each query
        queries = {q.split('=')[0] : q.split('=')[0]
                   for q in queries}
    else:
        queries = {}

    # now handle the mode. If it was given as an argument then we have
    # to set the appropriate query in the URI. If it was given in the
    # URI that takes precedence however.

    # check for a mode option in the given query section, if no mode
    # is given then we set it depending on what mode_spec was given
    if 'mode' not in queries:

        # default to rwc mode
        mode_query_value = 'rwc'

        if mode_spec is not None:

            # if the mode is one of the modes in the mode mapping use that to
            # generate the URI, otherwise raise an error
            if mode_spec not in dict(MODE_MAPPING):

                raise ValueError("kv mode spec '{}' not recognized".format(mode_spec))

            else:

                sqlite_mode = dict(MODE_MAPPING)[mode_spec]

                # if the sqlite_mode is recognized as a mode that can
                # be given in the query section do that
                if sqlite_mode is not None:
                    mode_query_value = sqlite_mode

                # otherwise we handle these special modes ourselves
                # here, checking file properties and raising errors as
                # necessary
                else:

                    # if the mode is either 'x' or 'w-' check to see if the db
                    # already exists. If it does raise an error.
                    if mode_spec in FAIL_IF_EXISTS_CREATE_MODES:

                        if osp.exists(url):
                            raise OSError("File exists")

                    # if it is 'w' we want to delete the old file
                    elif mode_spec in TRUNCATE_MODES:

                        # if it exists remove it
                        if osp.exists(url):
                            os.remove(url)

                    # the sqlite_mode for these is rwc (a), since we
                    # need to create it
                    mode_query_value = "rwc"

        # add the mode query to queries list
        queries['mode'] = mode_query_value

    # if thw queries are empty just use the protocol and URL
    if not queries:
        db_uri = SQLITE3_URI_TEMPLATE.format(protocol=protocol,
                                             url=url)
    # otherwise do the whole thing
    else:

        # build the query substring
        query = SQLITE3_QUERY_JOIN_CHAR.join(["{}={}".format(key, value)
                                              for key, value in queries.items()])

        # build the URI string
        db_uri = SQLITE3_QUERY_URI_TEMPLATE.format(protocol=protocol,
                                                   url=url,
                                                   query=query)

    return db_uri

class KV(MutableMapping):

    def __init__(self, db_url=None,
                 table='data',
                 primary_key='key',
                 value_name='value',
                 timeout=5,
                 mode='x',
                 append_only=False,
                 value_types=DEFAULT_VALUE_TYPES,
    ):

        # generate a good URI from the url and the mode
        db_uri = gen_uri(db_url, mode)

        self._mode = mode
        self._append_only = append_only

        # set the value types for this kv
        self._kv_types = value_types


        self._db_uri = db_uri

        # connect to the db
        self._db = sqlite3.connect(self._db_uri, timeout=timeout, uri=True)
        self._closed = False

        # set the isolation level to autocommit
        self._db.isolation_level = None

        # we can use read_uncommited only in append_only mode (no
        # updates) because you never have to worry about dirty reads
        # since you can't update
        if self.append_only:
            self._execute("PRAGMA read_uncommited=1")


        self._table = table
        self._primary_key = primary_key
        self._value_name = value_name

        # create the table if it doesn't exist and set the key names
        create_table_query = """
            CREATE TABLE IF NOT EXISTS {table_name}
            ({key_name} PRIMARY KEY, {value_name})
        """.format(table_name=self.table,
                   key_name=self.primary_key,
                   value_name=self.value_name)
        self._execute(create_table_query)

        self._locks = 0

    @property
    def mode(self):
        return self._mode

    @property
    def append_only(self):
        return self._append_only

    def close(self):

        if self._closed == True:
            raise IOError("The database connection is already closed")

        else:
            self._db.close()
            self._closed = True


    @property
    def db_uri(self):
        return self._db_uri

    @property
    def db(self):
        return self._db

    @property
    def table(self):
        return self._table

    @property
    def primary_key(self):
        return self._primary_key

    @property
    def value_name(self):
        return self._value_name

    @property
    def value_types(self):
        return self._kv_types


    def _execute(self, *args):
        return self._db.cursor().execute(*args)

    def __len__(self):
        [[n]] = self._execute('SELECT COUNT(*) FROM {table}'.format(table=self.table))
        return n

    def __getitem__(self, key):

        if key is None:
            query = ('SELECT {value} FROM {table} WHERE {key} is NULL'.format(
                     value=self.value_name,
                     table=self.table,
                     key=self.primary_key), ())
        else:
            query = ('SELECT {value} FROM {table} WHERE {key}=?'.format(
                                                      value=self.value_name,
                                                      table=self.table,
                                                      key=self.primary_key),
                     (key,),)

        cursor = self._execute(*query)
        result = cursor.fetchone()

        if result is None:
            raise KeyError
        else:
            return result[0]

    def __iter__(self):
        return (key for [key] in self._execute('SELECT {key} FROM {table}'.format(
                                               key=self.primary_key,
                                               table=self.table),
                                               ()
                                              )
                )

    def __setitem__(self, key, value):
        """Set a value, must be in bytes format."""

        # check the type of the value to make sure it is what this KV
        # supports, if it is None then it is the standard python type
        # translation

        if self.value_types is not None:
            assert isinstance(value, self.value_types), "Value must be a value supported by this kv"

        self.lockless_set(key, value)


    def __delitem__(self, key):

        logging.debug("Deleting the snapshot {}".format(key))

        # no deletions in append only mode
        if self.append_only:
            raise sqlite3.IntegrityError("DB is opened in append only mode, "
                                         "and {} has already been set".format(key))

        # delete it if it exists
        elif key in self:
            logging.debug("executing delete query")
            self._execute(self.del_query,
                          (key,))
            logging.debug("finished")

        else:
            raise KeyError

    @property
    def insert_query(self):
        query = 'INSERT INTO {table} VALUES (?, ?)'.format(table=self.table)

        return query

    @property
    def update_query(self):

        query = 'UPDATE {table} SET {value}=? WHERE {key}=?'.format(
            key=self.primary_key,
            value=self.value_name,
            table=self.table)

        return query

    @property
    def del_query(self):

        query = 'DELETE FROM {table} WHERE {key}=?'.format(
            key=self.primary_key,
            table=self.table)

        return query

    def lockless_set(self, key, value):
        """an implementation of the __setitem__ without the lock context
        manager which turns on the DEFERRED isolation level. The
        isolation level of the KV is set to autocommit so now lock is
        needed anyhow.

        """

        # insert the key-value pair if the key isn't in the db
        try:
            self._execute(self.insert_query, (key, value))

        # otherwise update the keys value
        except sqlite3.IntegrityError:

            # if we are in append only mode don't allow updates
            if self.append_only:
                raise sqlite3.IntegrityError("DB is opened in append only mode, "
                                             "and {} has already been set".format(key))
            else:
                self._execute(self.update_query,
                              (value, key))

    def set_in_tx(self, cursor, key, value):
        """Do a set with a cursor, this allows it to be done in a transaction."""

        try:
            cursor.execute(self.insert_query, (key, value))
        except sqlite3.IntegrityError:

            # if we are in append only mode don't allow updates
            if self.append_only:
                raise sqlite3.IntegrityError("DB is opened in append only mode, "
                                             "and {} has already been set".format(key))

            else:
                cursor.execute(self.update_query, (key, value))

        return cursor

    def del_in_tx(self, cursor, key):

        # no deletions in append only mode
        if self.append_only:
            raise sqlite3.IntegrityError("DB is opened in append only mode, "
                                         "and {} has already been set".format(key))

        elif key in self:
            cursor.execute(self.del_query, (key,))

        else:
            raise KeyError

        return cursor


    @contextmanager
    def lock(self):
        if not self._locks:
            self._execute('BEGIN TRANSACTION')
        self._locks = True
        try:
            yield
        finally:
            self._locks = False
            if not self._locks:
                self._execute('COMMIT')
