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

# Software copied and modified directly from this source
import os
import os.path as osp
import sqlite3
from collections import MutableMapping
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
NON_SQLITE_MODES = ('x', 'w-', 'w')
TRUNCATE_MODES = ('w',)
FAIL_IF_EXISTS_CREATE_MODES = ('w-', 'x')

SQLITE3_URI_TEMPLATE = "{protocol}:{url}"
SQLITE3_QUERY_URI_TEMPLATE = "{protocol}:{url}?{query}"

SQLITE3_QUERY_JOIN_CHAR = "&"

SQLITE3_INMEMORY_URI = ":memory:"


# the default types for the values that is checked, is bytes
DEFAULT_VALUE_TYPES = (bytes, bytearray)

def handle_mode(db_url, mode_spec):

    db_uri = False

    # if the db url is the in memory special string just use that
    if db_url == SQLITE3_INMEMORY_URI:

        db_uri = SQLITE3_URI_TEMPLATE.format(protocol='file',
                                             url=db_url)

    # if it is a mode that sqlite3 doesn't support explicitly so we
    # have to do manual processing for
    elif mode_spec in NON_SQLITE_MODES:

        # we just use th url as the uri since we don't need to specify
        # anything else becase create mode is the default
        db_uri = SQLITE3_URI_TEMPLATE.format(protocol='file',
                                             url=db_url)

        # if the mode is either 'x' or 'w-' check to see if the db
        # already exists. If it does raise an error.
        if mode_spec in FAIL_IF_EXISTS_CREATE_MODES:

            if osp.exists(db_url):
                raise FileExistsError

        # if it is 'w' we want to delete the old file
        elif mode_spec in TRUNCATE_MODES:

            # if it exists remove it
            if osp.exists(db_url):
                os.remove(db_url)

        else:
            raise ValueError("Unrecognized non-sqlite mode '{}'".format(mode_spec))


    # if the mode is one of the modes in the mode mapping use that to
    # generate the URI
    elif mode_spec in dict(MODE_MAPPING):

        sqlite_mode = dict(MODE_MAPPING)[mode_spec]

        # if it is a mode with no SQLite comparison dont do anything
        # special
        if sqlite_mode is not None:
            mode_q = "mode={}".format(sqlite_mode)
            query_str = SQLITE3_QUERY_JOIN_CHAR.join([mode_q])
            db_uri = SQLITE3_QUERY_URI_TEMPLATE.format(protocol='file',
                                                       url=db_url,
                                                       query=query_str)

    # should have generated a URI by now, and if we didn't raise an error
    if not db_uri:
        raise ValueError("The given url and mode_spec did not produce a valid URI")

    return db_uri

class KV(MutableMapping):

    def __init__(self, db_url=':memory:',
                 table='data',
                 primary_key='key',
                 value_name='value',
                 timeout=5,
                 mode='x',
                 value_types=DEFAULT_VALUE_TYPES,
    ):

        # generate a good URI from the url and the mode
        db_uri = handle_mode(db_url, mode)


        # set the value types for this kv
        self._kv_types = value_types


        self._db_uri = db_uri

        # connect to the db
        self._db = sqlite3.connect(self._db_uri, timeout=timeout, uri=True)
        self._db.isolation_level = None

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

        with self.lock():

            # insert the key-value pair if the key isn't in the db
            try:
                self._execute('INSERT INTO {table} VALUES (?, ?)'.format(table=self.table),
                              (key, value))

            # otherwise update the keys value
            except sqlite3.IntegrityError:
                self._execute('UPDATE {table} SET {value}=? WHERE {key}=?'.format(
                                      key=self.primary_key,
                                      value=self.value_name,
                                      table=self.table),
                              (value, key))

    def __delitem__(self, key):
        if key in self:
            self._execute('DELETE FROM {table} WHERE {key}=?'.format(
                                 key=self.primary_key,
                                 table=self.table),
                          (key,))
        else:
            raise KeyError

    @contextmanager
    def lock(self):
        if not self._locks:
            self._execute('BEGIN IMMEDIATE TRANSACTION')
        self._locks += 1
        try:
            yield
        finally:
            self._locks -= 1
            if not self._locks:
                self._execute('COMMIT')
