# fstab.py - read, manipulate, and write /etc/fstab files
# Copyright (C) 2008  Canonical, Ltd.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import re
import dataclasses as dc
import pkgutil
from warnings import warn

from jinja2 import Template

# optional imports
try:
    import pandas as pd
except ImportError:
    warn("Pandas installation not found. Install with 'pretty' extra.")

CONCRETE_COLUMNS = (
        "ws1",
        "device",
        "ws2",
        "directory",
        "ws3",
        "fstype",
        "ws4",
        "options",
        "ws5",
        "dump",
        "ws6",
        "fsck",
        "ws7",
    )

ABSTRACT_COLUMNS = (
        "device",
        "directory",
        "fstype",
        "options",
        "dump",
        "fsck",
    )

FSTAB_REGEX = r"" \
              r"^(?P<ws1>\s*)" \
              r"(?P<device>\S*)" \
              r"(?P<ws2>\s+)" \
              r"(?P<directory>\S+)" \
              r"(?P<ws3>\s+)" \
              r"(?P<fstype>\S+)" \
              r"(?P<ws4>\s+)" \
              r"(?P<options>\S+)" \
              r"(?P<ws5>\s+)" \
              r"(?P<dump>\d+)" \
              r"(?P<ws6>\s+)" \
              r"(?P<fsck>\d+)" \
              r"(?P<ws7>\s*)$"

COMMENT_CHARACTER = "#"

def parse_line(line_str):
    """Parses a non-comment line of an fstab file using a regex.

    Does no validation.

    Parameters
    ----------

    line_str : str

    Returns
    -------

    entry : FstabEntry obj

    """


    match = re.match(FSTAB_REGEX, line_str)

    cols = {}
    for col in ABSTRACT_COLUMNS:
        cols[col] = match.group(col)

    return FstabEntry(**cols)


@dc.dataclass(frozen=True)
class FstabEntry():
    """A line in an /etc/fstab line.

    Lines may or may not have a filesystem specification in them. The
    has_filesystem method tells the user whether they do or not; if they
    do, the attributes device, directory, fstype, options, dump, and
    fsck contain the values of the corresponding fields, as instances of
    the sub-classes of the LinePart class. For non-filesystem lines,
    the attributes have the None value.

    Lines may or may not be syntactically correct. If they are not,
    they are treated as as non-filesystem lines.

    """

    device: str
    directory: str
    fstype: str
    options: str
    dump: str
    fsck: str

HEADER = "# /etc/fstab: static file system information."
COLUMN_FIELDS = (
    'file_system',
    'mount_point',
    'type',
    'options',
    'dump',
    'pass',
)

COLUMN_HEADER = "# <file system> <mount point>   <type>  <options>       <dump>  <pass>"

def default_header():
    return '\n'.join([HEADER, COLUMN_HEADER])


def get_fstab_template():

    path = "fstab_template/fstab.j2"

    return pkgutil.get_data(__name__,
                         path)\
                  .decode()


def parse_fstab(file_str):

    entries = []
    header_lines = []
    for line in file_str.splitlines():

        in_header = True

        if line.startswith("#"):
            if in_header:
                header_lines.append(line)
            else:
                pass
        elif line.isspace():
            if in_header:
                header_lines.append(line)
            else:
                pass
        else:
            entries.append(parse_line(line))
            in_header = False

    header = '\n'.join(header_lines)
    return Fstab(
        entries=entries,
        header=header,
    )


@dc.dataclass(frozen=True)
class Fstab():
    """An /etc/fstab file.

    Parameters
    ----------

    entries : list of FstabEntry obj, optional
        Entry objects to put in the Fstab

    header : str, optional
        An optional user defined header for the file. Uses default if Ellipsis

    """

    header: str = dc.field(default_factory=default_header)
    entries: tuple = ()

    def to_df(self):

        df = pd.DataFrame(
            [dc.asdict(entry) for entry in self.entries],
        )
        return df

    def render(self):
        """
        Returns
        -------

        ftab_str : str
            The formmated fstab table


        """

        template = Template(get_fstab_template())
        result = template.render(**dc.asdict(self))

        return result


def from_dataframe(fstab_df,
                   header=None):

    entries = []
    for row_id, row in fstab_df.iterrows():
        rec = {}
        for colname in ABSTRACT_COLUMNS:
            rec[colname] = row[colname]

        entries.append(FstabEntry(**rec))

    fstab = Fstab(
        entries=entries,
        header=header
    )

    return fstab

def from_struct(fstab_struct):
    """Read it in from python dicts and lists.

    This could be from a JSON file or something similar.

    Parameters
    ----------

    fstab_struct : list of dict of str: val

    Returns
    -------

    fstab : Fstab obj

    """

    entries = []
    for entry in fstab_struct:
        entries.append(FstabEntry(**entry))

    return Fstab(entries=entries)
