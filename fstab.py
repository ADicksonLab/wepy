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


import os
import re
import tempfile


class Line(object):

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

    # Lines split this way to shut up coverage.py.
    attrs = ("ws1", "device", "ws2", "directory", "ws3", "fstype")
    attrs += ("ws4", "options", "ws5", "dump", "ws6", "fsck", "ws7")

    def __init__(self, raw):
        self.dict = {}
        self.raw = raw

    def __getattr__(self, name):
        if name in self.dict:
            return self.dict[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        forbidden = ("dict", "dump", "fsck", "options")
        if name not in forbidden and name in self.dict:
            if self.dict[name] is None:
                raise Exception("Cannot set attribute %s when line dies not "
                                "contain filesystem specification" % name)
            self.dict[name] = value
        else:
            object.__setattr__(self, name, value)

    def get_dump(self):
        return int(self.dict["dump"])
    
    def set_dump(self, value):
        self.dict["dump"] = str(value)

    dump = property(get_dump, set_dump)
    
    def get_fsck(self):
        return int(self.dict["fsck"])

    def set_fsck(self, value):
        self.dict["fsck"] = str(value)
            
    fsck = property(get_fsck, set_fsck)

    def get_options(self):
        return self.dict["options"].split(",")
            
    def set_options(self, list):
        self.dict["options"] = ",".join(list)
        
    options = property(get_options, set_options)
            
    def set_raw(self, raw):
        match = False
        
        if raw.strip() != "" and not raw.strip().startswith("#"):
            pat = r"^(?P<ws1>\s*)"
            pat += r"(?P<device>\S*)"
            pat += r"(?P<ws2>\s+)"
            pat += r"(?P<directory>\S+)"
            pat += r"(?P<ws3>\s+)"
            pat += r"(?P<fstype>\S+)"
            pat += r"(?P<ws4>\s+)"
            pat += r"(?P<options>\S+)"
            pat += r"(?P<ws5>\s+)"
            pat += r"(?P<dump>\d+)"
            pat += r"(?P<ws6>\s+)"
            pat += r"(?P<fsck>\d+)"
            pat += r"(?P<ws7>\s*)$"

            match = re.match(pat, raw)
            if match:
                self.dict.update((attr, match.group(attr)) for attr in self.attrs)

        if not match:
            self.dict.update((attr, None) for attr in self.attrs)

        self.dict["raw"] = raw

    def get_raw(self):
        if self.has_filesystem():
            return "".join(self.dict[attr] for attr in self.attrs)
        else:
            return self.dict["raw"]
        
    raw = property(get_raw, set_raw)

    def has_filesystem(self):
        """Does this line have a filesystem specification?"""
        return self.device is not None


class Fstab(object):

    """An /etc/fstab file."""

    def __init__(self):
        self.lines = []
    
    def open_file(self, filespec, mode):
        if type(filespec) in (str, unicode):
            return file(filespec, mode)
        else:
            return filespec

    def close_file(self, f, filespec):
        if type(filespec) in (str, unicode):
            f.close()

    def get_perms(self, filename):
        return os.stat(filename).st_mode # pragma: no cover

    def chmod_file(self, filename, mode):
        os.chmod(filename, mode) # pragma: no cover

    def link_file(self, oldname, newname):
        if os.path.exists(newname):
            os.remove(newname)
        os.link(oldname, newname)

    def rename_file(self, oldname, newname):
        os.rename(oldname, newname) # pragma: no cover
    
    def read(self, filespec):
        """Read in a new file.
        
        If filespec is a string, it is used as a filename. Otherwise
        it is used as an open file.
        
        The existing content is replaced.
        
        """
        
        f = self.open_file(filespec, "r")
        lines = []
        for line in f:
            lines.append(Line(line))
        self.lines = lines
        self.close_file(filespec, f)

    def write(self, filespec):
        """Write out a new file.
        
        If filespec is a string, it is used as a filename. Otherwise
        it is used as an open file.
        
        """
        
        if type(filespec) in (str, unicode):
            # We create the temporary file in the directory (/etc) that the
            # file exists in. This is so that we can do an atomic rename
            # later, and that only works inside one filesystem. Some systems
            # have /tmp and /etc on different filesystems, for good reasons,
            # and we need to support that.
            dirname = os.path.dirname(filespec)
            prefix = os.path.basename(filespec) + "."
            fd, tempname = tempfile.mkstemp(dir=dirname, prefix=prefix)
            os.close(fd)
        else:
            tempname = filespec
    
        f = self.open_file(tempname, "w")
        for line in self.lines:
            f.write(line.raw)
        self.close_file(filespec, f)

        if type(filespec) in (str, unicode):
            self.chmod_file(tempname, self.get_perms(filespec))
            self.link_file(filespec, filespec + ".bak")
            self.rename_file(tempname, filespec)
