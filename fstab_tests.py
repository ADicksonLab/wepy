# fstab_tests.py - unit tests for fstab.py
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
import StringIO
import tempfile
import unittest

import fstab


class LineTests(unittest.TestCase):

    def setUp(self):
        self.line = fstab.Line("/dev / type opt,ions 0 1")

    def testSetsRawToWhateverTheConstructorIsGiven(self):
        # Note that we set the line to something syntactically incorrect.
        # This is on purpose. The class is supposed to handle that.
        line = fstab.Line("yo")
        self.assertEqual(line.raw, "yo")

    def testDoesNotThinkEmptyLineHasFilesystem(self):
        self.assertFalse(fstab.Line("").has_filesystem())

    def testDoesNotThinkLineWithWhiteSpaceHasFilesystem(self):
        self.assertFalse(fstab.Line("   \t").has_filesystem())

    def testDoesNotThinkCommentHasFilesystem(self):
        self.assertFalse(fstab.Line("# foo").has_filesystem())

    def testDoesNotSyntacticallyIncorrectLineHasFilesystem(self):
        self.assertFalse(fstab.Line("yo").has_filesystem())

    def testDoesThinkACorrectLineHasFilesystem(self):
        self.assert_(self.line.has_filesystem())

    def testParsesDeviceCorrectly(self):
        self.assertEqual(self.line.device, "/dev")

    def testParsesDirectoryCorrectly(self):
        self.assertEqual(self.line.directory, "/")

    def testParsesFstypeCorrectly(self):
        self.assertEqual(self.line.fstype, "type")

    def testParsesOptionsCorrectly(self):
        self.assertEqual(self.line.options, ["opt", "ions"])

    def testParsesDumpCorrectly(self):
        self.assertEqual(self.line.dump, 0)

    def testParsesFsckCorrectly(self):
        self.assertEqual(self.line.fsck, 1)

    def testRaisesAttributeErrorForBadAttribute(self):
        self.assertRaises(AttributeError, getattr, self.line, "bad")

    def testSetsRawWhenSettingDevice(self):
        self.line.device = "/foo"
        self.assertEqual(self.line.raw, "/foo / type opt,ions 0 1")

    def testSetsRawWhenSettingDirectory(self):
        self.line.directory = "/foo"
        self.assertEqual(self.line.raw, "/dev /foo type opt,ions 0 1")

    def testSetsRawWhenSettingFstype(self):
        self.line.fstype = "foo"
        self.assertEqual(self.line.raw, "/dev / foo opt,ions 0 1")

    def testSetsRawWhenSettingOptions(self):
        self.line.options = ["foo"]
        self.assertEqual(self.line.raw, "/dev / type foo 0 1")

    def testSetsRawWhenAppendingOptions(self):
        self.line.options += ["foo"]
        self.assertEqual(self.line.raw, "/dev / type opt,ions,foo 0 1")

    def testSetsRawWhenSettingDump(self):
        self.line.dump = 2
        self.assertEqual(self.line.raw, "/dev / type opt,ions 2 1")

    def testSetsRawWhenSettingFsck(self):
        self.line.fsck = 2
        self.assertEqual(self.line.raw, "/dev / type opt,ions 0 2")

    def testDoesNotAllowSettingDeviceWhenUnset(self):
        line = fstab.Line("yo")
        self.assertRaises(Exception, setattr, line, "device", "/dev")



class MockFile(object):

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FstabTests(unittest.TestCase):

    content = """\
# this is a comment, followed by an empty line

# the next line is a syntax error
yo!

#UUID="2b0ab63e-fd1b-4d7a-b021-3827b4aa4c8b" /media/hda3 ext3 defaults 0 2
/dev/hda1 / ext3 defaults,errors=remount-ro 0 1
/dev/hda2 /boot ext3 defaults 0 2
"""

    def setUp(self):
        self.fstab = fstab.Fstab()
        self.fstab.read(StringIO.StringIO(self.content))

    def testReadsTheRightNumberOfLines(self):
        self.assertEqual(len(self.fstab.lines), len(self.content.splitlines()))

    def testParsesLinesButLastTwoAsHavingNoFilesystems(self):
        for line in self.fstab.lines[:-2]:
            self.assertEqual(line.has_filesystem(), False,
                             msg="Line has filesystem: %s" % 
                                    line.raw)

    def testParsesLastTwoLinesAsHavingFilesystem(self):
        for line in self.fstab.lines[-2:]:
            self.assertEqual(line.has_filesystem(), True,
                             msg="Line does not have filesystem: %s" % 
                                    line.raw)

    def testWritesTheSameOutput(self):
        f = StringIO.StringIO()
        self.fstab.write(f)
        self.assertEqual(f.getvalue(), self.content)

    def testOpensFileWhenGivenName(self):
        f = self.fstab.open_file("README", "r")
        self.assertEqual(type(f), file)

    def testOpensOpenFileAsItself(self):
        f = StringIO.StringIO()
        self.assertEqual(self.fstab.open_file(f, "r"), f)

    def testClosesOpenFile(self):
        f = MockFile()
        self.fstab.close_file(f, "README")
        self.assert_(f.closed)

    def catch_open_file(self, filename, mode):
        self.assert_(mode, "w")
        self.written_file = filename
        return StringIO.StringIO()

    def catch_link_file(self, oldname, newname):
        self.assertFalse(hasattr(self, "linked_file"))
        self.linked_file = (oldname, newname)

    def catch_rename_file(self, oldname, newname):
        self.assertFalse(hasattr(self, "renamed_file"))
        self.renamed_file = (oldname, newname)

    def catch_get_perms(self, filename):
        self.assertFalse(hasattr(self, "permed_file"))
        self.permed_file = filename
        return 0123

    def catch_chmod_file(self, filename, mode):
        self.assertFalse(hasattr(self, "chmodded_file"))
        self.chmodded_file = (filename, mode)

    def testWritesFirstToTemporaryFileThenRenames(self):
        self.fstab.open_file = self.catch_open_file
        self.fstab.link_file = self.catch_link_file
        self.fstab.rename_file = self.catch_rename_file
        self.fstab.get_perms = self.catch_get_perms
        self.fstab.chmod_file = self.catch_chmod_file
        self.fstab.write("foo")
        self.assert_(self.written_file.startswith(os.path.abspath("./foo.")))
        self.assertEqual(self.permed_file, "foo")
        self.assertEqual(self.chmodded_file[0], self.written_file)
        self.assertEqual(self.chmodded_file[1], 0123)
        self.assertEqual(self.linked_file[0], "foo")
        self.assertEqual(self.linked_file[1], "foo.bak")
        self.assertEqual(self.renamed_file[0], self.written_file)
        self.assertEqual(self.renamed_file[1], "foo")
        # We delete the written_file manually here, since our mock-rename
        # doesn't actually do anything.
        os.remove(self.written_file)

    def testLinkFileWorksWhenTargetDoesNotExist(self):
        fd, name1 = tempfile.mkstemp()
        os.close(fd)
        
        fd, name2 = tempfile.mkstemp()
        os.close(fd)
        os.remove(name2)
        
        self.assertEqual(self.fstab.link_file(name1, name2), None)
        os.remove(name1)

    def testLinkFileWorksWhenTargetExists(self):
        fd, name1 = tempfile.mkstemp()
        os.close(fd)
        
        fd, name2 = tempfile.mkstemp()
        os.close(fd)
        
        self.assertEqual(self.fstab.link_file(name1, name2), None)
        os.remove(name1)
        os.remove(name2)


class FstabUserTests(unittest.TestCase):

    def testAddRelatimeMountOption(self):
        fs = fstab.Fstab()
        fs.read(StringIO.StringIO("/dev / ext3 defaults 0 1\n"))
        for line in fs.lines:
            if line.has_filesystem() and "relatime" not in line.options:
                line.options += ["relatime"]
        f = StringIO.StringIO()
        fs.write(f)
        self.assertEqual(f.getvalue(), "/dev / ext3 defaults,relatime 0 1\n")
