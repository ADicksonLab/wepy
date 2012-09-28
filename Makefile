all:
	# Build a binary package.
	dpkg-buildpackage -b -uc -us

	# We don't need the .changes file.
	rm ../*.changes

	# Move the package back into the current directory.
	mv ../*.deb .

check:
	python -m CoverageTestRunner
	rm -f .coverage

clean:
	rm -rf build debian/python-fstab
	rm -f *.deb *.pyc *.pyo .coverage debian/files
	rm -f debian/python-fstab.debhelper.log
	rm -f debian/python-fstab.postinst.debhelper
	rm -f debian/python-fstab.prerm.debhelper
	rm -f debian/python-fstab.substvars
