
# IDEA: make a __version_info__ attribute that follows "growth versioning semantics"
# SNIPPET: for that idea
# breakage = 0
# regression = 0
# growth = 0
# prerelease_type = "a"
# prerelease_number = 0
# dev = 0
# postrelease = 0

# __version_info__ = (breakage,
#                     regression,
#                     growth,
#                     prerelease_type,
#                     prerelease_number,
#                     dev,
#                     postrelease,)

# __version__ = f"{breakage}.{regression}.{growth}{prerelease_type}{prerelease_number}.dev{dev}.post{postrelease}"

__version__ = '2020-03-09a0.dev0'
