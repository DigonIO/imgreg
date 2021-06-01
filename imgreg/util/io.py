"""
Directory processing utilities

Author: Fabian A. Preiss
"""
import fnmatch
import os
from typing import List, Set, Union


def fnmatch_filter(
    ls_files: Set[str], pattern: Union[str, List[str]] = "*"
) -> Set[str]:
    """Return the subset of strings that match given patterns."""
    if isinstance(pattern, str):
        patterns = [pattern]
    return {
        fname
        for fname in ls_files
        if True in [fnmatch.fnmatch(fname, p) for p in patterns]
    }


class DirectoryViewError(Exception):
    pass


class DirectoryView:
    def __init__(self, inputdir=".", file_pattern="*"):
        inputdir = inputdir[:-1] if inputdir[-1] == "/" else inputdir
        inputdir = os.path.expanduser(inputdir)
        if not os.path.isdir(inputdir):
            raise DirectoryViewError(f"{inputdir} is not a directory")
        ls_current_dir = os.listdir(inputdir)
        self.realpath = os.path.realpath(inputdir)
        self.basename = os.path.basename(self.realpath)
        self.files = fnmatch_filter(
            {
                file_name
                for file_name in ls_current_dir
                if os.path.isfile(inputdir + "/" + file_name)
            },
            file_pattern,
        )
        self.dirs = {
            DirectoryView(self.realpath + "/" + d, file_pattern=file_pattern)
            for d in ls_current_dir
            if os.path.isdir(self.realpath + "/" + d)
        }

    def file_path_generator(self, step: int = 1):
        for i, file in enumerate(sorted(self.files)):
            if not i % step:
                yield f"{self.realpath}/{file}"
