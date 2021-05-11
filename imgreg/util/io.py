"""
Directory processing utilities

Author: Fabian A. Preiss
"""
import os
import fnmatch
from typing import Union, List, Set


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


# TODO: make it possible to give a filetree
# dict and ignore folders not in filetree
def dirtree_reader(inputdir=".", file_pattern="*"):
    inputdir = inputdir[:-1] if inputdir[-1] == "/" else inputdir
    if not os.path.isdir(inputdir):
        print(f"{inputdir} is not a directory")
        return
    ls_current_dir = os.listdir(inputdir)
    ls_files = {x for x in ls_current_dir if os.path.isfile(inputdir + "/" + x)}
    ls_files = fnmatch_filter(ls_files, file_pattern)
    ls_dirs = [x for x in ls_current_dir if os.path.isdir(inputdir + "/" + x)]
    dict_dirs = [
        dirtree_reader(inputdir + "/" + x, file_pattern=file_pattern) for x in ls_dirs
    ]
    dict_dirs = {} if dict_dirs == [] else {"dirs": dict_dirs}
    dict_files = {} if ls_files == set() else {"files": ls_files}
    dict_dirs.update(dict_files)
    return {os.path.basename(inputdir): dict_dirs}


class File_Set_Ops:
    @staticmethod
    def a_or_b(a, b):
        return {"files": a["files"].union(b["files"])}

    @staticmethod
    def a_and_b(a, b):
        return {"files": a["files"].intersection(b["files"])}

    @staticmethod
    def a_diff_b(a, b):
        return {"files": a["files"].difference(b["files"])}

    @staticmethod
    def a_xor_b(a, b):
        return {"files": a["files"].symmetric_difference(b["files"])}

    @staticmethod
    def b_diff_a(a, b):
        return {"files": b["files"].difference(a["files"])}

    @staticmethod
    def a_b_op(a, b, op="or"):
        methods = {
            "or": File_Set_Ops.a_or_b,
            "and": File_Set_Ops.a_and_b,
            "\\": File_Set_Ops.a_diff_b,
            "xor": File_Set_Ops.a_xor_b,
            "/": File_Set_Ops.b_diff_a,
        }
        return methods[op](a, b)

    @staticmethod
    def a_copy(a):
        return {"files": a["files"].copy()}

    @staticmethod
    def a_set(a):
        return set()

    @staticmethod
    def a_op(a, op="or"):
        methods = {
            "or": File_Set_Ops.a_copy,
            "and": File_Set_Ops.a_set,
            "\\": File_Set_Ops.a_copy,
            "xor": File_Set_Ops.a_copy,
            "/": File_Set_Ops.a_set,
        }
        return methods[op](a)

    @staticmethod
    def file_set_op(a, b=set(), op="or"):
        if "files" in a and "files" in b:
            return File_Set_Ops.a_b_op(a, b, op)
        elif "files" in a and "files" not in b:
            return File_Set_Ops.a_op(a, op)
        elif "files" in b and "files" not in a:
            return File_Set_Ops.a_op(b, op)
        else:
            return {}
