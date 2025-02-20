import os
import ctypes as ct

__all__ = ["libcppbackend"]


loader = ct.LibraryLoader(ct.CDLL)
abspath = os.path.abspath(os.path.join(__file__, "..", "src", "liblinear.so"))
libcppbackend = loader.LoadLibrary(abspath)
