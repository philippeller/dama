import os
try:
   import cPickle as pickle
except:
   import pickle


def save(filename, obj):
    basename, extension = os.path.splitext(filename)
    if extension == "":
        extension = ".pkl"
    if not extension == ".pkl":
        raise ValueError("Only pickle (.pkl) format supprted, but %s was provided"%extension)
    filename = basename + extension

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load(filename):
    basename, extension = os.path.splitext(filename)
    if extension == "":
        extension = ".pkl"
    if not extension == ".pkl":
        raise ValueError("Only pickle (.pkl) format supprted, but %s was provided"%extension)
    filename = basename + extension

    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
