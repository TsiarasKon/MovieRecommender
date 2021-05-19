import pickle


""" save and load python dictionary """
def save_dict(dict, name):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pickle', 'rb') as f:
        return pickle.load(f)
