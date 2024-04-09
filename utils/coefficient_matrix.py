import pickle

params = pickle.load(open('model1_params.pkl', 'rb'))
with open('params.txt', 'w') as pf:
    for key, value in params.items():
        pf.write(key)
        pf.write(' = ')
        pf.write(str(value))
        pf.write('\n')