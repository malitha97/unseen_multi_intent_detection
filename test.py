import pickle

with open('Models/ohe.pickle', 'rb') as handle:
    ohe_new = pickle.load(handle)

print(ohe_new.classes_)

#Unseen Classes: atis_flight, atis_flight_no, atis_quality, atis_restriction