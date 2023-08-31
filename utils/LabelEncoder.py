## My label encoder (does one hot encoding)

# takes in a label and returns the encoded label
def KittiLabelEncoder(label):
    # switch statement for label encoding
    switcher = {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2,
        "Tram": 3,
        "Truck": 4,
        "Van": 5,
        "Person_sitting": 6,
        "Misc": 7,
        "DontCare": 8
    }

    # return the encoded label
    return switcher.get(label, "Invalid label")

