## My label encoder (does one hot encoding)


class LabelEncoder():

    # init function with our switcher with label 
    def __init__(self) -> None:
        # switch statement for label encoding
        switcher = {
            "Car": 0.000,
            "Pedestrian": 0.111,
            "Cyclist": 0.222,
            "Tram": 0.333,
            "Truck": 0.444,
            "Van": 0.555,
            "Person_sitting": 0.666,
            "Misc": 0.777,
            "DontCare": 0.888
        }
        self.switcher = switcher

    # takes in a label and returns the encoded label
    def KittiLabelEncoder(self, label):
        # return the encoded label
        return self.switcher.get(label, "Invalid label")
    
    # We use self in python functions because in python methods are passed automatically
    # but not received automatically, so we need self to receive an instance of the method
    # (hence, if you remove self, you get "1 positional arguments expected but got 0")
    def len_classes(self):
        return len(self.switcher)

