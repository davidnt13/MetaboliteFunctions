from MetFunctions import *
from GenerateDescriptors import calcCoati

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_data = "test_data.csv"
training_data = "training_data.csv"

print(calcCoati(test_data))
