import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
X=[i for i in np.arange(7,15.5,0.5)]
print(X)
a=[1.025,1.134,1.232,1.323,1.414,1.498,1.564,1.616,1.708,2.032,2.157,2.265,2.373,2.521,2.664,2.828,3.006]
plt.plot(X,a)
plt.show()