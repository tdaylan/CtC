from generate_data_tobemovedtoexop import retr_datamock
from genNet import nets
from gdatFile import gdatstrt

#get data
light_curves, labels = retr_datamock()

gdat = gdatstrt()

#create simple fully connected and CNN
nn_generator = nets(gdat = gdat)
fully_connected = nn_generator.fcon()
cnn = nn_generator.Cnn1D()
