# LSTM for international airline passengers problem with time step regression framing
import numpy
import matplotlib.pyplot as plt
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

from time import time

def getMatrix(textFile):
    inputMatrix = []
    with open(textFile) as f:
        for line in f:
            inputMatrix.append(line.split())

    return numpy.array([[float(coord) for coord in frame] for frame in inputMatrix])



# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):

	mainX = []
	mainY = []

	#print dataset.shape
	for j in range(0,dataset.shape[1]):
		dataX, dataY = [], []
		for i in range(len(dataset)-look_back-1):
			a = dataset[i:(i+look_back), j]
			#a = numpy.asarray(a).transpose().tolist()

			dataX.append(a)

			dataY.append(dataset[i + look_back,j])
			'''
			print 'dataX: ', dataX
			print '----'
			print 'dataY: ', dataY
			'''
			

		if j == 0:
			'''
			mainX = numpy.array(numpy.array([dataX]).transpose())
			mainY = numpy.array(numpy.array([dataY]))
			'''

			mainX = numpy.array([dataX])
			mainY = numpy.array([dataY])

		else:
			dataX = numpy.array([dataX])
			dataY = numpy.array([dataY])

	
			#print 'mainX: ', mainX.shape
			#print 'dataX: ',dataX.shape
			mainX = numpy.concatenate((mainX, dataX))	
			mainY = numpy.concatenate((mainY, dataY))	
		#print 'mainY.shape: ', mainY.shape
		#print 'mainX: ', mainX.shape
		#print 'mainY: ', mainY.shape

	#print mainX.shape


	return mainX, mainY

# convert an array of values into a dataset matrix
def original_create_dataset(dataset, look_back=1):
	dataX, dataY = [], []


	print dataset.shape

	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])


	return numpy.array(dataX), numpy.array(dataY)



def trainModel(dataset):
	##### VALUES TO EDIT #####	
	look_back = 20
	numInputNodes = 9
	#numInputNodes = 1
	#########################

	# fix random seed for reproducibility
	numpy.random.seed(7)


	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)

	# split into train and test sets
	train_size = int(len(dataset) * 0.67)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]



	# reshape into X=t and Y=t+1
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	# reshape input to be [samples, time steps, features]
	
	# NEW
	# reshape input to be [nodes, samples, time steps, features]


	#trainX = trainX.transpose()
	#trainY = trainY.transpose()
	trainX = numpy.reshape(trainX, (trainX.shape[1], trainX.shape[2] ,trainX.shape[0]))
	trainY = trainY.transpose()
	
	print trainX.shape
	print trainY.shape
	'''
	trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
	'''
	'''
	print trainX.shape
	print trainY.shape
	'''

	# create and fit the LSTM network


	model = Sequential()
	model.add(LSTM(4, input_dim=numInputNodes))
	model.add(Dense(9))
	model.compile(loss= 'mean_squared_error' , optimizer= 'adam' )
	model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)

	# Estimate model performance
	trainScore = model.evaluate(trainX, trainY, verbose=0)
	print( 'Train Score:' , scaler.inverse_transform(numpy.array([[trainScore]])))
	testScore = model.evaluate(testX, testY, verbose=0)
	print( 'Test Score:' , scaler.inverse_transform(numpy.array([[testScore]])))

	# generate predictions for training

	trainPredict = model.predict(trainX)

	start = time()

	testPredict = model.predict(testX)

	timePassed = time() - start
	print timePassed, ' s'
	
	'''
	print trainX.shape
	print 'trainPredict'
	print trainPredict.shape
	'''


	'''
	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(dataset)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
	'''

	testPredictPlot = testPredict
	trainPredictPlot = trainPredict

	# plot baseline and predictions
	plt.plot(dataset)
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()

	return model







# load the dataset
#dataframe = pandas.read_csv('sample-scripts/data/international-airline-passengers.csv' , usecols=[1], engine= 'python' , skipfooter=3)
#dataset = dataframe.values
#dataset = dataset.astype( 'float32')


dataset = getMatrix("handCoordinates.txt")
print 'dataset: ', dataset.shape
trainModel(dataset)
