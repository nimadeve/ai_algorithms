from modules import kmeans,fcm,anfis,dectree,svm,neuralnet
print('-'*50)
print('1) K-Means ')
print('2) FCM')
print('3) Anfis')
print('4) decision tree')
print('5) SVM')
print('6) Neural Network ')
print('-'*50)
input = input('Please Enter Option Number: \n')
print('*'*50)

def router(digit):
	if digit == 1:
		kmeans.kmean()
	if digit == 2:
		fcm.fcm()
	if digit == 3:
		anfis.anfis()
	if digit == 4:
		dectree.dectree()
	if digit == 5:
		svm.svm()
	if digit == 6:
		neuralnet.neuralnet()

def num_control(digit):
	if int(digit) <= 6 and int(digit)>=1: 
		router(int(digit))
	else:
		print('Number is not Valid !')

if input:
	num_control(input)

