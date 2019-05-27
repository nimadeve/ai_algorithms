class anfis:
	def __init__(self):
		import random
		import math
		from mpl_toolkits.mplot3d import Axes3D
		import matplotlib.pyplot as plt
		import numpy as np
		from matplotlib import cm

		class ANFIS:
			def __init__(self, rule_number):
				self.rule_number = rule_number
				random.seed(69)

				self.A_a = [random.random()*0.4 - 0.2 for i in range(rule_number)]
				self.A_b = [random.random()*0.4 - 0.2 for i in range(rule_number)]
				self.B_a = [random.random()*0.4 - 0.2 for i in range(rule_number)]
				self.B_b = [random.random()*0.4 - 0.2 for i in range(rule_number)]

				self.p = [random.random()*0.4 - 0.2 for i in range(rule_number)]
				self.q = [random.random()*0.4 - 0.2 for i in range(rule_number)]
				self.r = [random.random()*0.4 - 0.2 for i in range(rule_number)]

			def sigmoid(x,a,b):
				return 1./(1+math.exp(b*(x-a)))
			def tNorm(x,y):
				return x*y

			def output(self,x,y,train=False):
				mi_a_x=[]
				mi_b_y=[]
				antecedent=[]
				z=[]
				for i in range(self.rule_number):
					mi_a_x.append(ANFIS.sigmoid(x,self.A_a[i],self.A_b[i]))
					mi_b_y.append(ANFIS.sigmoid(y,self.B_a[i],self.B_b[i]))
					antecedent.append(ANFIS.tNorm(mi_a_x[i],mi_b_y[i]))
					z.append(self.p[i]*x+self.q[i]*y+self.r[i])

				antecedent_sum=sum(antecedent)
				o=0
				for i in range(self.rule_number):
					o+=antecedent[i]*z[i]
				o/=antecedent_sum

				if train:
					return (o,antecedent,mi_a_x,mi_b_y,z)
				return o

			def train(self, dataset, eta,batch_size,epochs=10000,print_error=1000):
				dA_a=[0 for i in range(self.rule_number)]
				dA_b=[0 for i in range(self.rule_number)]
				dB_a=[0 for i in range(self.rule_number)]
				dB_b=[0 for i in range(self.rule_number)]
				dp=[0 for i in range(self.rule_number)]
				dq=[0 for i in range(self.rule_number)]
				dr=[0 for i in range(self.rule_number)]

				self.errors=[]

				for epoch in range(epochs):
					for k in range(len(dataset)):
						x,y=dataset[k]
						o,weights,mi_a_x,mi_b_y,z=self.output(x[0],x[1],train=True)

						weights_sum=sum(weights)
						for i in range(self.rule_number):
							dA_a[i]+=-(y-o)*sum([weights[j]*(z[i]-z[j]) if j!=i else 0.0 for j in range(self.rule_number)])/(weights_sum**2)*mi_a_x[i]*(1-mi_a_x[i])*mi_b_y[i]*self.A_b[i]
							dA_b[i]+=(y-o)*sum([weights[j]*(z[i]-z[j]) if j!=i else 0.0 for j in range(self.rule_number)])/(weights_sum**2)*mi_a_x[i]*(1-mi_a_x[i])*mi_b_y[i]*(x[0]-self.A_a[i])
							dB_a[i] +=- (y - o) * sum([weights[j] * (z[i] - z[j]) if j!=i else 0.0 for j in range(self.rule_number)]) / (weights_sum**2) * mi_b_y[i] * (1 - mi_b_y[i])*mi_a_x[i] * self.B_b[i]
							dB_b[i] +=(y - o) * sum([weights[j] * (z[i] - z[j]) if j!=i else 0.0 for j in range(self.rule_number)]) / (weights_sum**2) * mi_b_y[i] * (1 - mi_b_y[i])*mi_a_x[i] * (x[1] - self.B_a[i])

							dp[i]+=-(y-o)*weights[i]/weights_sum*x[0]
							dq[i]+=-(y-o)*weights[i]/weights_sum*x[1]
							dr[i]+=-(y-o)*weights[i]/weights_sum

						if (k+1)%batch_size==0 or k==len(dataset)-1:
							for i in range(self.rule_number):
								self.A_a[i]+=-eta[0]*dA_a[i]
								self.A_b[i]+=-eta[0]*dA_b[i]
								self.B_a[i]+=-eta[0]*dB_a[i]
								self.B_b[i]+=-eta[0]*dB_b[i]

								self.p[i]+=-eta[1]*dp[i]
								self.q[i]+=-eta[1]*dq[i]
								self.r[i]+=-eta[1]*dr[i]

								#print(self.A_a, dA_a)

								dA_a[i]=0
								dA_b[i]=0
								dB_a[i]=0
								dB_b[i]=0
								dp[i]=0
								dq[i]=0
								dr[i]=0

					currentError=self.error(dataset)
					if (epoch+1)%print_error==0:
						print("Epoch: %d Error: %f"%(epoch+1,currentError))
					self.errors.append(currentError)

			def drawRules(self,x_range):
				for i in range(self.rule_number):
					y=[ANFIS.sigmoid(x,self.A_a[i],self.A_b[i]) for x in x_range]
					plt.ylim(0,1)
					plt.plot(x_range,y,'r')
					plt.title("Neizraziti skup A%d"%i)
					plt.show()
					y = [ANFIS.sigmoid(x, self.B_a[i], self.B_b[i]) for x in x_range]
					plt.ylim(0, 1)
					plt.plot(x_range, y, 'r')
					plt.title("Neizraziti skup B%d" % i)
					plt.show()
			def error(self,dataset):
				current_error = 0
				for x, y in dataset:
					o = self.output(x[0], x[1])
					current_error += (y - o) ** 2
				current_error /= 2*len(dataset)

				return current_error
			def drawFunction(self,f,dataset):
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')

				X=np.array([x[0] for x,y in dataset])
				Y=np.array([x[1] for x,y in dataset])
				Z = np.array([y for x, y in dataset])
				ax.plot_trisurf(X,Y,Z,cmap=cm.summer,alpha=0.6)
				Z = np.array([self.output(x[0],x[1]) for x, y in dataset])
				ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm)

				plt.title("Funkcija f i naucena funkcija")

				plt.show()

			def drawErrorSurface(self,f,dataset):
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')

				X = np.array([x[0] for x, y in dataset])
				Y = np.array([x[1] for x, y in dataset])
				Z = np.array([y-self.output(x[0], x[1]) for x, y in dataset])
				ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm)

				plt.title("Pogreska uzorka")

				plt.show()

			def drawErrorEpochs(self):
				plt.plot(range(len(self.errors)),self.errors,'r')

				plt.title("Srednja kvadratna pogreska kroz epohe")
				
				plt.show()

		anfis=ANFIS(3)
		f=lambda x,y: ((x-1)**2+(y+2)**2-5*x*y+3)*math.cos(x/5.)**2
		dataset=[((x,y),f(x,y)) for x in range(-4,5) for y in range(-4,5)]

		anfis.train(dataset, (0.0001,0.001), 1,10000)
		anfis.drawErrorEpochs()
		anfis.drawErrorSurface(f,dataset)
		anfis.drawFunction(f,dataset)
		anfis.drawRules(range(-4,5))
