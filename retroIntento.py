import numpy as np
import matplotlib.pyplot as plt
import math
 
def tanh(var):
    return np.tanh(var)
 
def tanh_derivada(var):
    return 1.0 - var**2
 
 
class RedNeuronal:
 
    def __init__(self, capas, activacion='tanh'):
        if activacion == 'tanh':
            self.activacion = tanh
            self.activacion_prime = tanh_derivada
 
        #Se inicializan los pesos
        self.pesos = []
        self.deltas = []
        #Se asignan los valores aleatorios a las capas
        for i in range(1, len(capas) - 1):
            rand = 2*np.random.random((capas[i-1] + 1, capas[i] + 1)) -1
            self.pesos.append(rand)
            
        rand = 2*np.random.random( (capas[i] + 1, capas[i+1])) - 1
        self.pesos.append(rand)
 
    def predict(self, x): 
        act = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.pesos)):
            act = self.activacion(np.dot(act, self.pesos[l]))
    
        return act
 
    def mostrar_pesos(self):
        print("Pesos")
        for i in range(len(self.pesos)):
            print(self.pesos[i])
 
    def obtener_deltas(self):
        return self.deltas

    def entrenar(self, x, y, jP, kP,tasa_aprendizaje=0.2, epocas=100000):
        #Se agrega la columna de bias a las entradas X
        ones = np.atleast_2d(np.ones(x.shape[0]))
        onesjP = np.atleast_2d(np.ones(jP.shape[0]))
        totalErrores = 0
        x = np.concatenate((ones.T, x), axis=1)
        jP = np.concatenate((onesjP.T, jP), axis=1)
        errorTotal = 1
        while (errorTotal > 0.0001):
            for w in range(epocas):
                #Se obtiene de forma aleatoria uno de los registros de entrada
                i = np.random.randint(x.shape[0])
                a = [x[i]]
                for z in range(len(self.pesos)):
                    valor = np.dot(a[z], self.pesos[z])
                    activacion = self.activacion(valor)
                    a.append(activacion)
                # Calculo de la diferencia en la capa de salida y el valor obtenido
                error = y[i] - a[-1]
                sumError = 0.5 * (error**2)
                #print("Error: ",sumError)
                totalErrores = totalErrores + sumError
                deltas = [error * self.activacion_prime(a[-1])]
                # Se empienza en la segunda capa
                for k in range(len(a) - 2, 0, -1): 
                    deltas.append(deltas[-1].dot(self.pesos[k].T)*self.activacion_prime(a[k]))
                self.deltas.append(deltas)
                deltas.reverse()
                # Backpropagation
                # Multiplcar los delta de salida con las activaciones de entrada para obtener el gradiente del peso.
                # actualizo el peso restandole un porcentaje del gradiente
                for i in range(len(self.pesos)):
                    capa = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.pesos[i] += tasa_aprendizaje * capa.T.dot(delta)
            print("Error MSEe: ",sumError)
            for wg in range(epocas):
                i2 = np.random.randint(jP.shape[0])
                a2 = [jP[i2]]
                for z in range(len(self.pesos)):
                    valorP = np.dot(a2[z], self.pesos[z])
                    activacionP = self.activacion(valorP)
                    a2.append(activacionP)
                # Calculo de la diferencia en la capa de salida y el valor obtenido
                errorP = kP[i] - a2[-1]
                secSumError = 0.5 * (errorP**2)
                deltas = [errorP * self.activacion_prime(a2[-1])]
                # Se empienza en la segunda capa
                for k in range(len(a) - 2, 0, -1): 
                    deltas.append(deltas[-1].dot(self.pesos[k].T)*self.activacion_prime(a2[k]))
                #self.deltas.append(deltas)
            print("Error MSEp: ",secSumError)
            errorTotal = ((34*sumError)+(17*secSumError))/51
            print("Error MSEt: ",errorTotal)
        print("No cumplio :(")



x = np.array([
    [0.0],[0.1],[0.2],[0.3],[0.4],[0.5],[0.6],[0.7],[0.8],[0.9],
    [1.0],[1.1],[1.2],[1.3],[1.4],[1.5],[1.6],[1.7],[1.8],[1.9],
    [2.0],[2.1],[2.2],[2.3],[2.4],[2.5],[2.6],[2.7],[2.8],[2.9],
    [3.0],[3.1],[3.2],[3.3],[3.4],[3.5],[3.6],[3.7],[3.8],[3.9],
    [4.0],[4.1],[4.2],[4.3],[4.4],[4.5],[4.6],[4.7],[4.8],[4.9],[5.0]
]) 

conjuntoK1 = np.array([
    [0.0],[0.1],[0.2],[0.3],[0.4],[0.5],[0.6],[0.7],[0.8],[0.9],
    [1.0],[1.1],[1.2],[1.3],[1.4],[1.5],[1.6],[1.7],[1.8],[1.9],
    [2.0],[2.1],[2.2],[2.3],[2.4],[2.5],[2.6],[2.7],[2.8],[2.9],
    [3.0],[3.1],[3.2],[3.3]
])

conjuntoPruebaK1 = np.array([
    [3.4],[3.5],[3.6],[3.7],[3.8],[3.9],[4.0],[4.1],[4.2],[4.3],
    [4.4],[4.5],[4.6],[4.7],[4.8],[4.9],[5.0]
])

yK1 = np.zeros((34,1))
yPK1 = np.zeros((17,1))

for a in range(len(conjuntoK1)):
    yK1[a]= math.sin(conjuntoK1[a])

for a in range(len(conjuntoPruebaK1)):
    yPK1[a]= math.sin(conjuntoPruebaK1[a])


RN = RedNeuronal([1,3,1],activacion ='tanh')
RN.entrenar(conjuntoK1, yK1, conjuntoPruebaK1, yPK1,tasa_aprendizaje=0.03,epocas=10000)
 
cont=0
for resul in x:
    #print("Y: ",y[cont],"YC: ",RN.predict(resul))
    cont=cont+1

deltas = RN.obtener_deltas()
grafica=[]
index=0
for arreglo in deltas:
    grafica.append(arreglo[1][0] + arreglo[1])
    index=index+1
plt.plot(grafica, color='r')
plt.ylabel('Error')
plt.xlabel('epocas')
plt.ylim([0, 1])
plt.show()
