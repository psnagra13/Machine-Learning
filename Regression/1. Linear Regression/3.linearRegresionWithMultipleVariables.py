'''
    Linear Regression with Multiple Variables

        X : input features (m x n) 
            m: number of examples in training set
            n: number of features in single trainig example
        Y : expected results

'''
import numpy as np

class LinearRegression:
    def __init__(self):


        print ("INFO : Linear Regression with multiple variables")
        

    def train (self, X, Y, learning_rate=0.01, number_of_iterations=100 ):
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.m = len(X)
        if self.m == 0 :
            print ( "No data found in X")
            return
        self.n = len(self.X[0])

        self.w, self.b = self.initialize_parameters(self.n)

        for i in range(self.number_of_iterations):
            Y_hat = self.make_prediction(X,w,b)
            loss = self.calculate_loss(Y,Y_hat)
            dw, db = self.calculate_gradient(X, Y, Y_hat)
            w, b = self.update_parameters(w,b,dw,db)
            
            print ('iteration : '+ str(i) + 'loss : ' + str(loss))

       
    def initialize_parameters(self, n):
        w = np.random.rand(n)
        b=0
        return w,b

    def calculate_loss(Y , Y_hat):
        m = len(Y)
        loss = 0
        for i in range(m):
            loss = loss + ((Y[i] - Y_hat[i])**2)/m
        return  loss

    def make_prediction(X , w, b):
        Y_hat = []
        m = len(X)
        n= len(X[0])

        for i in range(m):
            y=0
            for j in range(n):
                y = X[i][j]*w[j]
            y=y+b
            Y_hat.append(y)
        return Y_hat

    def calculate_gradient(X,Y,Y_hat):
        m = len(Y)
        dW = 0
        db =0
        for i in range(m):
            diff = Y[i] - Y_hat[i]
            dW += (diff*X[i])/(2*m)
            db += diff/(2*m)
        return dW, db

    def update_parameters(w,b,dw,db, learning_rate=0.01):
        w = w - learning_rate*dw
        b = b - learning_rate*db
        return w,b


    def generate_input_data(m,n):
        X = np.random.rand(m,n)
        Y = np.sum(X, axis=1)
        return X,Y




if __name__ == "__main__":
    m=3
    n=2
    X,Y = generate_input_data(m,n)
    w,b = initialize_parameters(n)

    print (X)
    print (Y)
    print (w)
    print (b)

    Y_hat = make_prediction(X,w,b)
    print (Y_hat)



    loss = calculate_loss(Y,Y_hat)
    dw, db = calculate_gradient(X, Y, Y_hat)
    w, b = update_parameters(w,b,dw,db)

    print ('loss = ' + str(loss))
    print (str(dw) +'  '+ str(db))
    print (str(w) +'  '+ str(b))
    print ('\n')




    for i in range(2):
        Y_hat = make_prediction(X,w,b)
        loss = calculate_loss(Y,Y_hat)
        dw, db = calculate_gradient(X, Y, Y_hat)
        w, b = update_parameters(w,b,dw,db)
        
        print ('loss = ' + str(loss))
        print (str(dw) +'  '+ str(db))
        print (str(w) +'  '+ str(b))
        print ('\n')

    print (str(w) +'  '+ str(b))
    print (Y_hat)

    

