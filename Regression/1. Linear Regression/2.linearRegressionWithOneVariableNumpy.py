import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# initialize dummy data
def generate_input_data(size):
    X = np.arange(float(size))
    Y= np.random.rand(size) *100
    return X,Y

def initialize_parameters():
    w=1
    b=0
    return w,b

def calculate_loss(Y , Y_hat):
    diff = Y_hat - Y
    diff = diff ** 2
    loss = diff.sum()/(2.0*Y.size)
    return  loss

def make_prediction(X , w, b):
    Y_hat = X*w + b
    return Y_hat

def calculate_gradient(X,Y,Y_hat):
    m = Y.size
    diff = Y_hat - Y
    dw = np.multiply(diff, X)
    dw = np.sum(dw) /m
    db = np.sum(diff) /m
    return dw, db

def update_parameters(w,b,dw,db, learning_rate=0.001):
    w = w - learning_rate*dw
    b = b - learning_rate*db
    return w,b



if __name__ == "__main__":
    X,Y = generate_input_data(1000)
    w,b = initialize_parameters()

    print (X)
    print (Y)

    X = (X - X.mean()) / X.std()



    Y_hat = make_prediction(X,w,b)
    print (Y_hat)

    costs = []


    loss = calculate_loss(Y,Y_hat)
    dw, db = calculate_gradient(X, Y, Y_hat)
    w, b = update_parameters(w,b,dw,db)

    print ('loss = ' + str(loss))
    print (str(dw) +'  '+ str(db))
    print (str(w) +'  '+ str(b))
    print ('\n')

    costs.append(loss)


    

    for i in range(10000):
        Y_hat = make_prediction(X,w,b)
        loss = calculate_loss(Y,Y_hat)
        dw, db = calculate_gradient(X, Y, Y_hat)
        w, b = update_parameters(w,b,dw,db)
        
        print ('loss = ' + str(loss))
        # print (str(dw) +'  '+ str(db))
        print (str(w) +'  '+ str(b))
        print ('\n')
        costs.append(loss)

    print (w)


    plt.title('Cost Function J')
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(costs)
    plt.show()


    

