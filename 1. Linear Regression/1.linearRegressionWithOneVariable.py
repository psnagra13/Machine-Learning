'''
    Linear Regression with only One Variable

        X : input features with only 1 dimension
        Y : expected results

'''


# initialize dummy data
def generate_input_data(size):
    # X=[1,2,3,4,5,6,7,8,9,10]
    # Y=[300,350,500,700,800,850,900,900,1000,1200]
    X = []
    Y=[]
    for i in range (size):
        X.append(i)
        Y.append(i*2)
    return X,Y

def initialize_parameters():
    w=0.8
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

    for i in range(len(X)):
        y = X[i]*w +b
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

def update_parameters(w,b,dw,db, learning_rate=0.001):
    w = w - learning_rate*dw
    b = b - learning_rate*db
    return w,b



if __name__ == "__main__":
    X,Y = generate_input_data(100)
    w,b = initialize_parameters()

    print (X)
    print (Y)

    Y_hat = make_prediction(X,w,b)
    print (Y_hat)



    loss = calculate_loss(Y,Y_hat)
    dw, db = calculate_gradient(X, Y, Y_hat)
    w, b = update_parameters(w,b,dw,db)

    print ('loss = ' + str(loss))
    print (str(dw) +'  '+ str(db))
    print (str(w) +'  '+ str(b))
    print ('\n')




    for i in range(1000):
        Y_hat = make_prediction(X,w,b)
        loss = calculate_loss(Y,Y_hat)
        dw, db = calculate_gradient(X, Y, Y_hat)
        w, b = update_parameters(w,b,dw,db)
        
        print ('loss = ' + str(loss))
        # print (str(dw) +'  '+ str(db))
        # print (str(w) +'  '+ str(b))
        print ('\n')

    print (str(w) +'  '+ str(b))

    

