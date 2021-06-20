import numpy
import scipy.special
import matplotlib.pyplot

class neuralNetwork():
    #initialize network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #learning rate
        self.lr = learningrate

        #init weight metrics
        #self.wih = numpy.random.rand(self.hnodes,self.inodes)-0.5
        #self.who = numpy.random.rand(self.onodes,self.hnodes)-0.5
        self.wih = numpy.random.normal(0.0, pow(self.inodes,-0.5), (self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes,-0.5), (self.onodes,self.hnodes))
        
        #sigmoid alma
        self.activation_function = lambda x:scipy.special.expit(x)

        
        pass

    #train network
    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        #matrisi agirlik matrisiyle carptik ve sigma fonk soktum...
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #cikani ağırlık matrisi ile carptim
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)

        #hatayi geri yayma
        delta_who = self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.who += delta_who  

        delta_wih = self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        self.wih += delta_wih

        pass
    
    #query the NN
    def query(self, input_list):
        #Gelen inputu matrise cevirdim
        inputs = numpy.array(input_list,ndmin=2).T

        #matrisi agirlik matrisiyle carptik ve sigma fonk soktum...
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #cikani ağırlık matrisi ile carptim
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

training_data_file = open("mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 10

for e in range(epochs):
    print ("Epochs: ",e)
    for records in training_data_list:
        all_values = records.split(",")
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01

        target = numpy.zeros(output_nodes)+0.01
        target[int(all_values[0])] = 0.99

        n.train(inputs,target)
        pass
    pass

test_data_file = open("mnist_test.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()

scoreCard = []

for records in test_data_list:
    all_values = records.split(",")
    correct_label = int(all_values[0])
    
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01

    outputs = n.query(inputs)
    label = numpy.argmax(outputs)

    if(label == correct_label):
        scoreCard.append(1)
    else:
        scoreCard.append(0)
        pass
    pass
print(scoreCard)
scoreCard_array = numpy.asarray(scoreCard)
print("Performance - ", scoreCard_array.sum()/scoreCard_array.size,"%")

all_values = test_data_list[1].split(",")
image_arr = numpy.asfarray(all_values[1:]).reshape(28,28)
matplotlib.pyplot.imshow(image_arr,cmap="Greys",interpolation="None")


#print(n.who)
#print(n.wih)