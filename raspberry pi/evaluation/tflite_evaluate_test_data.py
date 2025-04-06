import tflite_runtime.interpreter as tflite
import numpy as np
import pickle
import time
import sys


def get_label(ls):
    index_of_max = 0
    for i in range(len(ls)):
        if ls[i] > ls[index_of_max]:
            index_of_max = i
    return index_of_max


print("Started loading data")
# load validation data
validation_data = pickle.load(open("test_data.pickle","rb"))
print("Done loading data")
print(type(validation_data))

validation_images = []
validation_labels = []

count = 0

print("start converting data to float 32 to use the model")
# convert data to float 32, without abusing RAM memory, i.e. convert in-place
for index in range(len(validation_data)):
    print("Load image #:",count)
    features, label = validation_data[index] # get values
    features = np.array([features]).astype(np.float32) # convert
    validation_data[index] = [features, label] # reassign values
    count += 1




# ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
# Convert to numpy array to type float32 in order to feed into the neural net
#validation_images = np.array(validation_images).astype(np.float32)
#validation_labels = np.array(validation_labels).astype(np.float32)


print("Done processing data")





print("loading model")
model_name = sys.argv[1] # take in the name of the model
interpreter = tflite.Interpreter(model_path=model_name)
print("finished loading model")
interpreter.allocate_tensors() # need to run this before execution
print("finished alocate tensors")

output_details = interpreter.get_output_details()
input_details = interpreter.get_input_details()

print(output_details)
print(input_details)

total_images = len(validation_data)
correct_prediction = 0
count = 0
total_time = time.time()
print("Evaluating...")
for features, label in validation_data:
    s = time.time()

    # pass the image in to be processed
    interpreter.set_tensor(input_details[0]['index'], features)

    # run inference
    interpreter.invoke()
  
  
    # get result
    result = interpreter.get_tensor(output_details[0]['index'])
    e = time.time()
    result_label = get_label(result[0])
    #print("__Image #__:", count+1)
    #print("Inference time:",e-s, "seconds")
    if result_label == label:
        correct_prediction += 1
    count += 1
        
    #print("Accuracy:", correct_prediction/count)
e = time.time()
print("Accuracy:", correct_prediction/count)
print("Total time to evaluate", len(validation_data),"images:", e - total_time, "seconds")
print("Average frame per second:", len(validation_data)/(e-total_time), "FPS")
    