import numpy as np
import pickle

# load training data
training_data = pickle.load(open("training_data.pickle", "rb"))

# count the number of each categories so we can create test data.
categories_counts = []
def get_category_count():
  count = 0
  current_class_num = training_data[0][1]
  for i in range(len(training_data)):
    if training_data[i][1] == current_class_num:
      count+=1
    else:
      categories_counts.append(count)
      count = 1
      current_class_num = training_data[i][1]
  categories_counts.append(count)
get_category_count()

# Split data: 60% for training, 20% for validation, and 20% for testing
validation_data = []
test_data = []
new_training_data = []

start_index = 0
end_index = -1
for count in categories_counts:
  print(count)
  end_index += count
  test_count = count * 20 / 100
  if test_count - int(test_count) >= 0.5:
    test_count = int(test_count) + 1
  else:
    test_count = int(test_count)

  # originally we went from beginning to end_index, but we performed one more experiment in addition to redo every experiments,
  # and we don't want the test data to be the same to avoid bias. Sorry for the confusion the code may caused.
  test_data.extend(training_data[end_index:end_index - test_count: -1])
  validation_data.extend(training_data[(end_index - test_count) : end_index - (test_count * 2): -1])
  new_training_data.extend(training_data[end_index - (test_count * 2) : None if start_index == 0 else (start_index-1) : -1])
  start_index += count

print("old training data", len(training_data))
print("Test data size:", len(test_data))
print("validation data size", len(validation_data))
print("training data size", len(new_training_data))
#save data
pickle.dump(test_data, open("test_data.pickle", "wb"))
pickle.dump(validation_data, open("validation_data.pickle", "wb"))
pickle.dump(new_training_data, open("new_training_data.pickle", "wb"))
