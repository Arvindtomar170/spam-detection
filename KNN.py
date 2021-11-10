import os
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    
    ham_files_location = os.listdir("dataset/ham")
    spam_files_location = os.listdir("dataset/spam")
    data = []
    
    for file_path in ham_files_location:
        f = open("dataset/ham/" + file_path, "r")
        text = str(f.read())
        data.append([text, "ham"])
    for file_path in spam_files_location:
        f = open("dataset/spam/" + file_path, "r")
        text = str(f.read())
        data.append([text, "spam"])
        
    data = np.array(data)
    
    return data



def preprocess_data(data):
    
    punc = string.punctuation         
    sw = stopwords.words('english')     
    
    for record in data:
        
        for item in punc:
            record[0] = record[0].replace(item, "")
             
        splittedWords = record[0].split()
        newText = ""
        for word in splittedWords:
            if word not in sw:
                word = word.lower()
                newText = newText + " " + word  
        record[0] = newText      
    return data


def split_data(data):
    
    features = data[:, 0]   
    labels = data[:, 1]    
    
    training_data, test_data, training_labels, test_labels =\
        train_test_split(features, labels, test_size = 0.27, random_state = 42)
    return training_data, test_data, training_labels, test_labels


def get_count(text):
    wordCounts = dict()
    for word in text.split():
        if word in wordCounts:
            wordCounts[word] += 1
        else:
            wordCounts[word] = 1
    
    return wordCounts

def euclidean_difference(test_WordCounts, training_WordCounts):
    total = 0
     
    for word in test_WordCounts:
        if word in test_WordCounts and word in training_WordCounts:
            total += (test_WordCounts[word] - training_WordCounts[word])**2
            del training_WordCounts[word] 
        else:
            total += test_WordCounts[word]**2
    for word in training_WordCounts:
            total += training_WordCounts[word]**2
            
    return total**0.5


def get_class(selected_Kvalues):
    spam_count = 0
    ham_count = 0
    for value in selected_Kvalues:
        if value[0] == "spam":
            spam_count += 1
        else:
            ham_count += 1
    
    if spam_count > ham_count:
        return "spam"
    else:
        return "ham"

def knn_classifier(training_data, training_labels, test_data, K, tsize):
    print("Running KNN Classifier...")
    
    result = []
    counter = 1
    training_WordCounts = [] 
    for training_text in training_data:
            training_WordCounts.append(get_count(training_text))  
            
    for test_text in test_data:
        similarity = [] 
        test_WordCounts = get_count(test_text)  
        for index in range(len(training_data)):
            euclidean_diff =\
                euclidean_difference(test_WordCounts, training_WordCounts[index])
            similarity.append([training_labels[index], euclidean_diff])
        
        similarity = sorted(similarity, key = lambda i:i[1])    
        selected_Kvalues = [] 
        for i in range(K):
            selected_Kvalues.append(similarity[i])
        result.append(get_class(selected_Kvalues))
        
        print(str(counter) + "/" + str(tsize) + " done!")
        counter += 1
        
    return result

def main(K):
    data = load_data()
    data = preprocess_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)
    tsize = len(test_data)
    
    result = knn_classifier(training_data, training_labels, test_data[:tsize], K, tsize) 
    accuracy = accuracy_score(test_labels[:tsize], result)
    
    print("training data size\t: " + str(len(training_data)))
    print("test data size\t\t: " + str(len(test_data)))
    print("K value\t\t\t\t: " + str(K))
    print("Samples tested\t\t: " + str(tsize))
    print("% accuracy\t\t\t: " + str(accuracy * 100))
    print("Number correct\t\t: " + str(int(accuracy * tsize)))
    print("Number wrong\t\t: " + str(int((1 - accuracy) * tsize)))

main(11)

tsize = [150, 300, 600, 900, 1200, 1582]
accuracy = [82.7, 84.0, 80.7, 79.0, 76.6, 76.7]

plt.figure()
plt.ylim(0, 100)
plt.plot(tsize, accuracy)
plt.xlabel("Number of Test Samples")
plt.ylabel("% Accuracy")
plt.title("KNN Algorithm Accuracy")
plt.grid()
plt.show()
def get_k():
    data = load_data()
    data = preprocess_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)
    tsize = 150
    
    K_accuracy = []
    for K in range(1,50, 2):
        result = knn_classifier(training_data, training_labels, test_data[:tsize], K, tsize) 
        accuracy = accuracy_score(test_labels[:tsize], result)
        K_accuracy.append([K, accuracy*100])
        print("training data size\t: " + str(len(training_data)))
        print("test data size\t\t: " + str(len(test_data)))
        print("K value\t\t\t\t: " + str(K))
        print("Samples tested\t\t: " + str(tsize))
        print("% accuracy\t\t\t: " + str(accuracy * 100))
        print("Number correct\t\t: " + str(int(accuracy * tsize)))
        print("Number wrong\t\t: " + str(int((1 - accuracy) * tsize)))
    K_accuracy_sorted = sorted(K_accuracy, key = lambda i:i[1])
    print(K_accuracy_sorted)
    print("MAX: " + str(max(K_accuracy_sorted, key = lambda i:i[1])))
    K_accuracy = np.array(K_accuracy)
    K_values = K_accuracy[:, 0]
    accuracies = K_accuracy[:, 1]
    
    plt.figure()
    plt.ylim(0, 101)
    plt.plot(K_values, accuracies)
    plt.xlabel("K Value")
    plt.ylabel("% Accuracy")
    plt.title("KNN Algorithm Accuracy With Different K")
    plt.grid()
    plt.show()