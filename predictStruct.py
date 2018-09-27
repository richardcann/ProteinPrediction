# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:59:25 2018

@author: rmcp1g15
"""
import numpy as np
import random
dictd = []
class Structure(object):
    def __init__(self, name, sequence, secondary):
        self.name = name
        self.sequence = sequence
        self.secondary = secondary
    aminoAcids = "ARNDCQEGHILKMFPSTWYVBZ"
    labels = "HEC"
    def getRepresentation(self, windowsSize):
        half_size = int( (windowsSize - 1)/2 )
        windowSequence = (("."*half_size) + self.sequence +("."*half_size))
        X = []
        Y = []
        #shuffle numbers i to get random windowing
        for i in random.sample(range(half_size, 
                                           len(self.sequence) + half_size), len(self.sequence)):
            #matrix of 22*windowSize
            Xi = np.zeros( (windowsSize, len(self.aminoAcids)), dtype=np.bool_)
            current_seq = windowSequence[i-half_size: i+half_size+1]
            for j in range(windowsSize):
                if current_seq[j] != ".":
                    #mark true for the amino acid present in current window
                    ind = self.aminoAcids.index(current_seq[j])
                    Xi[j,ind] = True
            X.append(Xi.flatten())
            Yi = np.zeros(len(self.labels), dtype=np.bool_)
            label_ind = self.labels.index(self.secondary[i-half_size])
            #belonging class
            Yi[label_ind] = True
            Y.append(Yi.flatten())
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

def getStructs():       
    with open('ss.txt', "r") as f:
        id = []
        toAdd = ''
        currentLine = f.readline()
        id.append(currentLine.rstrip().split(':sequence')[0])
        while len(id) < 782163:
            currentLine = f.readline()
            if currentLine.startswith(">"):
                if toAdd != '':
                    id.append(toAdd)
                    toAdd = ''
                if currentLine.rstrip().split(':')[2] == 'sequence' and len(id) + 1 < 782163:
                    id.append(currentLine.rstrip().split(':sequence')[0])
            else:
                toAdd += currentLine.rstrip('\n')
    
    allStructs = []
    with open('all.txt', "w") as f:
        print(len(id))
        addedToFile = 0
        for i in range(0, len(id), 3):
            if addedToFile < 150 and len(id[i+1]) > 20:
                f.write(id[i]+"\n")
                f.write(id[i+1]+"\n")
                addedToFile += 1
            second = (id[i+2]).replace("G", "H").replace("I","H").replace(" ", "C")
            second = second.replace("X","C")
            if "X" not in id[i+1]:
                if " " not in id[i+1]:
                    if "U" not in id[i+1] and len(id[i+1]) > 20:
                        newStruct = Structure(id[i], id[i+1], 
                                              second.replace("B", "E").replace("T", "C").replace("S", "C"))
                        dictd.append(id[i+1])
                        dictd.append(second.replace("B", "E").replace("T", "C").replace("S", "C"))
                        allStructs.append(newStruct)
    return allStructs 

import os
import re

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html).strip()
  return cleantext

def getTotalMismatches(seq1, seq2):
    total = 0
    if len(seq1) != len(seq2):
        print('somethings wrong')
    else:
        for i in range(len(seq1)):
            if seq1[i] != seq2[i]:
                total += 1
    return total

def getStructure(sequence, dictionary):
    found = None
    for s in dictionary:
        if s.sequence == sequence:
            print(found)
            found = s
    return found

def getResultsAccuracy(results):
    dictionary = getStructs()
    totalNum = 0
    totalMismatches = 0
    for struct in results:
        foundStruct,=[x for x in dictionary if x.sequence == struct.sequence.strip()]
        totalMismatches += getTotalMismatches(foundStruct.secondary, struct.secondary)
        totalNum += len(struct.secondary)
    return 1-(totalMismatches/totalNum)
        
allResults = []
other = []
rootdir = 'C:/Users/suki_/Documents/SecondaryStructure/results'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith(".simple.html"):
            path = os.path.join(subdir, file)
            with open(path, "r") as f:
                di = list(f)
                name = path
                seq = cleanhtml(di[5]).strip()
                secondary = cleanhtml(di[6]).strip().replace('-','C')
                newStruct = Structure(name, seq, secondary)
                other.append(seq)
                other.append(secondary)
                allResults.append(newStruct)
dictionary = getStructs()
totalNum = 0
totalMismatches = 0
for struct in allResults:
    foundStruct = getStructure(struct.sequence.strip(), dictionary)
    if foundStruct != None:
        print(True)
        totalMismatches += getTotalMismatches(foundStruct.secondary, struct.secondary)
        totalNum += len(struct.secondary)
accuracy = 1-(totalMismatches/totalNum)
print("Sample JPred Accuracy: ")
print(accuracy)

			