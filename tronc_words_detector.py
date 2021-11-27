
"""
 Authors Agostinho, Beji, Watiez
June 2021 
"""

#NE PAS TOUCHER

ROOT_CHAR = ""
EMPTY_CHAR = ""
MODELS_DIR = "models/"

FILE_MODEL_EMBEDDINGS = "model_embeddings_mc_taille-50-125-epochs.p"
FILE_TRIE = "trie.p"
FILE_LEXICON = "lexique.txt"


#Modules à installer

#!pip install nltk
#!pip install gensim


import os
import sys
import re
import argparse


import nltk
"""Bird, Steven, Edward Loper and Ewan Klein (2009),
 Natural Language Processing with Python. O’Reilly Media Inc.
"""
from nltk.tokenize import RegexpTokenizer

import pickle

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

usage = """ Détecteur de mots tronqués dans un corpus de 1 ou plusieurs fichiers texte

   """+sys.argv[0]+""" [options] Fichier texte ou dossier contenant des fichiers texte
 
   Si un fichier est passé en argument, on cherche les mots tronqués dans celui-ci,
   Si c'est un dossier, on cherche dans tous les fichiers de ce dossier
 
 """

parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('test_file', default=None, help='Exemples utilisés comme voisins pour la prédiction KNN (au format .examples)')
args = parser.parse_args()

#---------------------------------------------------------------------------------------------------------------------------------------


class Trie:
    """
    La classe Trie permet de stocker un dictionnaire sous forme d'arbre.
    """
    def __init__(self):
        self.rootNode = Node(ROOT_CHAR)
    
    def addWord(self, w):
        temp_node = self.rootNode
        for char in w:
            temp_node = temp_node.addChild(char)
        temp_node.setLeaf(True)
            
    def __str__(self):
        return "-->".join(self.rootNode.wordsStartingFrom())
    
    def searchPrefix(self, w):
        cur_node = self.rootNode
        for char in w:
            next_node = cur_node.getChild(char)
            if next_node != None:
                cur_node = next_node
            else:
                return None
        return cur_node
   
    def searchWord(self, w):
        node = self.searchPrefix(w)
        return node != None and node.isLeaf()
        
    def fill(self, words_list):
        for word in words_list:
            self.addWord(word)
            
    def inspect(self):
        self.rootNode.inspect()

#---------------------------------------------------------------------------------------------------------------------------------------

class Node:
    """
    Node est la classe des noeuds qui composent la Trie
    """
    def __init__(self, c):
        self.char = c
        self.children = []
        self.leaf = False
    
    def __str__(self):
        return self.char
        
    def addChild(self, c):
        existingChild = self.getChild(c)
        if existingChild != None:
            return existingChild 
        else:
            new_node = Node(c)
            self.children.append(new_node)
            return new_node
        
    def getChildren(self):
        return self.children
    
    def getChild(self, c):
        for trie_node in self.children:
            if trie_node.getChar() == c:
                return trie_node
        return None
            
    def getChar(self):
        return self.char
        
    def isLeaf(self):
        return self.leaf
        
    def setLeaf(self, b):
        self.leaf = b

    def wordsStartingFrom(self, prefix=""):
        words_list = []
        if self.isLeaf():
            words_list.append(EMPTY_CHAR) 
        for trie_node in self.children:
            for s in trie_node.wordsStartingFrom():
                words_list.append(prefix + trie_node.getChar() + s)
        return words_list
        
    def inspect(self):
        for c in self.getChildren():
            c.inspect()
        print("{} - {}".format(self.char, self.leaf))


#---------------------------------------------------------------------------------------------------------------------------------------


class TroncWordsDetector:
    """
    Classe Contenant les méthodes et objets permettant de chercher
     des formes tronquées de mots dans un texte
    """
    
    def __init__(self):
        self.trie = Trie()
        self.ready()
        if FILE_MODEL_EMBEDDINGS not in os.listdir(MODELS_DIR):
            print("Modèle d'Embeddings non trouvé: Chercher {}".format(FILE_MODEL_EMBEDDINGS))
            #sys.exit()
        self.model_embeddings = pickle.load( open( MODELS_DIR + FILE_MODEL_EMBEDDINGS, "rb" ) )
        self.nb_tronc_in_files = []
        
    def ready(self):
        if FILE_TRIE in os.listdir(MODELS_DIR):
            self.trie = pickle.load( open( MODELS_DIR + FILE_TRIE, "rb" ) )
        else:
            if FILE_LEXICON in os.listdir(MODELS_DIR):
                with open(MODELS_DIR + FILE_LEXICON, encoding="utf-8") as f:
                    words_train = [line.rstrip() for line in f]
                    print(len(words_train))
                    for w in words_train:
                        self.trie.addWord(w.lower())
                pickle.dump( self.trie, open( MODELS_DIR + FILE_TRIE, "wb" ) )
            else:
                print("Aucun fichier de trie ou de lexique trouvé: chercher {} ou {}".format(FILE_TRIE, FILE_LEXICON))
                #sys.exit()
        
        
    def run(self, f):
        files_to_process = []
        if os.path.isdir(f): 
            for filename in os.listdir(f):
                files_to_process.append(os.path.join(f, filename))
        else:
            files_to_process.append(f)
            
        self.tok_and_lookForTronc(files_to_process)
        
    def tok_and_lookForTronc(self, files_to_process):
        for filename in files_to_process:
            tokenized_text = self.tokenize(filename)
            self.lookForTroncWords(filename, tokenized_text)
        print()
        for filename, count in self.nb_tronc_in_files:
            print("Fichier: {} - {} formes tronquées détectées".format(filename, count))
            
    """
    @returns list of lists
    """
    def tokenize(self, corpus_file):
        tokenizer = RegexpTokenizer(r'\w+')
        tokenized = []
        f = open(corpus_file, encoding="utf-8")
        s = f.read()
        f.close()
        sents = nltk.sent_tokenize(s, language='french')
        for sent in sents:
            tokenized.append(tokenizer.tokenize(sent))
        return tokenized

    def should_process(self, word, id_word):
        if len(word) < 2 or word.isupper() or (word[0].isupper() and id_word > 0):
            return False
        return True
    
    def lookForTroncWords(self, filename, tokenized_text):
        count = 0
        nb_sents = len(tokenized_text)
        for sent in tokenized_text:
            for j in range(len(sent)):
                word = sent[j]
                if not self.should_process(word, j):
                    continue
                    
                p = self.trie.searchPrefix(word.lower())
                if  p != None and not p.isLeaf():
                    possible_complete_forms = p.wordsStartingFrom(word)
                    bis_possible_complete_forms = possible_complete_forms
                    #----------
                    for w in possible_complete_forms:
                        #On vérifie si la possible forme complète a plus d'une lettre que la forme tronquée,
                        # ce qui pourrait être une forme au féminin ou un pluriel
                        if len(w) == len(word) + 1:  
                            bis_possible_complete_forms.remove(w)
                    if not len(bis_possible_complete_forms) == 0:
                        if len(possible_complete_forms) > 1:
                            most_probable_complete_form = self.check_complete_form(word, j, [w for w in sent])
                        else:
                            most_probable_complete_form = possible_complete_forms[0]
                        print("[" + word + "/{}] :  (fichier: {})".format(most_probable_complete_form, filename))
                        count += 1
        
        self.nb_tronc_in_files.append((filename, count))
        
    def check_complete_form(self, word, index, sentence):
        len_sent = len(sentence)
        for i in range(len_sent):
            if not i == index:
                sentence[i] = sentence[i].lower()
                if sentence[i].isnumeric():
                    sentence[i] = "*NUM*"
                elif not self.trie.searchWord(sentence[i]):
                    sentence[i] = "*UNK*"
        
        """win=2"""
        index += 2
        sentence = ["*d1*", "*d2*"] + sentence + ["*f1*", "*f2*"]
        context = [sentence[index - 2], sentence[index - 1], sentence[index + 1], sentence[index + 2]]
        
        for w, p in self.model_embeddings.predict_output_word(context, topn=10000):
            if w.startswith(word.lower()):
                return w


#---------------------------------------------------------------------------------------------------------------------------------------

f = args.test_file

import time
start_time = time.time()
detector = TroncWordsDetector()
detector.run(f)
print("Exécution terminée en - %s secondes -" % (time.time() - start_time))
