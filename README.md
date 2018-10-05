# SemanticTreeLSTM

This project explores the use of dependency trees in place of sentences in NLP. The specific NLP task used for the project is semantic relatedness. Two sentences are compared for semantic (meaning) similarity and a score from 1 to 5 is assigned where 1 means the sentences have no similarity at all and 5 means the sentences mean exactly the same thing. 

The project uses one model but two instances are tested. One instance of the model is trained on dependency trees that have been parsed into a sequence using depth-first exploration and the other instance, which is also the control, is trained on regular sentences. The depencency tree instance can be found in the file SemanticRelatednessLSTM-h.ipynb. The other instance is in SemanticRelatednessLSTM-h-Sentences_KL_train.ipynb

This project uses ideas from Tai et al (2015  https://arxiv.org/pdf/1503.00075.pdf)
