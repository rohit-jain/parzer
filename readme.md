# Statistical Dependency Parsing using SVM

## Abstract

In this project, I studied and re-implemented the technique for dependency parsing proposed by Yamada, Hiroyasu, and Yuji Matsumoto in "Statistical dependency analysis with support vector machines.". In addition to recreating the results we also experimented with the biomedical data from the GENIA biomedical corpus(Yuka et al., 2005) and the Spanish universal dependency dataset(McDonald et al.,2013) to understand out of domain implications.

#### Report, Presentation and data
PDF: <http://bit.ly/nlp-cs6741>

Presentation+Data+Models: <http://bit.ly/nlp-cs6741-full>

#### Instructions to run the code:

##### Folder Structure:
1. Converter - All files to pre-process data. Convert data sets to the parser formatting and also do the tagging.
2. Parser - All the files required to do the parsing
3. Data - Folder with all the pre-processed data, Download from the [link](https://drive.google.com/folderview?id=0B27E_jXWuNWdRFJMZ3ZYWFFwYTQ&usp=sharing)
4. Models - SVM models that have already been trained, download from [link](https://drive.google.com/folderview?id=0B27E_jXWuNWdNWk3V19RbHQ4QU0&usp=sharing)

##### Sections:
0. Pre-requisites
1. Running the code
2. Pre-processing and data format
3. Data Structures

#####Section 0: Pre-requisites

The following packages need to be installed:
    Python numpy, scipy, scikit-learn.
    If you are running an ubuntu machine on aws, run the following commands.
    sudo apt-get update
    sudo apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
    sudo apt-get install -y python-pip
    sudo pip install -U scikit-learn

   These packages are sufficient to run the parser with the pre-processed data in the data folder. If you wish to pre-process your own datastanford pos tagger and genia tagger need to be installed.
    Please refer to section 2

#####Section 1: Running the code
   
Make sure you are in the root directory i.e. dp
Assuming the pre-processing has been done these are the instructions to run the code using the existing models. To read more about pre-processing look at section 2.<br>

`python parser/__init__.py -c <cache size> -i <input dataset> -t <tagger>`

`cache size`: The cache size for SVM. significantly affects performance.

`input dataset`: The input dataset, can take the following 3 values<br>
    1. ptb - this will load the penn treebank dataset<br>
    2. genia - this will load the penn treebank corpus<br>
    3. spanish - this will load the spanish dependency corpus

`tagger`: the tagger to be used for test. the test files have been pre-processed to save time so this is mainly to load the tagged test files.
	it can take the following values:
    1. stanford - loads the stanford tagged test files
    2. gdep - loads the files pre-processed with the [genia tagger](https://github.com/saffsd/geniatagger), [python wrapper](https://pypi.python.org/pypi/geniatagger-python/0.1)

Eg. If you want to run penn treebank with stanford tagger for test sentences the command would be: `python parser/__init__.py -c 5120 -i ptb -t stanford`

Similarly for the genia treebank and geniatagger
`python parser/__init__.py -c 5120 -i genia -t gdep`

The models are stored in a folder under the "models" directory within the folder dataset_tagger. eg. the model for each post tage for penn tree bank with stanford tagger can be found under ptb_stanford.

######Note 1:
The parser looks for individual pos tag models under the model directory and only trains again if they are not found. Usuall training from scratch would take upto 10 hours for PTB sized dataset.
######Note 2:
For spanish only stanford tagger can be used because there are no pretrained models for spanish with geniatagger
######Note 3:
The code assumes file names for the pre-processed file names. If you wish to change these then you need to change the variable names in the main function of parser/__init__.py file.
The variables on lines 170-184 are commented and can be changed as per the need.

##### Section 2: Pre-processing
   All the pre-processing scripts are inside the converter folder. The pre-processing is different for each dataset and the goal is to get the training and testing files in the following format:
    Train file format:
    <Token> <Gold POS Tag> <Head node's position/dependency index>
    Eg. 
    It  PRP 1
    was VBD -1
    supposed    VBN 1
    to  TO  4
    be  VB  2
    a   DT  8
    routine JJ  8
    courtesy    NN  8
    call    NN  4
    .   .   1

    Test file format:
    <Token> <Assigned Tag>  <Gold POS Tag> <Head node's position/dependency index>
    Eg.
    You PRP PRP 1
    know    VBP VBP -1
    what    WP  WP  1
    the DT  DT  4
    law NN  NN  7
    of  IN  IN  4
    averages    NNS NNS 5
    is  VBZ VBZ 2
    ,   ,   ,   1
    do  VBP VBP 1
    n't RB  RB  9
    you PRP PRP 9
    ?   .   .   1

#####2.1 Pre-processing penn treebank
   PTB by default doesn't provide dependency trees. Original authors of the paper provided a tool(folder ptbconv.old) to convert to dependency tools based on collins head rules. the script convert.sh can be used to do that.

   `wsj_tagger.py`:
    This is used to produce the pre-processed file for test data for ptb. It can use either stanford tagger or genia tagger based on the flags specified in the main function of the script. 

######Note: For this to work the [stanford pos tagger](http://nlp.stanford.edu/software/tagger.shtml), [genia tagger](https://github.com/saffsd/geniatagger) and [genia's python wrapper](https://pypi.python.org/pypi/geniatagger-python/0.1) should be downloaded and installed.
For stanford tagger the environment variables CLASSPATH and STANFORD_MODELS are also required. sample values can be found in environment_variables file.

#####2.2 Pre-Processing genia corpus: 
 Training data:
    The code for this is available in `genia_converter.py`. It converts genia dependency tree to parser format using stanford dependency parser. A copy of stanford dependency parser is in the converter folder.
    
 Testing data:
    The code for this is available in `genia_test_tagger.py`. It can use either geniatagger or stanford tagger using the flag values in main function. Similar to wsj as a pre-requisite both need to be installed and environment variables need to be initialised.

#####2.3 Pre-Processing spanish universal dependency corpus:
The code for this is available in `universal_dependency_converter.py`. The file contains methods for producing both train and test tags. This is easliy extensible to other languages, only caveat is that the model for stanford pos tagger needs to change.

Spanish POS tags produced by the tagger comply with EAGLES standard but are simplified further ( Question 6: <http://nlp.stanford.edu/software/help/spanish-faq.shtml>). There is no straight forward way of mapping them to universal tagset. General mappings for universal tagset can be found [here](https://github.com/slavpetrov/universal-pos-tags/blob/master/es-eagles.map).

For the purpose of the experiment I ave done the mapping manually by inspecting both files and the mapping is available in spanish_mapping file.

#####Section 3: Data structures
2 primary data structure have been used. <br><br>
1. Node: code is available in node.py. This is the node of the parse tree, can contain any number of left and right children.<br>
2. Sentence: Data structure used to read and store the training and test sentences. Allows realing labeled/unlabeled sentences.
