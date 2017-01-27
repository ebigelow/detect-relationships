# detect-relationships

This is an implementation of [Visual Relationship Detection with Language Priors](http://cs.stanford.edu/people/ranjaykrishna/vrd/).




---
# TODO

- vg_model
- visualize our predictions, heatmaps etc!!!


- clean / reorganize code . . .
    - remove unused stuff from utils
    - clean up code for rel_uid / obj_uid
    - implement a version that doesn't use visual module / language module 
    - maybe separate utils into separate files for vg/vrd or cnn/model?


- test vrd relnet CNN for predicting test classes



---
### easy(ish)

- what happens when we dump images with less than some number of relationships?

- shuffle data points each iteration

- try w/ less random sampling for K


- `batch_update` instead of for each individual item (maybe for each image?)

- experiment w/ non-alternating training

- save more metadata










---
### paper

- explain Lu et al.'s results

- more background on VG bullshit

- go through TODOs

- email celu about:
    * recall @ k  results for predicate detection??  
    * testing -- in testing code, wtf is UnionCNN  --  why don't you include Z,s????


---
### code

- data for graphs
    - vrd model -- accuracy, C,L,K
    - CNN: add stuff for tensorboard graph for VGG training . . .
        -> change `variable_scope` to `name_scope` for pretty visualization


- vg_model
    -> test cnn on test/train data
    -? how long to run on macbook
    -? parallel trials + grid search

- test ADADELTA / ADAGRAD

- test dist in VRD dataset of relation frequencies
    => Ours  : (1) 7.5%  (5) 34.9%  (10) 68.1%  (20) 79.1%
   
       TRAIN : (1) 17%   (5) 55%    (10) 79%    (20) 94%
       TEST  : (1) 18%   (5) 56%    (10) 81%    (20) 94%

 
    * X  what happens if we use the *same* probs from test to predict training?
    * what do our distributions of guesses look like? similar s-v-o vs. just v?
    * what is confusion matrix like -- does it make sense?
    * worst case: our model is always predicting k ordered according to prior only 

    * **test dist in VRD of relationships i,j,k triplets  for top 50, 100**



























---
## scrap


- upload trimmed data AND data trimming scripts to google drive
    * make pretty and accessible to anyone w/ link

- early stopping if accuracy on validation set decrease?







