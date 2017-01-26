# detect-relationships

This is an implementation of [Visual Relationship Detection with Language Priors](http://cs.stanford.edu/people/ranjaykrishna/vrd/).




---
# TODO

- vg_model
- vg_cnn
- visualize our predictions, heatmaps etc!!!


- clean / reorganize code . . .
    - remove unused stuff from utils
    - clean up code for rel_uid / obj_uid
    - implement a version that doesn't use visual module / language module 
    - maybe separate utils into separate files for vg/vrd or cnn/model?


- on macbook, test vg CNN for predicting test classes
    - how accurate is vrd CNN alone for predicting predicates?


- could batching have gone wrong with training cnn on VG?



---
### easy(ish)

- what happens when we dump images with less than some number of relationships?

- shuffle data points each iteration

- try w/ less random shit for K

- prepare for C/L/K ablation experiments

- try updating in batches instead of for each individual item (maybe for each image??











---
### paper

- explain Lu et al.'s results
- go through TODOs




---
### code

- data for graphs
    - vrd model -- accuracy, C,L,K, ..???
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




- email celu about:
    * recall @ k  results for predicate detection??  
    * testing -- in testing code, wtf is UnionCNN  --  why don't you include Z,s????


- more testing w/ non-alternating training . . ?

- save more metadata


























---
## scrap


- upload trimmed data AND data trimming scripts to google drive
    * make pretty and accessible to anyone w/ link

- early stopping if accuracy on validation set decrease?







