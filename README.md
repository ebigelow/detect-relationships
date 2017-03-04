# detect-relationships

This is an implementation of [Visual Relationship Detection with Language Priors](http://cs.stanford.edu/people/ranjaykrishna/vrd/).








# TODO new








- visualize hits & misses in testing data


x compute new means for VRD images
x add TF summaries to training
x add different training operations
- train & test on VRD

- try new K sampling method
- run model on VRD



- compute new VG image mean
- check out VG data
    * stratified sample smaller dataset
    * IDEA: use a small sample of VG, then see if it works tested on VRD!
- train on smaller dataset
    * fix scene graph indexing -- 0 to 108,000









# TODO newest

basic stuff i need ASAP

- check out the distribution for miniVG
- retrain VRD (relabeled) & miniVG CNNs          --  ~15-30 hours
- confusion matrices for new VRD/VG models   
- visualize hits/misses

- why is there weight blowup in `vrd_model.py`?
- train vrd/vg models                            --  ~2 hours
- confusion & visualize hits for full vrd model

- Results
- Discussion




48 hours    - monday
4 days      - wednesday















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


- **email celu**
    * recall @ k  results for predicate detection??  
    * testing -- in testing code, wtf is UnionCNN  --  why don't you include Z,s????

- explain word2vec??

- more background on VG bullshit

- go through TODOs


- mention how tabularizing V,f in code speeds things up (runtime tests?)

- mention how we contributed to `github.com/visual_genome_python_driver` by adding local loading methods
    * split data up into "data batches" (for RAM), so we don't have to reload images for every TF batch


- rant about general issues with the concept of "a bunch of images with single-class object/predicate labels"
    * there's tons of each in every.single.image!!
    * explain that their "quadratic explosion" of O(N^2 K) detectors is a load of horseshit


---
### code



- **is it possible that we read K wrong, samples are only drawn from REAL data???**



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




- try with massively cut down model, with much more even distribution of object / predicates






















---
## scrap


- upload trimmed data AND data trimming scripts to google drive
    * make pretty and accessible to anyone w/ link

- early stopping if accuracy on validation set decrease?
