## Domain Specific Project Test Data 

- test_camel.txt

pair_left_func, pair_right_func, clone_label
1_L, 1_R, 1
2_L, 2_R, 1
3_L, 3_R, 0
i_L, i_R, 1

### Step 1. Checking Block-level Clone

Using a clone class of 1_L, 1_R, open data_func.jsonl to get the followings:
    - File path and range of function

Open data_block.jsonl to get the followings:
    - File path and range of blcok
    - If exist, save it data_func_block.jsonl

### Clone Prediction Result

- prediction_camel_before_adapt.txt 

pair_left_func, pair_right_func, clone_predict
1_L, 1_R, 1
2_L, 2_R, 0
3_L, 3_R, 1
i_L, i_R, 1

- prediction_camel_after_adapt.txt 

pair_left_func, pair_right_func, clone_predict
1_L, 1_R, 1
2_L, 2_R, 1
3_L, 3_R, 0
i_L, i_R, 1


### Refactobility

- prediction_camel_before_adapt.txt 

pair_left_func, pair_right_func, clone_predict, pair_left_in, pair_right_in, pair_left_out, pair_right_out 
1_L, 1_R, 1, 10, 10, 0, 0
2_L, 2_R, 0,,,,
3_L, 3_R, 1, 10, 15, 1, 0
i_L, i_R, 1, 10, 10, 1, 0

- prediction_camel_after_adapt.txt 

pair_left_func, pair_right_func, clone_predict, pair_left_in, pair_right_in, pair_left_out, pair_right_out 
1_L, 1_R, 1, 10, 10, 0, 0
2_L, 2_R, 1, 14, 14, 0, 0   // not found before adaptation
3_L, 3_R, 0,,,,             // found incorrectly before adaptation
i_L, i_R, 1, 10, 10, 1, 0


### Examples

MyObject.java, YourObject.java HerObject.java

- Left Function

void leftFunc() {
    // pre

    // body
    foo (myObject1, yourObject1, intVar1)
    ...
    // post
}

in(i) = {myObject1, yourObject1, intVar1}
inType(i) = {MyObject, YourObject, int}

- Right Function

void rightFunc() {
    // pre
    // body
    foo (myObject1, herObject1, intVar1)
    ...
    // post
}

in(i) = {myObject1, herObject1, intVar1}
inType(i) = {MyObject, HerObject, int}
