# Trait Specific App Rating Prediction


[![Build Status](https://travis-ci.org/fchollet/keras.svg?branch=master)](https://github.com/freefinity-project/riskisreal)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/freefinity-project/riskisreal/blob/master/LICENSE)

## What is this project about
Application ratings are usually based on user reviews. However, this system has an inherent bias towards applications which have been downloaded and rated by the users. A key challenge is to determine how good an app or game is, based on its gameplay videos, screenshots, application description, and other trivial features. In this exploratory study, we have compiled first of its kind, comprehensive dataset, but have also built predictive learning models which can mimic the human process of likability of a particular application or a game, based on visual and textual cues


## Datasets
We build scripts to compile the dataset from Google App Store.
Lets create a list of applications.

# Target Samples : 1494

**Games**

<center>
<table>
<tr>
    <th>Category</th>
    <th>Category Code</th> 
    <th># of Samples (Approx)</th>
 </tr>
<tr>
    <td>Action</td>
    <td>1</td> 
    <td>500</td>
 </tr>
<tr>
    <td>Adventure</td>
    <td>2</td> 
    <td>200</td>
 </tr>
<tr>
    <td>Arcade</td>
    <td>3</td> 
    <td>300</td>
 </tr>
<tr>
    <td>Board / Card / Casino / Casual</td>
    <td>4</td> 
    <td>200</td>
 </tr>
<tr>
    <td>Racing</td>
    <td>5</td> 
    <td>100</td>
 </tr>
</table>
</center>

No Stars / Unrated : 100



## Implementation
We used Python, scikit-learn, Pandas, NumPy, SciPy, Keras & Tensorflow.


## Running the code

- Install Python 2.7, Keras with Tensorflow backend
- Preprocess graphical features using running 
```
python apprater_nn.py
```
- Preprocess trivial and textual features using
```
python data_preprocessor.py
```
- Run the main program as
```
python main.py
```


