# Predicting user contribution in group recommendation and group decision
Python code for a project focusing on "Group Recommendation Systems (GRS)" and "Group Decision Making," aimed at predicting user contribution in the final decision.
### Project Description
User contributions play a critical role in influencing the dynamics and decision-making processes within a group. Each user brings unique insights, perspectives, and preferences, which collectively shape the group's choices. Considering the crucial importance of individual contributions, this project presents an innovative aggregation method for Group Recommendation Systems (GRS). We introduce two methods to measure the user's contributions in a group. The first method is based on Shapley values (ShV) and diversity between the group members, while the second one is based on Wonderful Life Utility (WLU) and similarity between the group members. These methods are validated across two datasets - food and car - which uses pairwise preference data. The experimental results demonstrate the system’s remarkable performance in recommending the right choice to the group. These methods were also tested with different grouping techniques, such as user clustering, to evaluate their performance. 
The logical diagram illustrating the process of the proposed method is as follows:
<p align="center">
<img style="width: 60%;" src="https://github.com/RozaAbolghasemi/User_Contribution_GRS/blob/main/FlowChart.png">
</p>

## Execution Dependencies
The codes can be run directly.
Also, the python code can be run by: 
```
python ./Shapley_Latest.py
```

We are using pandas, numpy, sklearn modules. Install them by
running.
```
pip install numpy
pip install pandas
pip install sklearn
```
The hyperparameters related to group sizes, and number of recommended (top) items can be adjusted to have the best results.

### Dataset


* Food dataset: This dataset comes from an online experiment, which focused on group decision-making about food preferences. Participants were grouped into fives and asked to update or maintain their food choices based on the group's average opinion. The experiment involved an online interface where participants provided pairwise scores for various food pairs. The interface displayed a probability score to indicate preferences. The data collected, presented in matrices, facilitated the study of consensus-building within groups. The paper details the methodology and experimental design, aiming to predict missing pairwise preferences in group decision-making.

* Car dataset: The dataset includes [car preferences](http://users.cecs.anu.edu.au/~u4940058/CarPreferences.html) gathered by Abbasnejad et al. in 2013. It was collected from 60 participants in the United States through [Amazon's Mechanical Turk](http://mturk.com). The dataset features ten different cars, which are compared as individual items. Each participant provided responses for all 45 possible pairs of items, leading to a total of 90 observations per participant. Besides the pairwise preference scores, the dataset contains two additional files with user attributes (education, age, gender, and region) and car attributes (body type, transmission, engine capacity, and fuel type).

----------------------------------------------------------------------

**License**
[MIT License](https://github.com/RozaAbolghasemi/User_Contribution_GRS/blob/main/LICENSE)

----------------------------------------------------------------------
