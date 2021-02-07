---
layout: post
title: Using ML to win FPL
subtitle: This post will walk you through how you can use Analytics & ML to remove human bias and make data driven decisions to improve your performance at FPL.
tags: [python,FPL]
---

# Using ML to get better at FPL

## Background
2020 was a particularly difficult year for everyone, with so much suffering around the world & life becoming still for everyone stuck in their homes due to lockdowns. As a sports lover, I found it particularly difficult to get used to a life without any live sport. Summers in India are particularly packed with excitement of home test series at the start of the year and then IPL for the months of April & May. Without access to any live sport, I turned to OTT platforms and started watching documentaries around football teams like Man City, Leeds United, Tottenham Hotspur, Sunderland etc. I finished all of them in no time and though I've been a cricket fan most of my life & watching football for me meant only international competitions like World Cup & Euros, I started getting attracted to club football.

As the spread of Covid started reducing around the world, football was the first sport to break through the shackles & premier league football resumed around the month of July. I got hooked to the sport and at the same time also got introduced to the world of Fantasy Football. I decided to give it a shot with new season beginning in September. Because I was still very new to football my knowledge around players capabilities in PL was still very limited, that led me to make several poor decisions. I also found myself getting biased around clubs that I liked & filling squad with players from those clubs. This led to a poor showing after the first few gameweeks of FPL. I decided to put my skillset of data analysis to work and use it to compensate for my lack of understanding of PL and understand how different players compare against each other & make transfer decisions backed by data. 

## Gathering Data

Following are the list of data sources used by me for the analysis
1. **Vaastav FPL Github** - This I think is the best available data repository for FPL containing historical data around players & team performances in FPL going back several years and gets refreshed on a weekly basis. Data is available [here](https://github.com/vaastav/Fantasy-Premier-League)
2. **FPL API** - I used FPL API to get info around upcoming fixtures & status of players fitness for a gameweek. Details around how to access the API can be found [here](https://medium.com/@frenzelts/fantasy-premier-league-api-endpoints-a-detailed-guide-acbd5598eb19)
3. **FivethirtyEight scores prediction** - I really like Nate Silver's [fivethirtyeight.com](http://fivethirtyeight.com)]which uses analytics & data science to predict outcomes of several real life events. They also make predictions around expected score for games in PL as well. I use this dataset capture level of difficulty of a fixture & scores prediction

## Data Overview
Our base dataset for this analysis is the Github repository mentioned above.  It contains various data points around a player's performance in a particular match as shown below - 

![enter image description here](https://lh3.googleusercontent.com/Yk-OyPDlKpyLErOsWZhjQCWlsn1TtmkDWKI9OTbIImwsVtZ2W1p74ViqESsXUoxj1Ery58DP4E3C0OVrtgC6l-ly-X1I3b5z_578cDQMiUJ8DY9Oz-MVr_mxedladOzTTTw0kpAwRojOdGGFmndJTYewZNCktFO4jnoWyuamUNPpauxMM_8Dn6p0-F0j2_Ebj9C4yOfpKD5IHeyC6fNyRrChvxj9xARfMOsI8HPxTxme7TqSKdJilsFGkffNA6duAs4y4jk8CJW8urn2r9v1p3oBTF5MjII7rbIHnvu1tlEOkLhOxxsdUcAu85XUwP68Tcx4HOW_DzNJRwc_VR7g-rLYDIG3LMkbNQEcGTvEn274j2DkIlL3i0ww8BH0pg4Xg1AMrMLlKU50BGGU8UtyatGoZA_8-gjWOQFfHiNmvv7hQA17XpEvOKi-_VQyDHjv8ZoafURD4uqfiyo4egrky88HP4ycJ6MVh6bKqptwgoecBsHgWnw0jTXz2XOH1OcKafhyjZ2yek5pMV7-F56buWfin3WkTF6Q8vD1xm0dv1fH1l-TLdNrd2nP7WaSK4dGBb-O_K_VLQXuNfaOkfj1f97G2_lo_IX2HahEdc1wunt7GQXSNa2tJTNqBkEDsF_F3LEq0Z9KpBQ-Jp3EJTFM-notL9w4fTYt26nIuh5gjipmJ3bz2x7EzHSCZa74=w1004-h268-no?authuser=0)

We combine the base datasets with additional data points from the FPL API & Fivethirtyeight datasets. Once we have the data ready, the first thing we do is to look at distribution of points scored by players during a game week. As the chart below suggests, huge majority of players score less than or equal to 2 points in a game week. Events providing returns like assists, goals & clean sheets are quite rare.  High number of players scoring zeros could be attributed to the fact that all premier league have big squads of around 25 players and only about 12-13 of them feature in a game. Since the distribution of points scored by players for more than 2 points is quite scattered, its quite difficult to actually predict the exact number of points scored by a player. 
![enter image description here](https://lh3.googleusercontent.com/BPueQDyIHcGhGKWpRBVbb19BVeE2_6cmTvCkXdQQzd3cETwqmzPRzFgqo1WaSDggPYqSg_Yygqd9JIKNa3D9OqT6O7ifYJoDHXckfPtmY3_B9_Sm3lroGE_OMb7fD2Y-GvXnaAfwORmDw6k9iYS0Gg3TTZiFcVe2QrVJqu_RrMesI2Hlb3AubrnmtSeXfequmjKjGrCXhjVFJlZRCoTapYwQqyabPJIvOOC3yFqezFx8Lvg0xRL_y7zcOJYKoED1JuYSyCOtnKetQnIhQXHRqXTR-MLMBOKSDH32r9ovwxRVKY-HHQsbImNFBd8ZqS33TxecyQtnVtscriUwFsEwzLZvlMSewqjMqX0hXr22S4gFyV8kPBwPMCQWMEB9vHtGqjK0_x22CEtGGu0--D1uCSvf-0PMfi_C2KPBNUrSjpm8PVRGFS3WkGuid2ezo7BgXb3XgFh3D06m-IEMTm1WxTT99NlF1pEk5l8XDymDfBzlHlRdXVwfktCZ8xqPQpVhW6foQOx40HpRZwY6OE2T8VBz2aNGOBrfdoruJRLT9gE-Hur4yF93AgXSfyRAh2lBJ00jLi-pXir0zwTqjVDTfGfsE81k-DfXdVss1rYOYR4YTLkdoCKR86TbIrZW2MoX7to4iwGaGScx4zVa8TTztBqAReppGhJqVyu8w0wHxf-thmRmLmD019wMlUYF=w1158-h525-no?authuser=0)

Let's now look at how the proportion of blanking(<=2 pts) and not blanking(2+points) looks by player position. Goalkeepers are most likely to blank based on the chart below, seems logical as well since only way GKs generally gain points is by maintaining clean sheets. For every other position, the proportions are quite similar. 
![enter image description here](https://lh3.googleusercontent.com/EXWfTKnVPsG1ANIbJraRR1jb7xHO-h1g12pIgfCGycSMb40aU-CdzHoq5niZzWvoYdfbymlqXWIkbFz8n-8H5EfRwnuWWYUUwjtpzKJbsaxP9D9y3kZHNfd4aSU9mR9zg6wH6eLtRiiyLybK-t4iqJcNmWxU3w5EYzmFiEB7rqNkFosxYSVRT1hhsUVvIiCcMyJ4j4nPiE1lkQiVN1DldpqcL-69leisLtqmS0HdhyWIfmuYBr5IEBFZHzX0rgoI8C9uLD4cCLRbZAwjE-AYYiKjEjdwU7TalrRFbfkf4mQLO0Xs4zF0ybhbq0zYzr66BZWleoo8LFyT36apxPyBjEYvjhhjC8sxxSY9-Y7n4Igkq_2vRl6i3NjorpoJ8sLVPiRYatTUKMPcH3DI6_qBMixFM9CZ8vU24qEIEGho-hxN-ZSguUIOvu1eLXyJ3Fn-DPGEDCylFS5hxOhuwpGD2XMB7N72iHQlPmh6rpduBzOQScHqZKJxZDkgA2KONSXbYN_Psp16nilL_iSJxXT5o957CUE1Ybu4YcV33hUf50b9cKr2TT6XvvcoOKtyZrAZUFR_T7-50GYCFY_XOSi7Oq73tt__Dzn8y991JNHZehzsuAAuY_kNoDa5JaHdmQot12rYWXkPUTq0Or4eUylzEZWC9Lqvv-s-KhMvgo_JONzcVd-pfS7CfwRxykgN=w1158-h525-no?authuser=0)

After taking a look at the above data points, I decided that it will be a better idea to build a classification model rather than a regression model for this exercise. **The objective of the model would be to identify players who are most likely to score 2+points in a week**. Since most players score less than 2 points in a week, this would be an example of imbalanced classification problem. We'll be building tree based classifiers for this problem.

## Feature Engineering
Quality of predictions for any model is directly correlated to the quality of features being fed into the model. I found the overall quality of data to be very rich & clean for this project, therefore lot of variables available in the raw datasets can directly be used as features.  On top of that I added several features on my end to capture form & opponent strengths into the model. After going through several iterations of model training on added features, here are the list of features I came up with -

### Player Performance 
* Influence, Creativity & Threat metrics
* Rolling average of points scored during last four weeks
* Position of a player
* Player's contribution to team's total points
* Rolling average of minutes played during last four weeks
* Goals scored, assists & clean sheets kept
* Yellow & Red Cards
* Number of incoming transfers by FPL managers during a game week

### Team Performance
* Diff. between team & opponent's position in the points table 
* Team's form over last four weeks
* Home or away fixture
* Projected scores from fivethirtyeight
* Total points scored by all players in the team
* Penetrations into opponent's box & number of penetrations allowed

## Model Design
My workflow has been designed in such a way that it uses historical data for the entire season until the latest gameweek for training the model & then makes prediction for the upcoming week. Predictions includes list of 11 players who are most likely to score more than 2 points during the gameweek. 
![enter image description here](https://lh3.googleusercontent.com/v60c_UhxWn15d3FXHmGvT1Gz_GKVNATErBdfL_2aoN2tFXsCDAYoz9CI9939SccPoYjffqKUL5s8CmVtR0kJ_03JXduim7HsiMt3zZQkQozT2UQIk9x3irL9lUkm0DyXxEWIXB-dQfMB4oFXNbSFhMsGsxhezNLD8LNI8k-A2AftuYeMXaqBdsFVNfrgYAPJ_hwlZ3VrAwP5UlkYWFbVZwIShVQqRE6IDHsmyOkxNfMCuYCv9Q4kJnXxoQHCg-cBU3rAT5Fzcu5cI8o8fVwrPdlbJQLzo5YtJJXJx4uad7VyiWjHlxQ70thrICoymz9D4ob306wVWWwNDmdJijDC9DmIsYraiKMc0JUJAsPEWTntR5dPts75uOYzqtHuGZbm6DylKUcbqlR-6sWb4eq3u_Y2WjSzRBcqADNxt0HqKC5bwaCxFU7bfR3gvhtDbgRhqor-MqY-0JO0ksYl3xFcxbCm9rsUSri4sDz_0RefybwO4MMTLTiU9koHRXAIYVeUpPx9argXLv4ugiAQRgQsWAtj3XqBPzPg8N68x3p024OB9XAjxTsiqFZg7GR8VGkB-V1Q0pv9jNyxZ_D_Q4-PSzVNLzWco6ZDlr7DXnoQI9fPbTZLbo_XMc87TvjLtSpb33IDDogijlIbmGmqp5Gnbk0YVdTWpa9yjR2OGGovqPEgk7O_49fK4v9Mmi6F=w872-h456-no?authuser=0)
The project is deployed as a pipeline that runs every week and uses historical data till the latest game week and makes prediction for the upcoming game week. 

## Model Development
As seen in the data earlier majority of the players during a particular gameweek tend to blank(score less than or equal to 2 points for appearance). Since football is a low scoring sport & events like goals etc. can be quite random, it is very hard to predict the exact number of points scored by a player during a game week. Therefore I turned this into a 2 class classification problem where i'm just trying to predict if a player would blank in a particular gameweek or score more than 2+ points. 

I decided to train tree based ensemble models using Random Forests & XGBoost and used a weighted average of the predictions used by both the models. The final output of model is a dataset with probability of each player not blanking during the upcoming game week. Here is the **feature importance** plot for the random forest model - 

![enter image description here](https://lh3.googleusercontent.com/Krl70aUFGAsTeP__xrI_l4MhXz2afnE_SXISZqQfQGzK9y1rTXQrKWlo2Rf5P5o8xmSAYZCaU9yNYNe89QHAorPq8dXS8ko-AKsAhYNZFe5IHsRsKAOFwcDqMjU0SVfNLS0yvTNY5x553pliiB0aBuVrc3IeI2-aeFNMvqe_rLMWJpxWZHasZxztJL4xIbpqUFcBybeZ55jiPju4etRGMXIi1aEBe_bxOP7l5AS0ISoX9hxUeNbz6B3HdsLO2lAsNl5PmBxaR_-ucKYw39pFnlTq-CKYlWVaN7udCJ8oHyoJFMRFU2Gr9pAqXk0j-nQzaxY_b9R10FPqW8KUgIiHljAjIqdM2ae-QEagw4vXzP89tbUSFerJe1hrVHpGsjO4hyTV_mlTqkLrbfoEGawO3qEtTWZISUonjj6oggV6mWh4bjMXqyIWIRAovP-H7jzO9hBfZhoebGnSqpa7M1Aj2vI48CGuT2Y-Z9a72-cEtiXmv7N-D5L27WoFtn_qU9j6wPOpW0oy_OCai5dKnv5so7VIr_K_ErG54v43M6bZBEoConhmDqNtrtFnpw8NRo0dTzmF1Qt0owTSh05Vg7f_s0r6XytmO_L0WQW2HmpjBNn3Bc7C3_SBMfYXqYdAzj8n35cJcEBQg9vtvZV9c6AZaDFLXfD7-HUFvgGo3VTl-WFp4AJbRjgW77qX1_VX=w1051-h901-no?authuser=0)
As we all know the 2020-21 season been a pretty weird one with teams going through runs of good & bad forms. The model seems to recognize that as well & the features with rolling average of last four weeks on points scored, ICT index etc. tend to have a lot of importance. This model in particular tends to answer the classic conundrum of FPL managers - form vs fixtures in favor of form. 

This model had an overall accuracy of 78%, but given that this is an imbalanced classification problem and we're interested in accurately identifying top 11 players who are likely to score, we're more interested in the true positive rate of the model.

![enter image description here](https://lh3.googleusercontent.com/OqqWptHtB5DaqOTdVh4xyE5PD8EuShhUnzoga4ozIlNU8Agq_kgxCzJsPeFRCbwggUu3PEIP-5DcwFYrpHFSeQSu0W61uwhF6Duuxnwmj-IYWCQ7YL0enlIKtibj2sYcdbBguuGYW20pQkb_qWkG5gvHKRm3B7KSgJZhTin3t01PFfk70830lDaHMV7zlB8liPIsB4kyyE_PvfnSd9ndo5lmwkwKQTANDdEHOKcKrmMP5piREAIDZlPhs3M3IOMGx264jKKIuRr7SBAJuU69CoGqXKEOCvoSZVoMxBGABCOwY0vFrgETiB73qd0nsA1LVd559_wHzOXfaa2_faqWjvZhFIIGzGTA7LSQOVhPbt_6QlA_k7cWrNwOFsTQ-ZKBRsnWVSnSe0AsR4WAfwdeEwxPIK3ORq-WemcFqYB4QNNixgBR78kvRgGR3PALKnTUewoKdBI8aOQPbKhO8L9_k_cUFCrmFO2GfudpaeP5A4Umwmdn7WpFYTcs3YZNxPP_Lgl9L_ShXc2OeSYy8XBQx062baJurhuaSBUzulnuu4D80oCLiqJs1dZzYoqYBAT_uvAT1l3z9bKS941AgWt5vFs-sHlw8G8ksJRA5e2_l9IC2yQTVHBdbr68mtzhRaPuLsbFAzzvs8iJwCKS5rWRUMT9uAaeQ3WouJ9kvIiXObIvs0MGodhigPtQSz_v=w697-h422-no?authuser=0)
 
![enter image description here](https://lh3.googleusercontent.com/DSkVsiZeyrzpX1E1v30Ci4P1lBnupOXP7iQWZ2vI5uhIoJh4ZXTqYTjuKo4ydsrg4UrEAsL9nRlaxqc_oqyZ1gSxl4TeunWNFWy-wIGZpUxxwfnHQbEx6fgaSjYQRkkFwRlXSdTag2_FGIlzD2zUZC46CFzm9Djjv6zM3JaRohDfKD6qJTtqdAxATKgr_OuKCZfISyPdSbnruZdDVGQwcW5SCzq-RBqdL5SnVjUPrSZG6QCDxAZ6UvujI72aGwejNRSq7zm1XKfMDpB0HktGg7s2LwUiARcGsz-o-BxBJ5X9PsYb7lDtu_4K1OvPE0x06ZWo6ImXvXLNBcdTGgR1Mirz8NSNz3mZTMbLEZgmsMNBriEdl2Y0MchJtNBvBQY2bdiSIQcFvmhdDHVBbzi-md3elNKL_UYa11-8zJaB1yaYnncWFwjAP-csbWFTsh-6Ga4NezNufWND9cw-0mvC-qOMAmaN6kQkqg173DZF95FUgd9ebsW5u9b30IUnWwNDE1dptfCfOH_jDYsf_-OFywhfNAPMHWmb9HsaSiu3XZXAC50b_NeXDeSykHYKowxJ_PlGTb9Mu_k4tCTIanW1IokDHlcAuCpe2vADa51VbxjUQocauE7CJsS9t2qGbrsT8qswXwRR7Dy246w1wQerBI8o5cevHHTxaK2eMo3gWxUCeAIwqLk4VZegu_8s=w394-h278-no?authuser=0)

The above charts show us that predicted probability distribution is heavily skewed to the left and very few players have predicted  probability over 0.5. **The AUC for ROC curve is 0.75 and the True Positive Rate is 70%**. This is not bad for an initial model but definitely room for improvement in future iterations.

## Output
The final output of the model is a list of 11 players who are most likely to score 2+points during the upcoming game week. The workflow runs for every game week and the output predictions are available on the **Streamlit** website created [here](http://fplgentoo.herokuapp.com).
![enter image description here](https://lh3.googleusercontent.com/EPsuJXpoYlz373ooluqOL9D-0jUnxrJ1iV156cHxoA-zr_bCjduo57MRFHzplZNm6C3zjcH6Zjk1MNJBndWDpafLdxpjRKJw4dNZ9-2deb98qFRwwjrzQ3yt7ws46v1xfQ8LE1zwcAlUxok67swXJ08fvTMWCTkKGFyo76rHZ-zdNssJ-X2WRiXHWZslq4bdY7lejHRPpNRn4JZH9XwcFSWVRtiRxca0FwGa07za3TwNF-_Y7PRa7NY06Rh006TGUVjMvwX0WR14uf4wWkv8k8AFwDS80xs88d2utmvqM7rV50A9z19HDWi3LHYXqvLsXvg4b-W5RmH9-mYqZxgp8cqNNt06i-oZL9zd8ZLW8Nopzjj4gk_w0y1r8zHP3n_s3tBTPYDPleJ55FwnKxcLFqyaqAomdfBmIBPeEB5TmEcHWtD16_ECKqbaX2exGDLfBH0FO_Lb8jpp62B_ESe_v9gXLZpL8OTSBBB_iNlROLcFdpAlix9otgvaBHvWqVyXAMis7-dRXtUptFV56p3wnOJZoQVEupiJMUOBjoFzR2cLDBU02LUTo22XjV0bDdEUrkbB_IA4kyvSBR0e_w38PI0E8xhRggYFf8JiIw5_I_XqMq-PqyNKPKo-Un99IewH6sQDo6QSoSfI0TYibz_8x899zviJh1SKQFrjSyCjData0_k9lmnDCE1JuWqT=w1812-h897-no?authuser=0)

I also look at points scored in previous game weeks by my team vs the actual dream team for the week & average score for the game week. So far we can say that the team predicted by the model is doing slightly better than the average human on points scored every game week. Looking at the predictions for the six game weeks the model has been running, model team scored 249 points vs sum of average score of 239 pts. 
![enter image description here](https://lh3.googleusercontent.com/cOn41P4A6Boo6gXTcMsmrIVVk6nkq5mCOz_l4skF_GO2wL3MDiiWH_IHcJ3rGc6hoyth0pVJcbrrf1dWE_9ETzZDAgpagrAEq9tIlF2z5YJj9Tu0AnNPVqCsPfSTkPqAGIU96zJrScP1194C_m2Xj-IzV9HqUNClBge1wzcxDmvOXwWk1EYwmw46vEO2ydr8q3Q_dhEg8OLnkkduFlixRGtj9zbTlYEsJHeYUp4BBIADtieAvXbW9qqi89LFnpB1Ronw4f9HuTilfIr9_IR082WtWHOCCtOftwqQFEt1x5fYDsTZcIx2fzx3IOCJ5f70e2C0hZXcxzY3C-CrTbsO1mVp-kMmvp9IEcFDugVLXjyNlYsas8f1ALkO7P0ucGc3Bj06E0rpsosFt33Fn47Zlm1mMOATEQyEZ2LFXE3tXeEDEoPhlYN9MCG4FJAHmH5gqt4IlbjxhWz6h2W0RlfkHSJ4ss69-NgwnBUeQKJzYNyFPsc0Kq4Cz4Oo86EHaDRb-uuc4oId6bLQqk-jduYD3dmkXOrv8T_foQ2LvsfXva1iKOcl9mKOiptXJh6j6d-Pfnpen9XjAcNo5yvaHiUHK07CuB9ZqgF8lFRFRLtKDkieOK5zFBGV-L1AeQKjzO_b202n9bazHZULyE8Mhv6FQ0exzyKWuJN6q2vskHgsCQ0qRBIKzIoVthk2ohop=w705-h414-no?authuser=0)




Hope is that model will continue to improve in its predictions through the season as it has access to higher volume of training data.

Code for the entire project can be found [here](https://github.com/arpitsolanki/FPLBot)

## Next Steps & Improvement 
This entire project was taken up by me as a Christmas project to understand PL football better & learn to use Streamlit for dashboarding. During the development of project I identified a few things that could be better - 
1. Gather different metrics for attacking & defensive footballers and train different models for each
2. Use Linear Programming to optimize the maximum points from the predicted team while ensuring that team budget doesn't cross 100M pounds. 
I'll be looking to work further on this & hopefully improve the performance of this model. 










