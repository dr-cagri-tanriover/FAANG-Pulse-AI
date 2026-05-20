---
title: FAANG Pulse AI
emoji: 📈
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 6.8.0
app_file: app.py
pinned: false
license: cc-by-4.0
short_description: AI driven FAANG stock trend spotter with user risk input.
datasets:
- ML-Owl/faang-engineered-time-series-features-2013-2025
tags:
- time-series
- finance
- stock-market
- trading
- trend-prediction
- forecasting
- anomaly-detection
- novelty-detection
- outlier-detection
---
[![HitCount](https://hits.dwyl.com/ML-Owl/FAANG-Pulse-AI.svg?style=flat-square&show=unique)](http://hits.dwyl.com/ML-Owl/FAANG-Pulse-AI)

# FAANG Pulse AI - *The Stock Trend Oracle!*

## Project Summary
This is an interactive inference application that indicates whether a FAANG stock is trending up, trending down or not trending at all based on a 30-day history where the user selects the last day of the time window. The application also allows the user to control the decision threshold (via a slider) based on personal risk tolerance. An API is also provided for users to use this inference application with more flexibility.

## Model Description
**Model type:** Isolation Forest  
**Framework:** scikit-learn (*deployed via .skops*)  
**Goal:** Given a FAANG stock and a date, to identify whether that stock is trending up/down or not trending at all.  
**Dataset Used**: https://huggingface.co/datasets/ML-Owl/faang-engineered-time-series-features-2013-2025  

## Intended Uses & Limitations
***Intended Use:*** Exploratory analysis of FAANG stock market trends and ML research.  
***Limitations:*** Not optimized for high-frequency trading. Relies on daily adjusted close price (from which the key features are derived over a 30-day history). Does not account for real-time news sentiment.  

## Feature Engineering
The model has been trained on the following 4 key features:  
1 - zcr (the zero crossing rate)  
2 - slope (of log prices line)  
3 - volatility (of price variations)  
4 - trend strength (of price movements)  

The engineering process for the above features is described in detail as part of the (comprehensive) documentation of the [FAANG Dataset](https://huggingface.co/datasets/ML-Owl/faang-engineered-time-series-features-2013-2025). Please see that reference for further details.

## Model Training & Validation

FAANG stock price movements are dynamic by nature and therefore true trend up/down events are quite rare. This was also confirmed in the dataset used. Out of the total 16200 samples captured between 2013 and 2025 for all FAANG stocks, only 291 instances (i.e., \~1.8%) were labeled as trending up or down. Stationary price instances are even rarer with only 42 labeled samples! The remaining samples are labeled as either 'oscillating' (\~68%) or 'other' (\~30%). Please note some of the samples labeled as 'other' may include trend up/down samples albeit with a weak trend strength.

Given how rare the trend up/down events are in our dataset, the ML problem was framed as anomaly/novely detection rather than pattern or trend detection. Once the problem was frames as such, training, validation and test splits were defined as follows:

|Split|Time window|All Samples|Anomaly Samples|
|:----:|:----:|:----:|:----:|
|Train|[2013,2017]|6060|0|
|Validation|[2018,2021]|5040|119|
|Test|[2024,2025]|2505|67|

***Note:*** *Years 2022 and 2023 in our dataset have been left out due to their high volatility compared to the remaining years.* 

The train and validation splits were used to perform a grid search in the following hyperparameter space of Isolation Forest:

|Hyperparameter|Search Space|
|:--:|:--:|
|contamination|["auto"]|
|max_features|[0.5, 0.7, 0.8]|
|n_estimators|[100, 200, 500]|
|max_samples|[256, 512, 1024, 2048]|

***Note:*** *Contamination is left as default during grid search but was defined after the optimal hyperparameters were selected - more on this later.*

## Best Trained Model & Testing

The grid search optimization was performed based on Precision-Recall AUC scoring, where the best model scored \~0.182. This should not be confused with the ROC curve where a random prediction would score 0.5. 0.182 simply means the trained model thinks 18.2% of the data is an anomaly. The true ratio of anomaly in the validation set is 2.36%, which is << 18.2%, and is an indication that the trained model is performing 7.7x better than a random guess! (*for comparison, in real world fraud or intrusion detection systems, a PR-AUC of 0.20 is considered as an initial target!*) However, it is also important to note 0.182 indicates some anomalies in the dataset look like normal and are very difficult to isolate.

<details>
<summary>Click arrow to see the hyperparameters of the best model.</summary>

|Hyperparameter|Value|
|:-----:|:-----:| 
|max_features|0.8|
|n_estimators|200|
|max_samples|512|
</details>

After identifying the best model, the only missing part of the puzzle before testing performance was identifying the optimum contamination hyperparameter. In Isolation Forest, the contamination (factor) determines where the zero threshold which separates the anomalies (i.e., the negative scores) from the normal samples (i.e., the positive scores) should be placed. In order to identify the optimal contamination factor, we first apply the "*Youden's J statistic*", which simply identifies the TPR and FPR point on the ROC that maximizes the distance (J) to the 45-degree random chance line as shown below.

<div align="center">
  <img src="support/Youdens_J_statistic.png" alt="Youdens J statistic" height="200px">
</div>

To find "J", we run inference using best model on our test data split. The optimal score that maximizes "J" was found to be 0.03637 for our test split. We then categorize samples with scores less than 0.03637 as anomaly and the rest as normal instances. The ratio of the anomaly instances to the total instances in our test split gives us the optimal contamination factor, which is 0.2554.

Note that 0.2554 means 25.54% of all the samples in the test split are treated as anomalies (i.e., samples with negative scores) while in reality this is just 2.67%. Therefore, making a binary decision based on the contamination factor alone is likely to give too many false positives, which is what we need to avoid.

This brings us to the user interface of this application, and the recommended usage that puts the user in control in terms of making safer inferences (*depending on his/her risk tolerance*) compared to the binary one described above.

## USER INTERFACE & APPLICATION USE

This inference engine has a simple user-friendly interface, which is almost self-explanatory. Nevertheless, the details for each GUI element are provided below for clarity. 

### 1 - Stock List Dropdown:
User can choose one of the 5 FAANG stocks using the dropdown menu provided. By default, Apple (AAPL) is selected.

### 2 - Date of Interest Calendar:
The date selected is the last date of a 30-day history. The engineered features are calculated based on a 30-day history in this application. The earliest date that can be selected is 02/13/2013 limited by the dataset used. By default, the current date will be selected, and selecting a future date is not permitted.

### 3 - Risk Tolerance Slider:
This slider provides flexibility for the user in terms of where the decision threshold is placed. By default, the slider is set to the zero position as defined by the optimum contamination factor calculated for the test set (*as described previously*). In order to assist the user in terms of how to set the slider, a bar/histogram chart that shows the distribution of Trend and No Trend samples in the test set used is provided (*notice the low count of Trend instances compared to No Trend instances as previously explained*). When the user changes the slider position, the decision threshold (*the red dashed line*) on the bar plot will move accordingly. Moving the decision threshold towards the positive Trend Scores will increase the false positives (*i.e., choice for high risk takers*). Conversely, moving the threshold left will reduce the false positives but runs the risk of also missing true (rare) trends.  

### 4 - Run Prediction Button
After selecting the stock, setting the date and adjusting the decision threshold (using the risk slider), the user can hit this button to run the inference engine. The inference process takes only a few seconds, and the result is reported in a separate text box and as a line plot (see next section).

### 5 - Prediction Result and Price Movement:
After the inference is completed, this text box will indicate one of the three classes: TREND_UP, TREND_DOWN or NO_TREND. In addition to the predicted class, the user will also see the log price movement over the last 30-days starting on the user-selected date. This plot is provided as additional information to the predicted class. Price movement plot can be particularly useful in cases where the predicted class may conservatively indicate a "NO_TREND" while the human eye can discern an upward or downward trend on the plot.

### 6 - API Access:
You can also use this inference engine via API. To see the details on how to do that, please click the "Use via API" at the bottom of the application page.

## License
This space is made available under the ***Creative Commons Attribution 4.0 (CC by 4.0)*** License. Please keep the following in mind while using this space:  

**Permissions:** Users can share (copy/redistribute) and adapt (remix/transform) the material for any purpose, including commercial.  
**Attribution:** You must provide the name of the creator, a copyright notice, a license notice, and a disclaimer notice.  
**Flexibility:** This license allows for modifications to the original work, as long as credit is given.  
**Global/No Restrictions:** This license is valid worldwide and does not permit additional legal or technological restrictions that limit what others can do.  

## Citation

If you use information in this space in your research or project, please cite it in one of the formats below:

**APA Format:**


>Tanriover, C. [ML-Owl]. (2026). *FAANG Pulse AI: Market Trend Detection* [Demo Space]. Hugging Face. https://huggingface.co/spaces/ML-Owl/FAANG-Pulse-AI

**BibTeX Format:**
```bibtex
@misc{tanriover2026faangaipulse,
  author = {Tanriover, Cagri (ML-Owl)},
  title = {FAANG Pulse AI: Market Trend Detection},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.co/spaces/ML-Owl/FAANG-Pulse-AI}}
}
```

## Legal Disclaimer

**1. Not Financial Advice:** The information, analysis, and data visualizations presented here do not constitute financial, investment, or professional advice. The owner of this Space is not a licensed financial advisor or broker.

**2. Accuracy & Model Risk:** Machine learning models are probabilistic and based on historical data (2013-2025). Past performance is not indicative of future results. Market conditions are volatile, and this model may produce false positives, false negatives, or inaccurate trend forecasts.

**3. Assumption of Risk:** Any individual or entity that applies the predictions, insights, or data provided in this Space for making investment decisions or financial trades does so strictly at their own risk.

**4. Limitation of Liability:** To the maximum extent permitted by law, the owner of this Space assumes no responsibility or liability for any financial loss, damages, or adverse outcomes resulting from the use or misuse of the information provided herein. By using this Space, you acknowledge that you are responsible for your own financial due diligence.

**5. No Warranties:** This dashboard service is provided "as is" without any warranties of any kind, express or implied, including but not limited to the accuracy, completeness, or fitness for a particular purpose of the data.  

