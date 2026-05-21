
import gradio as gr
import isolation_forest_engine as ife
from datetime import datetime, timedelta
from finance import StockDataFetcher

DISCLAIMER_TEXT = """
**⚖️ LEGAL DISCLAIMER:** *The predictions provided here are **purely exploratory** and for **research purposes only**. 
Investing based on this data is done at your own risk. The author assumes **no responsibility** for financial losses.* **[READ FULL DISCLAIMER BELOW]**
"""

FULL_LEGAL_TEXT = """
### 1. Not Financial Advice: *The information, analysis, and data visualizations presented here do not constitute financial, investment, or professional advice. The owner of this Space is not a licensed financial advisor or broker.*  
### 2. Accuracy & Model Risk: *Machine learning models are probabilistic and based on historical data (2013-2025). Past performance is not indicative of future results. Market conditions are volatile, and this model may produce false positives, false negatives, or inaccurate trend forecasts.*  
### 3. Assumption of Risk: *Any individual or entity that applies the predictions, insights, or data provided in this Space for making investment decisions or financial trades does so strictly at their own risk.*  
### 4. Limitation of Liability: *To the maximum extent permitted by law, the owner of this Space assumes no responsibility or liability for any financial loss, damages, or adverse outcomes resulting from the use or misuse of the information provided herein. By using this Space, you acknowledge that you are responsible for your own financial due diligence.*  
### 5. No Warranties: *This dashboard service is provided \"as is\" without any warranties of any kind, express or implied, including but not limited to the accuracy, completeness, or fitness for a particular purpose of the data.*
"""

QnA = {
    "SELECT YOUR QUESTION USING THIS DROP DOWN MENU": \
"THE ANSWER TO YOUR QUESTION WILL AUTOMATICALLY APPEAR IN THIS TEXTBOX.",
    "What does this application do?": \
"This application predicts whether a FAANG stock is trending up or trending down or neither.",
    "What is FAANG?": \
"FAANG is an acronym for Facebook, Apple, Amazon, Netflix, and Google.",
    "Does this app use ML/AI?": \
"Yes, this app uses ML/AI to predict the trend of a FAANG stock.",
    "Which dataset is used to train the model(s)?": \
"The dataset used to train the model(s) is the FAANG dataset that I created on Hugging Face, which you can access here: https://huggingface.co/datasets/ML-Owl/faang-engineered-time-series-features-2013-2025",
    "Does this application use real world data?": \
"Yes! When a user makes a request, data for a stock is fetched in real-time via the Yahoo Finance API. The ML predictions are all performed based on real-world data based on a user's request.",
    "How does the user interact with the application?": \
"The user selects the stocks of interest and provides a date for the stock to be analyzed. The user also provides their risk tolerance.",
    "What is the risk tolerance?": \
"Machine learning models in this application can be configured according to a trade off: either do not miss any trend up/down events while allowing some false positives or flag trend up/down events conservatively with minimal false positives. Therefore, how the model predicts will depend on the risk tolerance of a user specified via a slider input.",
} 

def get_answer(question):
    return QnA.get(question, "No answer available.")

def get_today():
    return datetime.now().strftime("%Y-%m-%d")

def QnA_UI():

    with gr.Column():
        gr.Markdown("## Alternatively, feel free to check out the dropdown Q&A section below to learn more on what to expect.")
        gr.Markdown("### (Feel free to ask relevant questions not included on the list and I will do my best to add them with my responses asap.)")
    
        QnA_dropdown = gr.Dropdown(label="What do you want to know?", choices=list(QnA.keys()))
        answer_text = gr.Textbox(label=" ", value=f"{get_answer(QnA_dropdown.value)}", interactive=False, lines=1, max_lines=20)

    QnA_dropdown.change(fn=get_answer, inputs=QnA_dropdown, outputs=answer_text)


def run_isf_inference(ifeObj, stocks_dropdown, calendar_item, risk_slider, progress=gr.Progress()):

    try:
        # Update ifeObj attributes with UI component values
        ifeObj.date_selected = calendar_item  # default date. To be updated by the UI
        ifeObj.update_stock_selected(stocks_dropdown)  # update's the object's attributes. (i.e. company name as well as the ticker)
        ifeObj.isf_score_threshold =  risk_slider  # default threshold specified by the user

        progress(0, desc="Fetching price data from Yahoo Finance")

        sdf = StockDataFetcher(ticker=ifeObj.stock_ticker_selected, use_adjusted=True)  # Adjusted price to be fetched by default.
        data = sdf.get_30workdays_frame(end_date=ifeObj.date_selected, price_type='Close')  # data  is a pandas Series. Close price is already Adjusted!

        #print(f"{data}")
        #print(f"days fetched:{len(data)}")

        progress(0.25, desc="Generating input features.")
        features, log_prices_fig = ifeObj.generate_input_features(data)
        print(f"Features: {features}")

        progress(0.50, desc="Running inference.")
        prediction_result = ifeObj.run_faang_inference(features)

        progress(1, desc="Inference completed!")
        
        return prediction_result, log_prices_fig

    except ValueError as e:
        return f"Data unavailable: {e}", None
    except RuntimeError as e:
        return f"Fetch failed: {e}", None
    except Exception as e:
        print(f"[ERROR] run_isf_inference: {type(e).__name__}: {e}")
        return "Prediction failed — please try a different date or stock.", None


def read_stock_OHLCV(stock_ticker_list: list, date_selected: str):
    
    sdf = StockDataFetcher(ticker="Void", use_adjusted=True)  # Adjusted price to be fetched by default.
    
    stock_prices = {}

    try:
        date_obj = datetime.strptime(date_selected, "%Y-%m-%d")
    except (ValueError, TypeError):
        return {"error": f"Invalid date format '{date_selected}'. Expected YYYY-MM-DD."}

    for eachTicker in stock_ticker_list:
        sdf.ticker = eachTicker  # update object's target ticker with the user request.
        # Each stock ticker will have a dictionary with the following keys: {"Date", "Open", "High", "Low", "Close", "Volume"}
        stock_prices[eachTicker] = sdf.fetch_last_valid_data(date_obj)

    return stock_prices


#------------------------------------------------------------------------------------------------------------------

ifeObj = ife.isfEngine()  # engine instance; use directly (do not wrap in gr.State for module-level use)
ifeObj.update_date_selected(get_today())  # set today's date as default at startup


with gr.Blocks() as demo:
    # The wrapper for everything in the launched demo.
    

    gr.Markdown("# FAANG Pulse AI - *The Stock Trend Oracle!*")
    gr.Markdown("""Predict whether a FAANG stock is trending up, trending down, or neither with a single click! <a href="https://huggingface.co/spaces/ML-Owl/FAANG-Pulse-AI/blob/main/README.md" target="_blank">View the README of this Space</a> for further details.""")


    gr.Markdown("---")
    gr.Markdown(DISCLAIMER_TEXT)
    gr.Markdown("---")

    #with gr.Row():
    #    QnA_UI()  # enable/disable as needed

    with gr.Row():

        with gr.Column():
            with gr.Row():
                with gr.Group():
                    gr.Markdown("### 🤖 ONE ML MODEL AVAILABLE:")
                    model_dropdown = gr.Dropdown(
                        label=" ",  # empty label to save space
                        choices=["Isolation Forest"],
                        interactive=False,
                        #info="More models will be added later on. Stay tuned!"
                    )  # One model implement4ed at the moment.

                with gr.Group():
                    gr.Markdown("### 📊 STOCKS LIST:")
                    stocks_dropdown = gr.Dropdown(
                    label="STEP 1 - Pick a stock of interest.",
                    value=ifeObj.stock_selected, choices=["APPLE", "AMAZON", "GOOGLE", "NETFLIX", "META (a.k.a. Facebook)"],
                    #info="Pick one of 5 stocks of interest."
                    )
                
                with gr.Group():
                    gr.Markdown("### 📅 DATE OF INTEREST:")
                    calendar_item = gr.DateTime(
                        label="STEP 2 - Pick a date to predict trend.",
                        type="string", # returns YYYY-MM-MM string
                        value=ifeObj.date_selected, # display today's date as default at startup
                        include_time=False
                    )
            
            with gr.Group():
                gr.Markdown("### ▶️ HIT BUTTON TO PREDICT TREND!")
                start_prediction_btn = gr.Button(
                    "FINAL STEP - Run Prediction",
                    variant="primary"
                )
            
            with gr.Group():
                gr.Markdown("### 🏁 PREDICTION RESULT:")
                predicted_regime = gr.Textbox(
                    label="Possible outcomes: TREND_UP, TREND_DOWN or NO_TREND.",
                    value="",
                    interactive=False
                )

            with gr.Group():
                gr.Markdown("### 📉 LOG PRICE MOVEMENT OVER THE LAST 30-DAYS:")      
                log_prices_plot = gr.Plot(label=" ")  # Visual plot element

        with gr.Column():
            with gr.Group():
                gr.Markdown("### 🎲 ADJUST YOUR RISK TOLERANCE AND VERIFY DECISION THRESHOLD IN HISTOGRAM BELOW:")
                risk_slider = gr.Slider(
                    label="STEP 3 - Moving the decision threshold right increases risk tolerance.",
                    minimum=ifeObj.slider_cfg['min'],
                        maximum=ifeObj.slider_cfg['max'],
                        value=ifeObj.slider_cfg['init'],
                        step=ifeObj.slider_cfg['step']
                    )
                
                risk_tolerance_plot = gr.Plot(label=" ")  # Visual plot element

    #########################################################
    # Following are the invisible elements for defining backend APIs
    # elem_id unique assignments are required to avoid conflict with the existing VISIBLE elements.
    with gr.Row(visible=False):
        api_in_stock_tickers_list = gr.JSON(visible=False, label="Stock List Input", elem_id="api_json_input")
        api_in_date_selected = gr.DateTime(visible=False, label="Date Selected Input", type="string", include_time=False, elem_id="api_datetime_input")
        api_out_stock_prices = gr.JSON(visible=False, label="Stock Prices Output", elem_id="api_json_output")
            # Virtual trigger for the backend API
        api_stock_prices_query_trigger = gr.Button(visible=False, elem_id="api_button_input")

    #########################################################

    # EVENT LISTENERS ARE INCLUDED BELOW (these also define the APIs that will be exposed!):

    # API for fetching stock prices (virtual trigger listener)
    api_stock_prices_query_trigger.click(
        fn=read_stock_OHLCV,
        inputs=[api_in_stock_tickers_list, api_in_date_selected],  # order matters!
        outputs=api_out_stock_prices,
        api_name="get_prices_on_date",
        api_visibility="public"  # also the default
    )

    # Update date selected when stock changes
    stocks_dropdown.change(
        fn=ifeObj.update_stock_selected,
        inputs=stocks_dropdown,
        outputs=None,
        api_name="pick_the_stock",
        api_visibility="private"  # not exposed as part of API
    )  # Updates object's attribute directly.

    # Calendar change updates
    calendar_item.change(
        fn=ifeObj.update_date_selected,
        inputs=calendar_item,
        outputs=calendar_item,
        api_name="pick_the_date",
        api_visibility="private"  # not exposed as part of API
    )  # Updates object's attribute directly.

    # Update plot when slider changes, and show plot on initial page load.
    risk_slider.input(
        fn=ifeObj.update_histogram_plot,
        inputs=risk_slider,
        outputs=risk_tolerance_plot,
        api_name="update_risk_tolerance",
        api_visibility="private"  # not exposed as part of API
    )

    # Periodic today's data update
    # When Huggig Face Space sleeps, this update will not happen! Hence the additional update in demo.load() as back up.
    timer_12hrs = gr.Timer(43200)  # 12 hours in seconds
    timer_12hrs.tick(
        fn=ifeObj.update_todays_date,
        outputs=[calendar_item],  # updates today is reflected to the calendar item as well to remain in sync.
        api_visibility="private"   # not exposed as part of API
        )  # update today's date at every timer tick event (i.e., once every 12 hours)

    # Connect the button to the function (only UI components as inputs; ifeObj is used inside the function)
    start_prediction_btn.click(
        fn=run_isf_inference,
        inputs=[gr.State(ifeObj), stocks_dropdown, calendar_item, risk_slider],
        outputs=[predicted_regime, log_prices_plot],
        api_name="run_trend_prediction",
        api_visibility="public"  # also the default
    )

    # Show initial histogram on page load (not inference)
    # Following load will be called each time the space wakes up!
    demo.load(
        fn=lambda: [ifeObj.update_todays_date(), ifeObj.update_histogram_plot(ifeObj.slider_cfg["init"])],
        inputs=None,
        outputs=[calendar_item, risk_tolerance_plot],
        api_visibility="private"   # not exposed as part of API
    )

    # Waiver of Liability as drop down text at the bottom.
    gr.Markdown("---")
    with gr.Accordion("📜 FULL TERMS OF USE & LIABILITY WAIVER:", open=True):
        gr.Markdown(FULL_LEGAL_TEXT)
    gr.Markdown("---")

    # Call http://127.0.0.1:7860/gradio_api/info to verify the API is working as expected.
    demo.launch(
        footer_links=["api", "gradio", "settings"], # ensures the API rendering is completed before the GUI is launched (for improved reliability)
        quiet=False # Keep this False for now to see if any 'Schema Errors' pop up
    )  


"""
if __name__ == "__main__":
    

    stock_ticker_list = ["AAPL", "AMZN", "GOOG", "NFLX", "META"]
    datetime = datetime.strptime("2026-03-15", "%Y-%m-%d")
    #datetime = datetime.now()
    stock_prices = read_stock_OHLCV(stock_ticker_list, datetime)
    
    # ifeObj = ife.isfEngine()  # engine instance; use directly (do not wrap in gr.State for module-level use)
    # sdf = StockDataFetcher(ticker="AAPL", use_adjusted=True)  # Adjusted price to be fetched by default.
    # data = sdf.get_30workdays_frame(end_date="2026-03-06", price_type='Close')  # data  is a pandas Series. Close price is already Adjusted!
    # ifeObj.generate_input_features(data)

"""