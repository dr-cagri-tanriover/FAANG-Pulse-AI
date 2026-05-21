
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import skops.io as sio


class isfEngine():
    def __init__(self):
        self.histDataFilePath = Path("support/normal_vs_anomaly_isf_scores.csv")

        try:
            _histDf = pd.read_csv(self.histDataFilePath)
        except FileNotFoundError:
            raise FileNotFoundError(f"Histogram data file {self.histDataFilePath} not found. "
                                    "Ensure the 'support/' directory is present."
            )

        _histDf = _histDf.dropna(how='all', axis=1)  # Ignore columns with all NaNs

        _histDf = self.column_combiner(_histDf)  # Combines TREND_UP and TREND_DOWN into TRENDING column and the rest into NOT TRENDING column.

        self.histData = dict.fromkeys(_histDf.columns) # empty dictionary with just the key names

        # valid isf scores in each column will be added to the dictionary (as list) for plotting a histogram later.
        for eachKey in self.histData.keys():
            self.histData[eachKey] = _histDf.loc[_histDf[eachKey].notna() == True, eachKey].to_list()

        self.slider_cfg = {'min': 0.0, 'max': 1.0, 'step': 0.0, 'init': 0.0}  # values used by Gradio slider element.

        self.generate_slider_values()  # updates self.slider_cfg once during initialization.

        self.todays_date = None
        self.update_todays_date()

        self.date_selected = '2026-01-01'  # default date. To be updated by the UI
        self.stock_selected = "APPLE"  # default selection at the start
        self.isf_score_threshold =  self.slider_cfg['init']  # default threshold at the start
        self._hist_cache = None  # filled by _ensure_hist_cache(); (bin_edges, bin_centers, bin_width, counts_by_label)

        try:
            unknown_types = sio.get_untrusted_types(file="optimized_isf_pipeline.skops")  # basically confirms I trust all types used.
            self.inference_pipeline = sio.load("optimized_isf_pipeline.skops", trusted=unknown_types)  # includes scaler and pretrained model in the same package!
        except FileNotFoundError:
            raise FileNotFoundError("Model file 'optimized_isf_pipeline.skops' not found. "
                                    "Pull Git LFS files with: git lfs pull"
            )

    def generate_slider_values(self, num_slider_steps=100, range_min=None, range_max=None):
        all_vals = [v for lst in self.histData.values() for v in lst]
        range_min = range_min if range_min is not None else min(all_vals)
        range_max = range_max if range_max is not None else max(all_vals)

        self.slider_cfg['min'] = range_min
        self.slider_cfg['max'] = range_max
        self.slider_cfg['step'] = (range_max - range_min) / num_slider_steps

    def update_todays_date(self):
        self.todays_date = datetime.now().strftime("%Y-%m-%d")
        return self.todays_date

    def update_stock_selected(self, stock_selected):
        name_stock_map = {"APPLE": "AAPL", "AMAZON": "AMZN", "GOOGLE": "GOOG", "NETFLIX": "NFLX", "META (a.k.a. Facebook)": "META"}

        if stock_selected not in name_stock_map:
            raise ValueError(f"Unknown/unsupported stock selected: {stock_selected}. "
                             f"Valid options are: {list(name_stock_map.keys())}"
                             )
        
        self.stock_selected = stock_selected  # Company name selected by the user.
        self.stock_ticker_selected = name_stock_map[stock_selected]  # Convert company name to its ticker symbol that will be used internally.
        return self.stock_ticker_selected

    def update_date_selected(self, date_selected):
        dt_earliest_date_supported = datetime.strptime('2013-02-13', "%Y-%m-%d") # earliest date supported by the dataset
        dt_todays_date = datetime.strptime(self.todays_date, "%Y-%m-%d")

        try:
            dt_date_selected = datetime.strptime(date_selected, "%Y-%m-%d")
        except (ValueError, TypeError):
            return self.date_selected

        # User specified date validation and update
        if dt_date_selected < dt_earliest_date_supported:
            self.date_selected = dt_earliest_date_supported.strftime("%Y-%m-%d")
        elif dt_date_selected > dt_todays_date:
            self.date_selected = dt_todays_date.strftime("%Y-%m-%d")
        else:
            self.date_selected = dt_date_selected.strftime("%Y-%m-%d")

        return self.date_selected

    def column_combiner(self, df):
        """Combines anomaly_TREND_UP and anomaly_TREND_DOWN into TRENDING column and the rest into NOT TRENDING column."""
        ps_trend = pd.concat([df['anomaly_TREND_UP'].dropna(), df['anomaly_TREND_DOWN'].dropna()], ignore_index=True)
        ps_notrend = pd.concat([df['normal_OSCILLATING'].dropna(), df['normal_OTHER'].dropna()], ignore_index=True)
        df = pd.DataFrame({'TREND': ps_trend, 'NO TREND': ps_notrend})
        return df

    def _ensure_hist_cache(self, n_bins=30, range_min=None, range_max=None):
        """Compute histogram bins and counts once; reused by plot_histograms so only the slider line changes."""
        if self._hist_cache is not None:
            return  # cache content built once for speed!

        all_vals = [v for lst in self.histData.values() for v in lst]
        _min = min(all_vals) if range_min is None else range_min
        _max = max(all_vals) if range_max is None else range_max
        bin_edges = np.linspace(_min, _max, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        counts_by_label = {}
        for label, data in self.histData.items():
            counts, _ = np.histogram(data, bins=bin_edges)
            counts_by_label[label] = counts
        self._hist_cache = (bin_edges, bin_centers, bin_width, counts_by_label)


    def update_histogram_plot(self, slide_value=0.0):
        """Return a matplotlib figure for Gradio's gr.Plot() to display."""
        return self.plot_histograms(slide_value)


    def plot_histograms(self, slide_value=0.0, n_bins=30, range_min=None, range_max=None):
        """Draw histograms from cached bin/counts; only the vertical line (slide_value) changes per call."""
        self._ensure_hist_cache(n_bins=n_bins, range_min=range_min, range_max=range_max)
        bin_edges, bin_centers, bin_width, counts_by_label = self._hist_cache

        fig, ax = plt.subplots()
        for label, counts in counts_by_label.items():
            ax.bar(bin_centers, counts, width=bin_width * 0.9, alpha=0.4, label=label, align='center')
        ax.axvline(slide_value, color='red', linewidth=2, linestyle='--', label='decision threshold')
        ax.set_xlabel('Trend Score')
        ax.set_ylabel('Count')
        ax.set_title('Trend/No Trend Histograms - Test Set')
        ax.legend()
        plt.close(fig)
        return fig

    def get_hist_bins(self, label):
        """Return the final bin edges, centers, and counts for a series after plot_histograms().
        Each series has its own bins because its data spans a different value range."""
        if not hasattr(self, "hist_bins"):
            return None
        return self.hist_bins.get(label)


    def generate_input_features(self, data_30days):

        # Order of features below follows the same order of features in trainset (excluding 'gt')
        features = {'slope': None, 'zcr': None, 'volatility': None, 'trend_strength': None, 'gt': None}

        features['slope'], plot_dict = self.get_slope_of_log_prices_line(data_30days)

        features['zcr'] = self.get_zcr_of_prices(data_30days)
      
        features['volatility'] = self.get_volatility_of_returns(data_30days)

        if features['volatility'] == 0.0:
            features['trend_strength'] = float('inf') if abs(features['slope']) > 0 else 0.0  # avoiding divide by zero potential issue.
        else:
            features['trend_strength'] = abs(features['slope']) / features['volatility']

        features['gt'] = self.get_ground_truth_label(features)

        log_prices_fig = self.plot_log_prices(plot_dict)

        return features, log_prices_fig


    def get_slope_of_log_prices_line(self, data_30days: pd.Series) -> float:
        """
        Compute the slope of the linear fit (line of best fit) to the log of the prices in the data_30days.
        Any $0 price is replaced with a very small value (1e-10) to avoid log(0) error.
        
        Args:
            data_30days: pandas.Series containing the sample data for prices on multiple days (given by Date Index)
            
        Returns:
            Slope of the linear fit (line of best fit) to the log of the prices in the sample_group
        """

        # Replace zero prices with very small value to avoid log(0) error
        prices = data_30days.replace(0, 1e-10)
        
        # Take log of prices
        log_prices = np.log(prices.values)  # accessing the values in the Series object to convert to numpy array
        
        # Create x values (time indices: 0, 1, 2, ..., n-1)
        x = np.arange(len(log_prices))
        
        # Fit line of best fit: y = a + bx using least squares
        # Using numpy polyfit (degree 1 for linear fit)
        # Returns [slope, intercept]
        coefficients = np.polyfit(x, log_prices, deg=1)
        slope = coefficients[0]  # First coefficient is the slope
        
        return float(slope), dict({'log_prices': log_prices, 'dates': data_30days.index})


    def get_zcr_of_prices(self, data_30days: pd.Series) -> float:
        """
        Compute the zero crossing rate of the price changes over a time window given by the number of elements in the data_30days.
        
        Args:
            data_30days: pandas.Series containing the sample data for prices on multiple days
            
        Returns:
            Zero crossing rate of the price changes in the data_30days
        """

        difference_in_prices = np.diff(data_30days.values)  # computing the difference in prices between consecutive days
        sign_of_differences = np.sign(difference_in_prices)  # computing the sign of the difference in prices (outputs 1, 0 or -1)

        # We need to handle 0 occurences in sign_of_differences before counting the zero crossings
        # Find index of first non-zero element
        non_zero_indices = np.nonzero(sign_of_differences)[0]
        if len(non_zero_indices) > 0:
            first_non_zero_index = non_zero_indices[0]
            if first_non_zero_index > 0:
                # need to pad the the signs moving towards index 0 with the sign of the first non-zero element
                sign_of_differences[:first_non_zero_index] = sign_of_differences[first_non_zero_index]
            #else: no need to pad the signs moving towards index 0 with the sign of the first non-zero element
        else:
            # All elements are zero - no price changes
            first_non_zero_index = None
            # simply pad all sign elements to a non-zero value (either 1 or -1)
            sign_of_differences[:] = 1

        # At this point, there may still be zero entries in sign_of_differences. We need to remove them.
        # We simply sustain the previous non-zero sign for the zero entries.
        sign_of_differences = np.array([sign_of_differences[i-1] if sign_of_differences[i] == 0 else sign_of_differences[i] for i in range(len(sign_of_differences))])

        # At this point, there should be no zero entries in sign_of_differences. We can begin counting the zero crossings now.
        total_sign_changes = int(np.sum(sign_of_differences[1:] != sign_of_differences[:-1]))

        zcr = total_sign_changes / (len(sign_of_differences) - 1)  # total number of sign changes divided by the total number of signs available.)
        
        return zcr

    def get_volatility_of_returns(self, data_30days: pd.Series) -> float:
        """
        Compute the volatility of the returns in the data_30days. Daily returns as calculated as logs.

        Args:
            data_30days: ppandas.Series containing the sample data for prices on multiple days
            
        Returns:
            Volatility of the returns in the data_30days
        """

        log_returns = np.log(data_30days.values[1:] / data_30days.values[:-1])  # ln(price(t)/price(t-1))

        daily_volatility = np.std(log_returns, ddof=1)  # Use n-1 in denominator (ddof=1 for sample std)

        return float(daily_volatility)


    def get_ground_truth_label(self, features: dict) -> str:
        """
        Compute the ground truth label for the given features.
        features.keys()=['slope', 'zcr', 'volatility', 'trend_strength']
        
        Args:
            features: Dictionary containing the features
            
        Returns:
            Ground truth label for the given features
        """

        gt_label = 'OTHER'  # by default
        
        if abs(features['slope']) < 0.001 and features['zcr'] < 0.46 and features['volatility'] <= 0.008:
            gt_label = 'STATIONARY'
        elif abs(features['trend_strength']) < 0.36 and features['zcr'] >= 0.46 and features['volatility'] > 0.008:
            gt_label = 'OSCILLATING'
        elif features['slope'] >= 0.003 and features['trend_strength'] >= 0.36 and features['zcr'] < 0.46 and features['volatility'] < 0.02:
            gt_label = 'TREND_UP'
        elif features['slope'] <= -0.003 and features['trend_strength'] >= 0.36 and features['zcr'] < 0.46 and features['volatility'] < 0.02:
            gt_label = 'TREND_DOWN'
        else:
            gt_label = 'OTHER'
        
        return gt_label


    def plot_log_prices(self, plot_dict: dict) -> plt.Figure:
        """
        Plot the log of the prices in the plot_dict.
        plot_dict.keys()=['log_prices', 'dates']
        
        Args:
            plot_dict: Dictionary containing the log of the prices and the dates
            
        Returns:
            Figure object containing the plot of the log of the prices
        """
        fig, ax = plt.subplots()
        ax.plot(plot_dict['dates'], plot_dict['log_prices'])
        ax.tick_params(axis='x', labelsize=10, rotation=45)  # x labels at an angle for legibility.
        ax.set_xlabel('Date')
        ax.set_ylabel('Log of Prices')
        ax.set_title('30-day history')
        ax.grid(True)
        plt.tight_layout()
        plt.close(fig)
        return fig


    def run_faang_inference(self, features):

        features_modified = {key:[value] for key, value in features.items()}  # this is a one row dataframe where each entry needs to be in square brackets for conversion on the next line!        features_df = pd.DataFrame(features)
        features_df = pd.DataFrame(features_modified)
        features_df = features_df.drop(columns=['gt'])  # drop the ground truth label column as it is not needed for inference.

        isf_score = self.inference_pipeline.decision_function(features_df)

        if isf_score <= self.isf_score_threshold:
            # there is a trend!
            if features['slope'] > 0:
                return "TREND_UP"
            else:
                return "TREND_DOWN"
        else:
            return "NO_TREND"
