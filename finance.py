"""
Finance API module for handling stock market data interactions using yfinance.

This module provides an object-oriented framework for fetching and processing
stock market data, particularly for time-series analysis and trend classification.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Union, Dict
import time



class StockDataFetcher:
    """
    Main class for fetching and managing stock market data from yfinance API.
    
    This class handles API interactions, data retrieval, and provides methods
    for accessing OHLCV data, adjusted prices, and time-series frames.

    NOTE: Multilevel columns ARE NOT supported.
    """
    
    def __init__(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        use_adjusted: bool = True
    ):
        """
        Initialize the StockDataFetcher with ticker(s) and date parameters.
        
        Args:
            ticker: Single ticker symbol (str) or list of ticker symbols
            start_date: Start date for data retrieval (str 'YYYY-MM-DD' or datetime)
            end_date: End date for data retrieval (str 'YYYY-MM-DD' or datetime)
            period: Alternative to start/end dates. Options: '1d', '5d', '1mo', '3mo',
                   '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
            use_adjusted: If True, use adjusted prices (default: True)
        
        Note:
            Either (start_date, end_date) or period should be provided, not both.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.use_adjusted = use_adjusted
        self._data: Optional[pd.DataFrame] = []  # empty list to be filled with data frames when fetch_data() is called.
        self._ticker_objects: Dict[str, yf.Ticker] = {}
        
        # Validate input
        if start_date and end_date and period:
            raise ValueError("Cannot specify both date range and period. Use one or the other.")
        #if not (start_date and end_date) and not period:
        #    raise ValueError("Must specify either (start_date, end_date) or period.")
    
    def fetch_last_valid_data(self, date_selected: datetime) -> pd.DataFrame:
        """
        Yahoo finance will return empty data frames in the following cases:
        1 - Current day until the stock market closes
        2 - Weekends
        3 - Public holidays

        This function is designed to fetch the last valid data on Fetch the last valid data on self.ticker 
        starting from the date_selected moving backwards one day at a time.
        The search depth is limited to 15 days of history for a fast response.
        """

        max_days_to_search = 15
        current_offset_days = 0

        max_retries = 5
        retry_delay_seconds = 5.0
        retries = 0

        while True:
            try:
                data = yf.download(
                    tickers=self.ticker,
                    start=date_selected - timedelta(current_offset_days),  # target date to fetch data from
                    end=date_selected + timedelta(1),  # one day ahead for the Yahoo finance interface to work!
                    auto_adjust=self.use_adjusted,
                    prepost=False,
                    threads=True,
                    multi_level_index=False,  # multi index not supported to be able to access the column names correctly
                    timeout=15
                )
            except Exception:
                # yf.download() can throw an exception and exit immediately!    
                data = pd.DataFrame()  # in case of any exception, treat it as empty data and retry until max_retries is reached.

            if not data.empty:
                break
            else:
                if retries < max_retries:
                    retries += 1
                    time.sleep(retry_delay_seconds)
                else:
                    if current_offset_days < max_days_to_search:
                        # Go back one day and request data again
                        current_offset_days += 1
                        retries = 0  # reset as data for a new day is requested
                    else:
                        break  # exit the while loop

        if data.empty:
            # This should never happen as we should have found data by now!
            raise ValueError(f"No data retrieved for ticker {self.ticker} after {max_days_to_search} days of search")

        # Data is valid. Return it in the following format: {"Date", "Open", "High", "Low", "Close", "Volume"}
        return {
            "Date": list(data.index.astype(str))[0],
            "Open": round(float(data['Open'].iloc[0]), 2),
            "High": round(float(data['High'].iloc[0]), 2),
            "Low": round(float(data['Low'].iloc[0]), 2),
            "Close": round(float(data['Close'].iloc[0]), 2),
            "Volume": int(float(data['Volume'].iloc[0]))
        }


    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch stock data from yfinance API.
        
        Returns:
            DataFrame with OHLCV data. If use_adjusted = True (default), columns: Close, High, Low, Open, Volume
            with each price automatically adjusted. 
            
            and optionally if use_adjusted=False, columns: Adj Close,Close, High, Low, Open, Volume where all prices
            are UNadjusted.
            
        Raises:
            ValueError: If data fetching fails or returns empty data
        """

        max_retries = 3
        retry_delay_seconds = 0.5
        retries = 0
        try:
            # Download data
            print("Data download in progress...")
            if self.period:
                while True:
                    try:
                        data = yf.download(
                            tickers=self.ticker,
                            period=self.period,
                            auto_adjust=self.use_adjusted,
                            prepost=False,
                            threads=True,
                            multi_level_index=False,  # multi index not supported to be able to access the column names correctly
                            timeout=15
                        )
                    except Exception:
                        data = pd.DataFrame()  # in case of any exception, treat it as empty data and retry until max_retries is reached.

                    if not data.empty:
                        break
                    else:
                        if retries < max_retries:
                            retries += 1
                            time.sleep(retry_delay_seconds)
                        else:
                            print(f"Failed to fetch data for ticker {self.ticker} after {max_retries} retries")
                            break

            else:
                while True:
                    try:
                        data = yf.download(
                            tickers=self.ticker,
                            start=self.start_date,
                            end=self.end_date,
                            auto_adjust=self.use_adjusted,
                            prepost=False,
                            threads=True,
                            multi_level_index=False,  # multi index not supported to be able to access the column names correctly
                            timeout=15
                        )
                    except Exception:
                        data = pd.DataFrame()  # in case of any exception, treat it as empty data and retry until max_retries is reached.

                    if not data.empty:
                        break
                    else:
                        if retries < max_retries:
                            retries += 1
                            time.sleep(retry_delay_seconds)
                        else:
                            print(f"Failed to fetch data for ticker {self.ticker} after {max_retries} retries")
                            break

            if not data.empty:
                print("Data download completed")
                self._data = data
                return data
            else:
                raise ValueError(f"No data retrieved for ticker(s): {self.ticker}")
            
        except Exception as e:
            raise ValueError(f"Error fetching data for {self.ticker}: {e}") from e


    def get_ohlcv_data(
        self,
        use_adjusted: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            use_adjusted: Override instance use_adjusted setting
            
        Returns:
            DataFrame with OHLCV columns
        """
        if len(self._data) == 0:
            self.fetch_data()
        
        use_adj = use_adjusted if use_adjusted is not None else self.use_adjusted
        data = self._data.copy()
        
        # Handles single ticker only
        if use_adj and 'Adj Close' in data.columns:
            # Replace Close with Adj Close
            data['Close'] = data['Adj Close']
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def get_price_series(
        self,
        price_type: str = 'Adj Close',
    ) -> pd.Series:
        """
        Get a specific price series (Open, High, Low, Close, Adj Close).
        
        Args:
            price_type: Type of price ('Open', 'High', 'Low', 'Close', 'Adj Close')
            
        Returns:
            Series with price data indexed by date
        """
        if len(self._data) == 0:
            self.fetch_data()
        
        data = self._data.copy()
        
        if price_type not in data.columns:
            raise ValueError(f"Price type {price_type} not found in data columns: {data.columns.tolist()}")

        return data[price_type]


    def compute_a_workday_date(self, start_date: Union[str, datetime], number_of_workdays: int = 30, direction: str = 'backward') -> datetime:
        """
        Compute a date that is a certain number of workdays (MONDAY-FRIDAY) away from a given date.

        Args:
            start_date: Start date
            number_of_workdays: Number of workdays to move away from the start date
            direction: Direction to move away from the start date ('backward' or 'forward')
            
        Returns:
            Date that is a certain number of workdays away from the start date
        """
        
        if isinstance(start_date, str):
            current_date = pd.to_datetime(start_date)
        else:
            current_date = start_date

        counted_days = 0  # start_date is also included in the range of workdays

        if direction not in ['backward', 'forward']:
            raise ValueError("Direction must be either 'backward' or 'forward'")

        if direction == 'backward':
            # End date is specified. We are seeking the start date.
            update_day = lambda current_date: current_date - timedelta(days=1)
        else:
            # Start date is specified. We are seeking the end date.
            update_day = lambda current_date: current_date + timedelta(days=1)

        first_entry = True
        while counted_days < number_of_workdays:
            # check the current_day first if it is a workday
            if not first_entry: # Skip the first day as it is already included!
                current_date = update_day(current_date)  # update the current date
            else:
                first_entry = False

            if current_date.weekday() < 5:  #  datetime codes days as 0: Mon 1: Tue .... 5: Sat and 6: Sun
                counted_days += 1  # advance counter since it is a workday
                

        return current_date  # inclusive date of the number of workdays away from the start date


    def get_30workdays_frame(self, start_date: Union[str, datetime]=None, end_date: Union[str, datetime]=None, price_type: str = 'Adj Close') -> pd.Series:
        """
        Extract a 30-workday time-series frame starting from a specific date.
        Either start_date or end_date must be provided but not both!

        If start_date is provided, the frame will start from the start_date, move FORWARD in time by 30 workdays and end at the end of the 30-workday period.
        If end_date is provided, the frame will start from the beginning of the 30-workday period, move BACKWARD in time by 30 workdays and end at the end_date.

        Args:
            start_date: Start date for the 30-day frame
            end_date: End date for the 30-day frame
            price_type: Type of price to extract ('Open', 'High', 'Low', 'Close', 'Adj Close')
            
        Returns:
            Series with 30 days of price data
        """

        if start_date is None and end_date is None:
            raise ValueError("Either start_date or end_date must be provided")
        if start_date is not None and end_date is not None:
            raise ValueError("Either start_date or end_date must be provided, but not both")

        offset_days = 0
        days_to_fetch = 30
        max_fetch_iterations = 10
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)

            # We need the following while loop as we cannot calculate public holidays that fall on a weekday where stock market is closed!
            self.start_date = start_date
            iterations = 0
            while len(self._data) < days_to_fetch:  # fetch data until we have at least days_to_fetch days of data
                if iterations >= max_fetch_iterations:
                    raise RuntimeError(
                        f"Could not collect {days_to_fetch} trading days for {self.ticker} "
                        f"starting {start_date.date()} after {max_fetch_iterations} fetch attempts."
                    )
                end_date = self.compute_a_workday_date(start_date, number_of_workdays=days_to_fetch + offset_days, direction='forward')
                self.end_date = end_date
                self.fetch_data()  # fetch data from yahoo finance for the specified data range.
                offset_days += (days_to_fetch - len(self._data))
                iterations += 1
        else:
            #end_date is not None
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
                end_date += timedelta(days=1)  # yfinance excludes the end date. So we start a day later to make include the end date specified by user!

            #start_date = self.compute_a_workday_date(end_date, number_of_workdays=30, direction='backward')

            # We need the following while loop as we cannot calculate public holidays that fall on a weekday where stock market is closed!
            self.end_date = end_date
            iterations = 0
            while len(self._data) < days_to_fetch:  # fetch data until we have at least days_to_fetch days of data
                if iterations >= max_fetch_iterations:
                    raise RuntimeError(
                        f"Could not collect {days_to_fetch} trading days for {self.ticker} "
                        f"ending {end_date.date()} after {max_fetch_iterations} fetch attempts."
                    )
                start_date = self.compute_a_workday_date(end_date, number_of_workdays=days_to_fetch + offset_days, direction='backward')
                self.start_date = start_date
                self.fetch_data()  # fetch data from yahoo finance for the specified data range.
                offset_days += (days_to_fetch - len(self._data))
                iterations += 1

        if len(self._data) != days_to_fetch:
            raise ValueError(f"Fetched {len(self._data)} active stock market days of data instead of {days_to_fetch}")
        # At this point we have a valid start_date and end_date to and a fetched self._data. 
        # Get price series of interest from self._data
        price_series = self.get_price_series(price_type=price_type)
                
        if len(price_series) == 0:
            raise ValueError(f"No data available for 30-day frame starting {start_date} and ending {end_date}")
        
        return price_series


    def get_30_day_frame(
        self,
        start_date: Union[str, datetime],
        price_type: str = 'Adj Close'
    ) -> pd.Series: 
        """
        Extract a 30-day time-series frame starting from a specific date.
        
        Args:
            start_date: Start date for the 30-day frame
            price_type: Type of price to extract ('Open', 'High', 'Low', 'Close', 'Adj Close')
            
        Returns:
            Series with 30 days of price data
        """
        if len(self._data) == 0:
            self.fetch_data()
        
        # Convert start_date to datetime if string
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # Get price series
        price_series = self.get_price_series(price_type=price_type)
        
        # Calculate end date (30 days after start)
        end_date = start_date + timedelta(days=30)
        
        # Filter data for the 30-day window
        frame = price_series[(price_series.index >= start_date) & 
                            (price_series.index <= end_date)]
        
        if len(frame) == 0:
            raise ValueError(f"No data available for 30-day frame starting {start_date}")
        
        return frame
    
    def get_all_30_day_frames(
        self,
        price_type: str = 'Adj Close',
        overlap: bool = False
    ) -> List[pd.Series]:
        """
        Extract all possible 30-day frames from the dataset.
        
        Args:
            price_type: Type of price to extract
            overlap: If True, create overlapping frames (every day). 
                    If False, create non-overlapping frames (every 30 days)
            
        Returns:
            List of Series, each containing a 30-day frame
        """
        if len(self._data) == 0:
            self.fetch_data()
        
        price_series = self.get_price_series(price_type=price_type)
        
        if len(price_series) < 30:
            raise ValueError("Insufficient data: need at least 30 days of data")
        
        frames = []
        step = 1 if overlap else 30
        
        for i in range(0, len(price_series) - 29, step):
            frame = price_series.iloc[i:i+30]
            if len(frame) == 30:  # Only include complete 30-day frames
                frames.append(frame)
        
        return frames
    
    def get_returns(
        self,
        price_type: str = 'Adj Close',
        log_returns: bool = False
    ) -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            ticker: Specific ticker (required if multiple tickers)
            price_type: Type of price to use for returns calculation
            log_returns: If True, calculate log returns. If False, calculate simple returns
            
        Returns:
            Series with returns data
        """
        price_series = self.get_price_series(price_type=price_type)
        
        # Filter out zero prices before calculating returns
        price_series = price_series[price_series > 0]
        
        if log_returns:
            # Log returns: ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
            log_prices = np.log(price_series)
            returns = log_prices.diff()
        else:
            # Simple returns: (P_t - P_{t-1}) / P_{t-1}
            returns = price_series.pct_change()
        
        return returns.dropna()
    
    def get_ticker_info(self, ticker: str) -> Dict:
        """
        Get additional information about a ticker (company name, sector, etc.).
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary with ticker information
        """
        if ticker not in self._ticker_objects:
            self._ticker_objects[ticker] = yf.Ticker(ticker)
        
        ticker_obj = self._ticker_objects[ticker]
        info = ticker_obj.info
        
        return {
            'symbol': ticker,
            'longName': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'currency': info.get('currency', 'N/A')
        }
    
    @property
    def data(self) -> pd.DataFrame:
        """Get the raw data DataFrame."""
        if len(self._data) == 0:
            self.fetch_data()
        return self._data
    
    @property
    def tickers(self) -> List[str]:
        """Get list of ticker symbols."""
        return self.ticker


class StockDataProcessor:
    """
    Utility class for processing and analyzing stock data.
    
    This class provides methods for calculating metrics like slopes, volatility,
    and other statistical measures needed for trend classification.
    """
    
    @staticmethod
    def calculate_slope(price_series: pd.Series, use_log: bool = False) -> float:
        """
        Calculate the slope of a linear fit to price data.
        
        Args:
            price_series: Series of prices
            use_log: If True, fit line to log of prices
            
        Returns:
            Slope value (b)
        """
        if len(price_series) < 2:
            raise ValueError("Need at least 2 data points to calculate slope")
        
        # Filter out zero prices if using log
        if use_log:
            price_series = price_series[price_series > 0]
            if len(price_series) < 2:
                raise ValueError("Insufficient non-zero prices for log calculation")
            y = np.log(price_series.values)
        else:
            y = price_series.values
        
        # Create x values (days)
        x = np.arange(len(y))
        
        # Fit linear regression: y = a + b*x
        # Using least squares: b = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return slope
    
    @staticmethod
    def calculate_volatility(returns: pd.Series) -> float:
        """
        Calculate volatility as standard deviation of returns.
        
        Args:
            returns: Series of returns
            
        Returns:
            Volatility (standard deviation)
        """
        return returns.std()
    
    @staticmethod
    def calculate_trend_strength(
        price_series: pd.Series,
        use_log: bool = False
    ) -> float:
        """
        Calculate trend strength as |slope| / std.
        
        Args:
            price_series: Series of prices
            use_log: If True, use log prices for both slope and std
            
        Returns:
            Trend strength value
        """
        slope = StockDataProcessor.calculate_slope(price_series, use_log=use_log)
        
        if use_log:
            price_series = price_series[price_series > 0]
            std = np.log(price_series).std()
        else:
            std = price_series.std()
        
        if std == 0:
            return float('inf') if abs(slope) > 0 else 0.0
        
        return abs(slope) / std


###########################################################################################
###########################################################################################
if __name__ == "__main__":
    print("Finance module sandbox started")
    print("Creating StockDataFetcher object")
    #sdf = StockDataFetcher(ticker="AAPL",start_date="2025-12-01", end_date="2026-02-27")
    #sdf = StockDataFetcher(ticker="AAPL",period='1mo', use_adjusted=False)
    sdf = StockDataFetcher(ticker="AAPL")  # Adjusted price to be fetched by default.
    print("Fetching data")
    data = sdf.get_30workdays_frame(end_date="2026-03-06", price_type='Close')
    #data = sdf.get_30workdays_frame(start_date="2025-02-01", price_type='Close')
    #data = sdf.get_30_day_frame(start_date="2026-01-01", price_type='Close')
    #data = sdf.fetch_data()
    #data = sdf.get_ohlcv_data(use_adjusted=False)
    print("Data fetched")
    print(data)
    print(f"days fetched:{len(data)}")
    print("Finance module sandbox ended")