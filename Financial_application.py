
# Libraries
import numpy as np                        
import pandas as pd                       
import matplotlib.pyplot as plt     
import seaborn as sns      
import plotly.express as px               
import plotly.graph_objects as go         
from datetime import datetime, timedelta  
import streamlit as st                    
import yfinance as yf
import streamlit.components.v1 as components
import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "majorHoldersBreakdown,"
                         "indexTrend,"
                         "defaultKeyStatistics,"
                         "majorHoldersBreakdown,"
                         "insiderHolders")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret



st.set_page_config(page_title="S&P 500 Financial Dashboard ",page_icon=" 📈",layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: white;'>S&P 500 Financial Dashboard 📈 </h1>
    <p style='text-align: center; color: white; font-size: 15px;'>DATA SOURCE: <a href='https://finance.yahoo.com' target='_blank'>Yahoo Finance</a></p>
""", unsafe_allow_html=True)  


url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
# Fetch the table of S&P 500 companies
try:
    sp500_table = pd.read_html(url, header=0)[0]  # Load the first table on the page

    # Extract the Symbol column, which contains the tickers
    sp500_stocks = sp500_table['Symbol'].tolist()

except Exception as e:
    print(f"An error occurred while fetching the data: {e}")

stock_symbol = st.sidebar.selectbox("Stock Selection", sp500_stocks)

today = datetime.now().date()
before = today - timedelta(700)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)

# Ensure the end date is after the start date
if start_date > end_date:
    st.error('Error: End date must fall after start date.')

# Update button in the sidebar
if st.sidebar.button("Update Data"):
    # Fetching the stock data from Yahoo Finance
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    # Display the data in the app
    st.write(f"Stock data for {stock_symbol} from {start_date} to {end_date}:")
    st.dataframe(stock_data)  # Show data as a table
    
    # Option to download data as CSV
    st.sidebar.download_button(
        label="Download Data as CSV",
        data=stock_data.to_csv(),
        file_name=f"{stock_symbol}_data.csv",
        mime="text/csv"
    )




tabs = st.tabs(["Summary", "Chart", "Financials","Monte Carlo simulation","Analysis"])

# Content for Tab 1
with tabs[0]:
    
    col1, col2 = st.columns(2) 
    with col1:
        stock = yf.Ticker(stock_symbol)

        # Attempt to get stock info with error handling
        try:
            stock_info = stock.info
        except Exception as e:
            st.write(f"Error fetching stock info: {e}")
            stock_info = {}

        company_name = stock_info.get('longName', 'N/A')
        symbol = stock_info.get('symbol', stock_symbol)  # Use input symbol if none available
        description = stock_info.get('longBusinessSummary', 'N/A')
        

        st.subheader(f"{company_name} ({symbol}) Overview")

        short_description = ' '.join(description.split(' ')[:30])
        show_more_text = f"{short_description}..."

        # Check if the "show_more" state exists in the session state
        if 'show_more' not in st.session_state:
            st.session_state['show_more'] = False

        # Display the truncated or full description based on the checkbox
        if st.session_state['show_more']:
            st.write(description)
        else:
            st.write(show_more_text)

        # Add a checkbox to control the "show more" state
        st.checkbox("Show more", value=st.session_state['show_more'], key="show_more")


        # Split stock details into two parts for a horizontal display
        stock_details_1 = {
            "Previous Close": f"${stock_info.get('previousClose', 'N/A'):,.2f}" if stock_info.get('previousClose') else "N/A",  # in USD
            "Open": f"${stock_info.get('open', 'N/A'):,.2f}" if stock_info.get('open') else "N/A",  # in USD
            "Bid": f"${stock_info.get('bid', 'N/A'):,.2f}" if stock_info.get('bid') else "N/A",  # in USD
            "Ask": f"${stock_info.get('ask', 'N/A'):,.2f}" if stock_info.get('ask') else "N/A",  # in USD
            "Day's Range": f"${stock_info.get('dayLow', 'N/A'):,.2f} - ${stock_info.get('dayHigh', 'N/A'):,.2f}" if stock_info.get('dayLow') and stock_info.get('dayHigh') else "N/A",  # in USD
            "52 Week Range": f"${stock_info.get('fiftyTwoWeekLow', 'N/A'):,.2f} - ${stock_info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}" if stock_info.get('fiftyTwoWeekLow') and stock_info.get('fiftyTwoWeekHigh') else "N/A",  # in USD
            "Volume": f"{stock_info.get('volume', 'N/A'):,.0f}" if stock_info.get('volume') else "N/A",  # shares
            "Avg. Volume": f"{stock_info.get('averageVolume', 'N/A'):,.0f}" if stock_info.get('averageVolume') else "N/A",  # shares
            "Market Cap": f"${stock_info.get('marketCap', 'N/A'):,.0f}" if stock_info.get('marketCap') else "N/A",  # in USD
            "Gross Margin": f"{stock_info.get('grossMargins', 'N/A') * 100:,.2f}%" if stock_info.get('grossMargins') else "N/A",  # in percentage
            "Operating Margin": f"{stock_info.get('operatingMargins', 'N/A') * 100:,.2f}%" if stock_info.get('operatingMargins') else "N/A",  # in percentage
        }

        def format_date(date_value):
            if date_value and isinstance(date_value, list) and len(date_value) > 0:
                return datetime.fromtimestamp(date_value[0] / 1000).strftime('%Y-%m-%d')
            return "N/A"

        stock_details_2 = {
            "Beta (5Y Monthly)": f"{stock_info.get('beta', 'N/A'):.2f}" if stock_info.get('beta') else "N/A",  # unitless
            "PE Ratio (TTM)": f"{stock_info.get('trailingPE', 'N/A'):.2f}" if stock_info.get('trailingPE') else "N/A",  # unitless
            "EPS (TTM)": f"${stock_info.get('trailingEps', 'N/A'):,.2f}" if stock_info.get('trailingEps') else "N/A",  # in USD
            "Dividend Yield": f"{stock_info.get('dividendYield', 'N/A') * 100:,.2f}%" if stock_info.get('dividendYield') else "N/A",  # in percentage
            "Earnings Date": format_date(stock_info.get('earningsDate', 'N/A')),  # formatted date
            "1y Target Est": f"${stock_info.get('targetMeanPrice', 'N/A'):,.2f}" if stock_info.get('targetMeanPrice') else "N/A",  # in USD
            "Return on Equity (ROE)": f"{stock_info.get('returnOnEquity', 'N/A') * 100:,.2f}%" if stock_info.get('returnOnEquity') else "N/A",  # in percentage
            "Debt to Equity": f"{stock_info.get('debtToEquity', 'N/A'):,.2f}" if stock_info.get('debtToEquity') else "N/A",  # unitless
            "Current Ratio": f"{stock_info.get('currentRatio', 'N/A'):,.2f}" if stock_info.get('currentRatio') else "N/A",  # unitless
            "Forward PE": f"{stock_info.get('forwardPE', 'N/A'):,.2f}" if stock_info.get('forwardPE') else "N/A",  # unitless
            "Price to Book": f"{stock_info.get('priceToBook', 'N/A'):,.2f}" if stock_info.get('priceToBook') else "N/A",  # unitless
        }
        # Create a two-column display for stock details
        col1_stock, col2_stock = st.columns(2)

        with col1_stock:
            for label, value in stock_details_1.items():
                st.markdown(
                    f"<div style='border-bottom: 1px solid gray; padding: 5px; font-size: 15px; display: flex; justify-content: space-between;'>"
                    f"<span style='text-align: left;'>{label}:</span> "
                    f"<span style='text-align: right; font-weight: bold;'>{value}</span></div>",
                    unsafe_allow_html=True
                )

        with col2_stock:
            for label, value in stock_details_2.items():
                st.markdown(
                    f"<div style='border-bottom: 1px solid gray; padding: 5px; font-size: 15px; display: flex; justify-content: space-between;'>"
                    f"<span style='text-align: left;'>{label}:</span> "
                    f"<span style='text-align: right;font-weight: bold;'>{value}</span></div>",
                    unsafe_allow_html=True
                )


    with col2:
        st.markdown("  ")
        st.markdown("  ")
        st.markdown("  ")
        st.markdown("  ")
        # Fetch historical data
        stock = yf.Ticker(stock_symbol)
        historical_data = stock.history(period='5y')

        # Create the trace for the stock's close price with a filled area under the line
        trace = go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            line=dict(color='green', width=2),
            fill='tozeroy',  # Fill the area below the line
            fillcolor='rgba(0,110,0, 0.7)',  # Adjust the color and transparency for the fill
            line_color='green',
            marker=dict(color=historical_data['Close'], colorscale='greens', showscale=True)
        )

        # Define the layout with time range selection
        layout = go.Layout(
            title=f'{stock_symbol} Stock Price',
            width=1500,
            height=600,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="MAX")
                    ])
                ),
                rangeslider=dict(visible=False),
                type="date"
            ),
            yaxis=dict(title='Close Price (USD)', side='right',showgrid=False)
        )

        # Create the figure with the trace and layout
        fig = go.Figure(data=[trace], layout=layout)

        # Display the figure using Streamlit
        st.plotly_chart(fig)
    
    st.markdown("---")
    stock = yf.Ticker(stock_symbol)

    # Major Holders
    st.subheader("Major Holders") 
    try:
        major_holders = stock.major_holders
        # Display Major Holders as a table
        st.table(major_holders)
    except :
        st.write("Major holders data not available.")


# Top Institutional Holders
    st.subheader("Top Institutional Holders")
    try:
        institutional_holders = stock.institutional_holders

        # Format columns with commas for better readability
        institutional_holders['Shares'] = institutional_holders['Shares'].apply(lambda x: f"{x:,}")
        institutional_holders['pctHeld'] = institutional_holders['pctHeld'].apply(lambda x: f"{x * 100:.2f}%")
        institutional_holders['Value'] = institutional_holders['Value'].apply(lambda x: f"${x:,}")

        # Format "Date Reported" to show only the date
        institutional_holders['Date Reported'] = pd.to_datetime(institutional_holders['Date Reported']).dt.date

        # Reorder columns to place "Date Reported" as the third column
        columns_order = ["Holder", "Shares", "Date Reported", "Value", "pctHeld"]
        institutional_holders = institutional_holders[columns_order]

        # Reset index and set a new custom index from 1 to 10
        institutional_holders = institutional_holders.reset_index(drop=True)
        institutional_holders.index = range(1, len(institutional_holders) + 1)

        # Display as a table without the original index
        st.table(institutional_holders)
    except:
        st.write("Institutional holders data not available.")






# Content for Tab 2
with tabs[1]:
    # Fetch stock data (example using Apple stock with a period of 5 years)
    stock = yf.Ticker(stock_symbol).history(period='5y')

    # Calculate the 50-day Moving Average (MA)
    stock['MA50'] = stock['Close'].rolling(window=50).mean()

    # Create dropdown options for plot types and time intervals
    plot_type = 'Line'  # Default plot type
    interval = 'Day'    # Default time interval

    # Create the price plot (line or candlestick based on user selection)
    if plot_type == 'Line':
        price = go.Scatter(x=stock.index, y=stock['Close'], mode='lines', name='Price', yaxis='y1')
    elif plot_type == 'Candlestick':
        price = go.Candlestick(
            x=stock.index,
            open=stock['Open'],
            high=stock['High'],
            low=stock['Low'],
            close=stock['Close'],
            name='Candlestick',
            yaxis='y1'
        )

    # Moving Average plot
    ma50 = go.Scatter(x=stock.index, y=stock['MA50'], mode='lines', name='MA 50', line=dict(color='orange'))

    # Create the volume bar chart with colors based on price movement
    colors = np.where(stock['Close'].pct_change() > 0, 'green', 'red')
    volume = go.Bar(x=stock.index, y=stock['Volume'], name='Volume', yaxis='y2', marker=dict(color=colors))

    # Define the layout with time range selection and dual y-axes
    layout = go.Layout(
        title=f"{company_name}({symbol}) Price and Volume",   
        width=1500,  # Set the width of the chart
        height=600,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="MAX")
                ])
            ),
            rangeslider=dict(visible=True),  # Add a range slider for custom date selection
            type="date"
        ),
        yaxis=dict(title='Close Price (USD)', side='right'),  # y-axis for price
        yaxis2=dict(title='Volume', overlaying='y', side='left', showgrid=False),  # y-axis for volume
    )

    # Create the figure with all components
    fig = go.Figure(data=[price, ma50, volume], layout=layout)

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)



# Content for Tab 3
with tabs[2]:

    stock = yf.Ticker(stock_symbol)
    
    # Create columns for the main and sub-tabs to align them in a single horizontal level
    main_col, sub_col = st.columns([1, 3])

    # Main Tabs (Income Statement, Balance Sheet, Cash Flow)
    with main_col:
        main_tab = st.radio("Select Financial Statement:", ["Income Statement", "Balance Sheet", "Cash Flow"], index=0)

    # Sub Tabs (Annual, Quarterly)
    with sub_col:
        sub_tab = st.radio("Select Period:", ["Annual", "Quarterly"], index=0)


    # Show the selected financial statement and period in the central data table
    if main_tab == "Income Statement":
        if sub_tab == "Annual":
            annual_financials = stock.financials
            st.dataframe(annual_financials, use_container_width=True)
        else:
            quarterly_financials = stock.quarterly_financials
            st.dataframe(quarterly_financials, use_container_width=True)

    elif main_tab == "Balance Sheet":
        if sub_tab == "Annual":
            annual_balance_sheet = stock.balance_sheet
            st.dataframe(annual_balance_sheet, use_container_width=True)
        else:
            quarterly_balance_sheet = stock.quarterly_balance_sheet
            st.dataframe(quarterly_balance_sheet, use_container_width=True)

    elif main_tab == "Cash Flow":
        if sub_tab == "Annual":
            annual_cash_flow = stock.cashflow
            st.dataframe(annual_cash_flow, use_container_width=True)
        else:
            quarterly_cash_flow = stock.quarterly_cashflow
            st.dataframe(quarterly_cash_flow, use_container_width=True)




# Content for Tab 4
with tabs[3]:  # This refers to the Monte Carlo Simulation tab
    stock = yf.Ticker(stock_symbol)
    stock_price = stock.history(period='5y')
    close_price = stock_price['Close']
    
    col1, col2 = st.columns(2)  # Create two columns for side-by-side layout
    with col1:
        num_simulations = st.radio("Number of Simulations", [200, 500, 1000], key="num_simulations")
    with col2:
        time_horizon = st.radio("Time Horizon (days)", [30, 60, 90], key="time_horizon")
    
    # Calculate daily returns and standard deviation
    daily_returns = close_price.pct_change().dropna()
    mean_return = daily_returns.mean()
    daily_volatility = daily_returns.std()

    # Starting price (last closing price)
    last_price = close_price.iloc[-1]

    # Set up simulation parameters
    np.random.seed(123)
    simulations = num_simulations
    time_horizone = time_horizon

    # Run the Monte Carlo simulation
    simulation_df = pd.DataFrame()

    for i in range(simulations):
        # List to store each future price
        next_price = []
        last_price = close_price.iloc[-1]  # Reset last_price for each simulation

        for j in range(time_horizone):
            # Generate the random percentage change
            future_return = np.random.normal(0, daily_volatility)
            # Calculate the future price
            future_price = last_price * (1 + future_return)
            # Save the price
            next_price.append(future_price)
            last_price = future_price
        
        # Store the result of this simulation in the dataframe
        next_price_df = pd.Series(next_price, name=f'Sim {i+1}')
        simulation_df = pd.concat([simulation_df, next_price_df], axis=1)

    # Plot the simulation stock price paths
    fig, ax = plt.subplots(figsize=(11, 6))
    # Plot each simulation path
    ax.plot(simulation_df, linewidth=1, alpha=0.7)
    # Plot the last known price as a reference line
    ax.axhline(y=close_price.iloc[-1], color='red', linestyle='-')

    x_ticks = np.arange(0, time_horizon + 1, 5)
    ax.set_xticks(x_ticks)

    fig.patch.set_facecolor((0.451, 0.576, 0.702)) 
    ax.set_facecolor((0.451, 0.576, 0.702)) 

    # Customize the plot
    ax.set_title(f"Monte Carlo Simulation of {stock_symbol} Stock Price over {time_horizon} Days")
    ax.set_xlabel("Days")
    ax.set_ylabel("Stock Price")
    # Add legend with current stock price
    ax.legend([f"Current Price: ${close_price.iloc[-1]:.2f}"], loc="upper right", framealpha=0.7)
    ax.get_legend().legend_handles[0].set_color('red')

    # Display plot in Streamlit
    st.pyplot(fig)
    
    ending_price = simulation_df.iloc[-1:, :].values[0, ]
    future_price_95ci = np.percentile(ending_price, 5)  # 5th percentile for 95% VaR
    VaR = close_price.iloc[-1] - future_price_95ci
    st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')

    
# Content for Tab 5
with tabs[4]: 

    # Default selected stock symbols
    default_symbols = ["AAPL", "TSLA"]

    # User selects stock symbols from the SP500 list
    options = st.multiselect("Select Stock Symbols", sp500_stocks, default=default_symbols)
    stock_symbols_list = [symbol.strip().upper() for symbol in options]

    # Fetch stock data for the selected symbols
    st.subheader("Stock Price Comparison")
    stock_data = {}
    for symbol in stock_symbols_list:
        stock_data[symbol] = yf.Ticker(symbol).history(period='5y')['Close']

    # Combine stock prices into a DataFrame for plotting
    stock_prices_df = pd.DataFrame(stock_data)
    st.line_chart(stock_prices_df)

    # Check if at least two stocks are selected
    if len(stock_symbols_list) < 2:
        st.warning("Please select at least two stocks for side-by-side comparison.")
    else:
        # Fetch detailed financial metrics for each selected stock
        metrics_data = {}
        for symbol in stock_symbols_list:
            stock = yf.Ticker(symbol)
            info = stock.info
            metrics_data[symbol] = {
                "Market Cap (USD)": info.get('marketCap', 'N/A'),
                "P/E Ratio": info.get('trailingPE', 'N/A'),
                "Dividend Yield (%)": info.get('dividendYield', 'N/A') * 100 if info.get('dividendYield') else 'N/A',
                "Debt-to-Equity Ratio": info.get('debtToEquity', 'N/A'),
                "Revenue (TTM) (USD)": info.get('totalRevenue', 'N/A'),
                "Net Income (TTM) (USD)": info.get('netIncomeToCommon', 'N/A'),
                "EPS": info.get('trailingEps', 'N/A'),
                "52-Week High (USD)": info.get('fiftyTwoWeekHigh', 'N/A'),
                "52-Week Low (USD)": info.get('fiftyTwoWeekLow', 'N/A'),
                "5-Year Monthly Beta": info.get('beta', 'N/A')
            }

        # Convert to a DataFrame for raw data
        metrics_df = pd.DataFrame(metrics_data)

        # Create a copy for display formatting
        display_df = metrics_df.copy()

        # Format each column in display_df and highlight larger values
        for index, row in display_df.iterrows():
            # Identify the max value for each row (numeric values only)
            numeric_values = [value for value in row if isinstance(value, (int, float))]
            col_max = max(numeric_values) if numeric_values else None

            # Format and bold the larger value
            for col in display_df.columns:
                value = row[col]
                if value == col_max and pd.notna(value):
                    display_df.at[index, col] = f'<span style="color:green; font-weight:bold;">{value:,.2f}</span>' if isinstance(value, (int, float)) else f'<span style="color:green; font-weight:bold;">{value}</span>'
                else:
                    display_df.at[index, col] = f"{value:,.2f}" if isinstance(value, (int, float)) else f"{value}"

        # Display the formatted DataFrame in Streamlit
        st.subheader(f"Financial Comparison: {', '.join(stock_symbols_list)}")
        st.write(display_df.to_markdown(), unsafe_allow_html=True)
