# data/sample.py
class SampleTickers:
    def __init__(self):
        # TOP 100 HSI Companies
        self.hong_kong_tickers = [
            '0001.HK', '0002.HK', '0003.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK', '0016.HK', '0017.HK', '0019.HK',
            '0027.HK', '0066.HK', '0083.HK', '0101.HK', '0135.HK', '0144.HK', '0175.HK', '0267.HK', '0288.HK', '0291.HK',
            '0316.HK', '0322.HK', '0358.HK', '0386.HK', '0388.HK', '0390.HK', '0494.HK', '0669.HK', '0688.HK', '0700.HK',
            '0762.HK', '0823.HK', '0836.HK', '0857.HK', '0883.HK', '0939.HK', '0941.HK', '0992.HK', '1038.HK', '1044.HK',
            '1088.HK', '1093.HK', '1109.HK', '1113.HK', '1177.HK', '1199.HK', '1211.HK', '1288.HK', '1299.HK', '1398.HK',
            '1810.HK', '1876.HK', '1928.HK', '1997.HK', '2007.HK', '2018.HK', '2020.HK', '2269.HK', '2313.HK', '2318.HK',
            '2319.HK', '2328.HK', '2382.HK', '2388.HK', '2628.HK', '2688.HK', '2689.HK', '2799.HK', '2800.HK', '2822.HK',
            '2823.HK', '2828.HK', '3328.HK', '3333.HK', '3690.HK', '3988.HK', '6098.HK', '6862.HK', '9618.HK', '9633.HK',
            '9888.HK', '9898.HK', '9988.HK', '9999.HK', '0004.HK', '0008.HK', '0010.HK', '0023.HK', '0025.HK', '0027.HK',
            '0066.HK', '0088.HK', '0099.HK', '0100.HK', '0129.HK', '0151.HK', '0177.HK', '0207.HK', '0215.HK', '0222.HK'
        ][:100]

        # TOP 100 KOSPI Companies
        self.south_korea_tickers = [
            '005930.KS', '000660.KS', '051910.KS', '035420.KS', '005380.KS', '005490.KS', '012330.KS', '000270.KS', '068270.KS', '096770.KS',
            '017670.KS', '066570.KS', '015760.KS', '105560.KS', '034730.KS', '032830.KS', '018260.KS', '003550.KS', '003670.KS', '000810.KS',
            '000720.KS', '009150.KS', '006400.KS', '000100.KS', '009240.KS', '010140.KS', '000880.KS', '001040.KS', '002210.KS', '003410.KS',
            '003450.KS', '004020.KS', '004370.KS', '005300.KS', '006280.KS', '007070.KS', '007310.KS', '008060.KS', '008350.KS', '010050.KS',
            '010130.KS', '010950.KS', '011000.KS', '011070.KS', '011170.KS', '011780.KS', '011790.KS', '011810.KS', '012450.KS', '014680.KS',
            '014820.KS', '014830.KS', '015540.KS', '016360.KS', '017040.KS', '017180.KS', '017390.KS', '018250.KS', '018880.KS', '019170.KS',
            '020000.KS', '020150.KS', '021240.KS', '023150.KS', '023590.KS', '024110.KS', '025540.KS', '025860.KS', '026890.KS', '027740.KS',
            '027970.KS', '028050.KS', '028260.KS', '028670.KS', '029780.KS', '030000.KS', '030200.KS', '030210.KS', '032640.KS', '033780.KS',
            '034020.KS', '034120.KS', '034220.KS', '034300.KS', '034310.KS', '034590.KS', '035000.KS', '035250.KS', '035510.KS', '036460.KS',
            '036570.KS', '036580.KS', '037270.KS', '037560.KS', '038500.KS', '039130.KS', '039490.KS', '041650.KS', '042660.KS', '044380.KS'
        ][:100]

        # Top 100 S&P500 Companies
        self.us_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 'UNH',
            'MA', 'HD', 'DIS', 'BAC', 'PYPL', 'CMCSA', 'ADBE', 'NFLX', 'XOM', 'VZ',
            'CRM', 'INTC', 'ABT', 'PFE', 'CSCO', 'T', 'CVX', 'WMT', 'MRK', 'PEP',
            'KO', 'TMO', 'ABBV', 'NKE', 'ACN', 'AVGO', 'MCD', 'MDT', 'NEE', 'TXN',
            'LIN', 'UNP', 'HON', 'BMY', 'QCOM', 'ORCL', 'PM', 'AMGN', 'DHR', 'UPS',
            'IBM', 'LLY', 'BA', 'LOW', 'SBUX', 'MMM', 'COST', 'FIS', 'AMT', 'CAT',
            'GS', 'AXP', 'BLK', 'CHTR', 'BKNG', 'ISRG', 'GILD', 'TGT', 'MDLZ', 'SPGI',
            'MO', 'C', 'ZTS', 'ADP', 'INTU', 'TJX', 'VRTX', 'SYK', 'ANTM', 'CCI',
            'ATVI', 'PLD', 'CSX', 'USB', 'CI', 'CME', 'DUK', 'SO', 'MS', 'BDX',
            'GE', 'EQIX', 'CL', 'ILMN', 'TMUS', 'REGN', 'APD', 'ADSK', 'AMD', 'FDX'
        ][:100]

    def hk_samples(self):
        return self.hong_kong_tickers

    def us_samples(self):
        return self.us_tickers

    def sk_samples(self):
        return self.south_korea_tickers
