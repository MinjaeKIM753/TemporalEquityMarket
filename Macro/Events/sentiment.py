import pandas as pd
import json
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from scipy.stats import pearsonr
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import os
import glob
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from multiprocessing import Pool
import pickle
from statsmodels.tsa.stattools import grangercausalitytests
import time
from sklearn.linear_model import LinearRegression
import warnings

finbert_tokenizer = None
finbert_model = None
roberta_tokenizer = None
roberta_model = None

def initialize_models(device='cuda' if torch.cuda.is_available() else 'cpu'):
    global finbert_tokenizer, finbert_model, roberta_tokenizer, roberta_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if finbert_tokenizer is None:
        finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        finbert_model.to(device)
        finbert_model.eval()
    
    if roberta_tokenizer is None:
        roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        roberta_model.to(device)
        roberta_model.eval()

def process_batch_finbert(batch_texts, device):
    with torch.no_grad():
        inputs = finbert_tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        ).to(device)
        outputs = finbert_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        scores = (probabilities[:, 2] - probabilities[:, 0]).float()
        return scores.cpu().numpy()

def process_batch_roberta(batch_texts, device):
    with torch.no_grad():
        inputs = roberta_tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        ).to(device)
        outputs = roberta_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        scores = (probabilities[:, 2] - probabilities[:, 0]).float()
        return scores.cpu().numpy()

def clean_description(desc):
    patterns = [
        r'^On [A-Z][a-z]+ \d{1,2},? \d{4},?\s+',
        r'^On [A-Z][a-z]+ \d{1,2},?\s+',
        r'^On \d{4}-\d{2}-\d{2},?\s+',
        r'^In [A-Z][a-z]+ \d{4},?\s+',
        r'^In early [A-Z][a-z]+ \d{4},?\s+',
        r'^In late [A-Z][a-z]+ \d{4},?\s+',
        r'^In mid-[A-Z][a-z]+ \d{4},?\s+',
        r'^During [A-Z][a-z]+ \d{4},?\s+',
        r'^As of [A-Z][a-z]+ \d{4},?\s+'
    ]
    for pattern in patterns:
        desc = re.sub(pattern, '', desc)
    return desc

def fix_json_string(content_str):
    content_str = content_str.strip()
    if 'Note:' in content_str:
        content_str = content_str.split('Note:')[0]
    
    content_str = content_str.replace('\\"', '"')
    content_str = content_str.replace('\\n', ' ')
    content_str = content_str.replace('\\r', ' ')
    content_str = content_str.replace('\t', ' ')
    
    content_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content_str)
    content_str = re.sub(r'\s+', ' ', content_str)
    
    valid_objects = []
    current_obj = ""
    brace_count = 0
    in_string = False
    escape = False
    
    for char in content_str:
        if char == '\\':
            escape = not escape
            current_obj += char
            continue
        
        if not escape and char == '"':
            in_string = not in_string
            current_obj += char
            
        elif not in_string:
            if char == '{':
                if brace_count == 0:
                    current_obj = '{'
                else:
                    current_obj += char
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                current_obj += char
                
                if brace_count == 0:
                    try:
                        obj = current_obj
                        obj = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', obj)
                        
                        if '"Description": "' in obj and not obj.rstrip('}').rstrip().endswith('"'):
                            obj = obj.rsplit('"Description":', 1)[0] + '}'
                            
                        obj = obj.replace(': undefined', ': null')
                        
                        json.loads(obj)
                        valid_objects.append(obj)
                    except:
                        pass
                    current_obj = ""
            else:
                current_obj += char
        else:
            current_obj += char
        
        escape = False
    
    if valid_objects:
        result = '[' + ','.join(valid_objects) + ']'
        try:
            json.loads(result)
            return result
        except:
            if len(valid_objects) > 1:
                return '[' + ','.join(valid_objects[:-1]) + ']'
    
    return '[]'

def parse_jsonl(file_path):
    save_path = os.path.join("Macro/Events", "parsed_events.pkl")
    if os.path.exists(save_path):
        print("Loading saved event data...")
        return pd.read_pickle(save_path)
    
    data = []
    error_lines = []
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            try:
                record = json.loads(line)
                timestamp = record["custom_id"].split("_")[-1]
                content_str = record["response"]["body"]["choices"][0]["message"]["content"]
                
                content_str = fix_json_string(content_str)
                events = json.loads(content_str)
                
                national_events = []
                other_events = []
                
                for event in events:
                    category = event.get("Category")
                    if not category:
                        continue
                    
                    if category == "National":
                        national_events.append(event)
                    else:
                        other_events.append(event)
                
                for idx, event in enumerate(national_events):
                    if idx < 3:
                        country = "Hong Kong"
                    elif idx < 6:
                        country = "South Korea"
                    elif idx < 9:
                        country = "U.S"
                    else:
                        country = "Japan"
                    
                    description = clean_description(event.get("Description", ""))
                    data.append({
                            "timestamp": timestamp,
                        "category": "National",
                        "country": country,
                            "title": event.get("Title", ""),
                        "description": description
                    })
                
                for event in other_events:
                    if event["Category"] == "International":
                        country = "Global"
                    elif event["Category"] == "Social/Technology":
                        country = "Global"
                    else:
                        continue
                    
                    description = clean_description(event.get("Description", ""))
                    data.append({
                        "timestamp": timestamp,
                        "category": event["Category"],
                        "country": country,
                        "title": event.get("Title", ""),
                        "description": description
                        })
                    
            except Exception as e:
                print(f"\nError at line {line_num}: {str(e)}")
                error_lines.append((line_num, line))
                continue
    
    if not data:
        raise ValueError("No valid parsed data")
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df.to_pickle(save_path)
    print(f"\nTotal parsed events: {len(df)}")
    print(f"Total error lines: {len(error_lines)}")

    return df

def classify_unknowns(df):
    df = df.sort_values(by=["timestamp", "country"]).reset_index(drop=True)
    for ts in df["timestamp"].unique():
        unknowns = df[(df["timestamp"] == ts) & (df["country"] == "Unknown")].index
        if len(unknowns) >= 6:
            df.loc[unknowns[:3], "country"] = "Global"
            df.loc[unknowns[-3:], "country"] = "Industrial/Scientific"
        elif len(unknowns) > 0:
            df.loc[unknowns, "country"] = "Global"
    return df

def print_event_counts(df):
    event_counts = df["timestamp"].dt.date.value_counts()
    dates_under_15 = event_counts[event_counts < 15]
    
    if not dates_under_15.empty:
        print("\n=== Dates with fewer than 15 events ===")
        for date, count in dates_under_15.items():
            print(f"{date}: {count} events")
        print("\nDates that need reproduction:")
        date_list = [date.strftime("%Y-%m-%d") for date in dates_under_15.index]
        print(sorted(date_list))
    
    print(f"\nTotal parsed events: {len(df)}")
    print("\nCategory distribution:")
    print(df["category"].value_counts())
    print("\nCountry distribution:")
    print(df["country"].value_counts())

def analyze_sentiments(df, save_dir="Macro/Events/Saves"):
    print("\nAnalyzing with multiple sentiment analysis tools...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    initialize_models(device)
    
    tqdm.pandas()
    
    sentiment_tools = {
        "finbert": {"file": "finbert_sentiment.pkl", "weight": 0.6},
        "roberta": {"file": "roberta_sentiment.pkl", "weight": 0.4}
    }
    batch_size = 128
    
    finbert_path = os.path.join(save_dir, sentiment_tools["finbert"]["file"])
    if not CFG.force_update and os.path.exists(finbert_path):
        print("Loading saved FinBERT results...")
        df["finbert_sentiment"] = pd.read_pickle(finbert_path)
    else:
        print("Analyzing with FinBERT...")
        descriptions = df["description"].tolist()
        sentiments = []
        
        for i in tqdm(range(0, len(descriptions), batch_size)):
            batch_texts = descriptions[i:i + batch_size]
            batch_sentiments = process_batch_finbert(batch_texts, device)
            sentiments.extend(batch_sentiments)
            
            if i % 1000 == 0 and i > 0:
                temp_df = pd.Series(sentiments, name="finbert_sentiment")
                temp_df.to_pickle(finbert_path + f".temp_{i}")
        
        df["finbert_sentiment"] = sentiments
        df["finbert_sentiment"].to_pickle(finbert_path)
        
        for temp_file in glob.glob(finbert_path + ".temp_*"):
            os.remove(temp_file)
    
    roberta_path = os.path.join(save_dir, sentiment_tools["roberta"]["file"])
    if not CFG.force_update and os.path.exists(roberta_path):
        print("Loading saved RoBERTa results...")
        df["roberta_sentiment"] = pd.read_pickle(roberta_path)
    else:
        print("Analyzing with RoBERTa...")
        descriptions = df["description"].tolist()
        sentiments = []
        
        for i in tqdm(range(0, len(descriptions), batch_size)):
            batch_texts = descriptions[i:i + batch_size]
            batch_sentiments = process_batch_roberta(batch_texts, device)
            sentiments.extend(batch_sentiments)
            
            if i % 1000 == 0 and i > 0:
                temp_df = pd.Series(sentiments, name="roberta_sentiment")
                temp_df.to_pickle(roberta_path + f".temp_{i}")
        
        df["roberta_sentiment"] = sentiments
        df["roberta_sentiment"].to_pickle(roberta_path)
        
        for temp_file in glob.glob(roberta_path + ".temp_*"):
            os.remove(temp_file)
    
    weights = {k: v["weight"] for k, v in sentiment_tools.items()}
    df["ensemble_sentiment"] = calculate_ensemble_sentiment(df, weights)
    
    return df

def download_stock_data(df):
    market_map = {
        "Hong Kong": "^HSI",
        "South Korea": "^KS11",
                "U.S": "^GSPC",
        "Japan": "^N225",
        "Global": ["^HSI", "^KS11", "^GSPC", "^N225"],
        "Industrial/Scientific": ["^HSI", "^KS11", "^GSPC", "^N225"]
    }
    df["stock_index"] = df["country"].map(market_map)
    print("\n=== Country to Market Index Mapping ===")
    print(df.groupby("country")["stock_index"].first())
    
    df = df.dropna(subset=["stock_index"])
    df = df.explode("stock_index")
    
    stocks = {}
    start_date = df["timestamp"].min() - pd.Timedelta(days=5)
    end_date = df["timestamp"].max() + pd.Timedelta(days=5)
    
    print(f"\nDownloading stock data... ({start_date.date()} ~ {end_date.date()})")
    
    for ticker in df["stock_index"].unique():
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                stocks[ticker] = data["Close"]
                print(f"{ticker}: Downloaded {len(data)} data points")
            else:
                print(f"{ticker}: No data available")
        except Exception as e:
            print(f"Error downloading {ticker}: {str(e)}")
    
    if stocks:
        df_returns = pd.concat(stocks.values(), axis=1, keys=stocks.keys())
        df_returns.columns = df_returns.columns.get_level_values(0)
        df_returns = df_returns.pct_change(fill_method=None)
        
        df = df.set_index("timestamp")
        df["market_returns"] = np.nan
        
        for country, ticker in market_map.items():
            if isinstance(ticker, list):
                mask = df["country"] == country
                df.loc[mask, "market_returns"] = df_returns[ticker].mean(axis=1)
            else:
                mask = df["country"] == country
                df.loc[mask, "market_returns"] = df_returns[ticker]
        
        return df.reset_index()
    
    return df

def extract_nouns(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns = [word for word, tag in tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']]
    return ' '.join(nouns)

def process_chunk(texts):
    return [extract_nouns(text) for text in texts]

def process_textblob_chunk(chunk):
    results = {}
    for text in chunk:
        blob = TextBlob(text)
        for sentence in blob.sentences:
            sentiment = sentence.sentiment.polarity
            tagged_words = pos_tag(word_tokenize(str(sentence)))
            for word, tag in tagged_words:
                if tag in ['NN', 'NNS']:
                    word = word.lower()
                    if word not in results:
                        results[word] = {'textblob': [], 'type': 'common'}
                    results[word]['textblob'].append(sentiment)
                elif tag in ['NNP', 'NNPS']:
                    if word not in results:
                        results[word] = {'textblob': [], 'type': 'proper'}
                    results[word]['textblob'].append(sentiment)
    return results

def process_vader_chunk(chunk):
    results = {}
    analyzer = SentimentIntensityAnalyzer()
    for text in chunk:
        sentences = sent_tokenize(text)
        for sentence in sentences:
            sentiment = analyzer.polarity_scores(sentence)['compound']
            tagged_words = pos_tag(word_tokenize(sentence))
            for word, tag in tagged_words:
                if tag in ['NN', 'NNS']:
                    word = word.lower()
                    if word not in results:
                        results[word] = {'vader': [], 'type': 'common'}
                    results[word]['vader'].append(sentiment)
                elif tag in ['NNP', 'NNPS']:
                    if word not in results:
                        results[word] = {'vader': [], 'type': 'proper'}
                    results[word]['vader'].append(sentiment)
    return results

def process_finbert_chunk(chunk, device):
    results = {}
    for text in chunk:
        with torch.no_grad():
            inputs = finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = finbert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            sentiment = (probs[0, 2] - probs[0, 0]).item()
            
            sentences = sent_tokenize(text)
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                tagged_words = pos_tag(words)
                
                for word, tag in tagged_words:
                    is_proper = (
                        tag in ['NNP', 'NNPS'] and
                        word[0].isupper() and
                        len(word) > 1 and
                        not any(c.isdigit() for c in word)
                    )
                    
                    if tag in ['NN', 'NNS'] or not is_proper:
                        word = word.lower()
                        if word not in results:
                            results[word] = {'finbert': [], 'type': 'common'}
                        results[word]['finbert'].append(sentiment)
                    else:
                        if word not in results:
                            results[word] = {'finbert': [], 'type': 'proper'}
                        results[word]['finbert'].append(sentiment)
    return results

def process_roberta_chunk(chunk, device):
    results = {}
    for text in chunk:
        with torch.no_grad():
            inputs = roberta_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = roberta_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            sentiment = (probs[0, 2] - probs[0, 0]).item()
            tagged_words = pos_tag(word_tokenize(text))
            for word, tag in tagged_words:
                if tag in ['NN', 'NNS']:
                    word = word.lower()
                    if word not in results:
                        results[word] = {'roberta': [], 'type': 'common'}
                    results[word]['roberta'].append(sentiment)
                elif tag in ['NNP', 'NNPS']:
                    if word not in results:
                        results[word] = {'roberta': [], 'type': 'proper'}
                    results[word]['roberta'].append(sentiment)
    return results

def process_finbert_with_device(chunk):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return process_finbert_chunk(chunk, device)

def process_roberta_with_device(chunk):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return process_roberta_chunk(chunk, device)

def analyze_keywords(df):
    print("\nAnalyzing keyword impact...")

    df = df.copy()
    
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    initialize_models(device)
    
    save_dir = "Macro/Events/Saves"
    os.makedirs(save_dir, exist_ok=True)
    
    countries = df['country'].unique()
    keyword_sentiments = {country: {'common': {}, 'proper': {}} for country in countries}
    
    tools = {
        'finbert': (process_finbert_with_device, "finbert_keywords.pkl"),
        'roberta': (process_roberta_with_device, "roberta_keywords.pkl")
    }
    
    for country in countries:
        print(f"\n=== {country} Keyword Analysis ===")
        country_df = df[df['country'] == country]
        texts = country_df['description'].tolist()
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for tool_name, (process_func, save_file) in tools.items():
            save_path = os.path.join(save_dir, f"{country}_{save_file}")
            temp_files = []
            
            if not CFG.force_update and os.path.exists(save_path):
                print(f"Loading saved {tool_name.upper()} results...")
                with open(save_path, 'rb') as f:
                    results = pickle.load(f)
            else:
                print(f"\nAnalyzing with {tool_name.upper()}...")
                start_time = time.time()
                
                results = {}
                with tqdm(total=total_batches, desc=f"{tool_name.upper()} batch processing") as pbar:
                    for batch_idx in range(total_batches):
                        try:
                            batch_result = process_func(texts[batch_idx * batch_size:min((batch_idx + 1) * batch_size, len(texts))])
                            for noun, data in batch_result.items():
                                if noun not in results:
                                    results[noun] = {'type': data['type']}
                                if tool_name not in results[noun]:
                                    results[noun][tool_name] = []
                                results[noun][tool_name].extend(data[tool_name])
                            
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            if (batch_idx + 1) % 1000 == 0:
                                temp_path = f"{save_path}.temp_{batch_idx}"
                                with open(temp_path, 'wb') as f:
                                    pickle.dump(results, f)
                                temp_files.append(temp_path)
                            
                        except Exception as e:
                            print(f"\nError processing batch {batch_idx + 1}: {str(e)}")
                            continue
                        
                        pbar.update(1)
                
                with open(save_path, 'wb') as f:
                    pickle.dump(results, f)
                
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                
                elapsed_time = time.time() - start_time
                print(f"{tool_name.upper()} analysis completed: {elapsed_time:.2f} seconds")
            
            for noun, data in results.items():
                noun_type = data['type']
                if noun not in keyword_sentiments[country][noun_type]:
                    keyword_sentiments[country][noun_type][noun] = {}
                keyword_sentiments[country][noun_type][noun].update({k: v for k, v in data.items() if k != 'type'})
    
    final_save_path = os.path.join(save_dir, "all_keywords.pkl")
    with open(final_save_path, 'wb') as f:
        pickle.dump(keyword_sentiments, f)
    
    return keyword_sentiments

def visualize_sentiment_analysis(df):
    save_dir = "Macro/Events/Visualizations/Sentiment_Analysis"
    os.makedirs(save_dir, exist_ok=True)
    
    df_viz = df.copy()
    if df_viz.index.name == 'timestamp':
        df_viz = df_viz.reset_index()
    elif 'timestamp' not in df_viz.columns and 'index' in df_viz.columns:
        df_viz = df_viz.rename(columns={'index': 'timestamp'})
    
    sentiment_tools = ["finbert_sentiment", "roberta_sentiment", "ensemble_sentiment"]
    
    plt.figure(figsize=(15, 8))
    for tool in sentiment_tools:
        tool_name = tool.replace('_sentiment', '')
        avg_sentiment = df_viz.groupby('country')[tool].mean()
        plt.bar(avg_sentiment.index, avg_sentiment.values, alpha=0.6, label=tool_name)
    
    plt.title("Average Sentiment by Country and Tool")
    plt.xlabel("Country")
    plt.ylabel("Average Sentiment")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'average_sentiment_by_country.png'))
    plt.close()
    
    for tool in sentiment_tools:
        tool_name = tool.replace('_sentiment', '')
        for country in df_viz['country'].unique():
            if country == 'Global':
                continue
                
            market_data = df_viz[df_viz['country'] == country]
            
            print(f"market_data: {market_data}")
            print(f"tool: {tool}")
            daily_sentiments = market_data.groupby(['timestamp', 'category'])[tool].mean().unstack()
            
            market_returns = market_data.groupby('timestamp')['market_returns'].first()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
            
            for category in ['International', 'National', 'Social/Technology']:
                if category in daily_sentiments.columns:
                    ax1.plot(daily_sentiments.index, daily_sentiments[category],
                            label=f'{category} Sentiment',
                            alpha=0.7)
            
            ax2.plot(market_returns.index, market_returns,
                    label='Market Returns',
                    color='black',
                    alpha=0.7)
            
            correlation_results = []
            for category in ['International', 'National', 'Social/Technology']:
                if category in daily_sentiments.columns:
                    corr, p_value = pearsonr(daily_sentiments[category], market_returns)
                    
                    try:
                        granger_test = grangercausalitytests(
                            pd.concat([daily_sentiments[category], market_returns], axis=1),
                            maxlag=5,
                            verbose=False
                        )
                        granger_p_value = min([granger_test[i+1][0]['ssr_ftest'][1] 
                                            for i in range(5)])
                    except:
                        granger_p_value = None
                    
                    correlation_results.append({
                        'category': category,
                        'correlation': corr,
                        'p_value': p_value,
                        'granger_p_value': granger_p_value
                    })
            
            total_sentiment = daily_sentiments.mean(axis=1)
            total_corr, total_p_value = pearsonr(total_sentiment, market_returns)
            
            try:
                total_granger_test = grangercausalitytests(
                    pd.concat([total_sentiment, market_returns], axis=1),
                    maxlag=5,
                    verbose=False
                )
                total_granger_p_value = min([total_granger_test[i+1][0]['ssr_ftest'][1] 
                                           for i in range(5)])
            except:
                total_granger_p_value = None
            
            correlation_results.append({
                'category': 'Total',
                'correlation': total_corr,
                'p_value': total_p_value,
                'granger_p_value': total_granger_p_value
            })
            
            corr_text = "Correlation Analysis:\n"
            for result in correlation_results:
                corr_text += f"{result['category']}: r={result['correlation']:.2f}, p={result['p_value']:.3f}\n"
                if result['granger_p_value'] is not None:
                    corr_text += f"Granger p={result['granger_p_value']:.3f}\n"
            
            plt.figtext(0.02, 0.02, corr_text, fontsize=8)
            
            plt.suptitle(f'{country} Market: {tool_name} Sentiment vs Market Returns')
            ax1.set_ylabel('Sentiment Score')
            ax2.set_ylabel('Market Returns')
            ax2.set_xlabel('Date')
            
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{country}_{tool_name}_sentiment_vs_returns.png'))
            plt.close()

def calculate_ensemble_sentiment(df, weights):
    return (
        weights['finbert'] * df["finbert_sentiment"] +
        weights['roberta'] * df["roberta_sentiment"]
    )

def analyze_market_sentiments(df):
    save_dir = "Macro/Events/Visualizations/Market_Sentiments"
    os.makedirs(save_dir, exist_ok=True)
    
    sentiment_tools = ['finbert_sentiment', 'roberta_sentiment']
    categories = ['International', 'National', 'Social/Technology']
    market_indices = {
        '^HSI': 'Hong Kong',
        '^KS11': 'South Korea',
        '^GSPC': 'U.S',
        '^N225': 'Japan'
    }
    
    df_viz = df.copy()
    if df_viz.index.name == 'timestamp':
        df_viz = df_viz.reset_index()
    elif 'timestamp' not in df_viz.columns and 'index' in df_viz.columns:
        df_viz = df_viz.rename(columns={'index': 'timestamp'})
    
    correlation_results = []
    
    for index, country in market_indices.items():
        market_data = df_viz[df_viz['stock_index'] == index].copy()
        
        for tool in sentiment_tools:
            try:
                daily_sentiments = market_data.groupby(['timestamp', 'category'])[tool].mean().unstack()
                
                market_returns = market_data.groupby('timestamp')['market_returns'].first()
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
                
                for category in categories:
                    if category in daily_sentiments.columns:
                        ax1.plot(daily_sentiments.index, daily_sentiments[category],
                                label=f'{category} Sentiment',
                                alpha=0.7)
                
                ax2.plot(market_returns.index, market_returns,
                        label='Market Returns',
                        color='black',
                        alpha=0.7)
                
                tool_name = tool.split('_')[0].capitalize()
                plt.suptitle(f'{country} Market: {tool_name} Sentiment vs Market Returns')
                ax1.set_ylabel('Sentiment Score')
                ax2.set_ylabel('Market Returns')
                ax2.set_xlabel('Date')
                
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper left')
                
                plt.tight_layout()
                
                filename = f"{country}_{tool_name}_sentiment.png"
                plt.savefig(os.path.join(save_dir, filename))
                plt.close()
                
                for category in categories:
                    if category in daily_sentiments.columns:
                        corr, p_value = pearsonr(daily_sentiments[category], market_returns)
                        
                        try:
                            granger_test = grangercausalitytests(
                                pd.concat([daily_sentiments[category], market_returns], axis=1),
                                maxlag=5,
                                verbose=False
                            )
                            granger_p_value = min([granger_test[i+1][0]['ssr_ftest'][1] 
                                                for i in range(5)])
                        except:
                            granger_p_value = None
                        
                        correlation_results.append({
                            'country': country,
                            'tool': tool_name,
                            'category': category,
                            'correlation': corr,
                            'p_value': p_value,
                            'granger_p_value': granger_p_value
                        })
                
                total_sentiment = daily_sentiments.mean(axis=1)
                total_corr, total_p_value = pearsonr(total_sentiment, market_returns)
                
                try:
                    total_granger_test = grangercausalitytests(
                        pd.concat([total_sentiment, market_returns], axis=1),
                        maxlag=5,
                        verbose=False
                    )
                    total_granger_p_value = min([total_granger_test[i+1][0]['ssr_ftest'][1] 
                                               for i in range(5)])
                except:
                    total_granger_p_value = None
                
                correlation_results.append({
                    'country': country,
                    'tool': tool_name,
                    'category': 'Total',
                    'correlation': total_corr,
                    'p_value': total_p_value,
                    'granger_p_value': total_granger_p_value
                })
                
            except Exception as e:
                print(f"Error occurred - {country}, {tool}: {str(e)}")
                continue
    
    if correlation_results:
        correlation_df = pd.DataFrame(correlation_results)
        
        for tool in correlation_df['tool'].unique():
            tool_data = correlation_df[correlation_df['tool'] == tool]
            
            pivot_corr = tool_data.pivot(index='country', columns='category', values='correlation')
            pivot_p = tool_data.pivot(index='country', columns='category', values='p_value')
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_corr, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', mask=pivot_p > 0.05)
            plt.title(f'Correlation Analysis - {tool}')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{tool.lower()}_correlation.png'))
            plt.close()
        
        return correlation_df
     
    return None

def analyze_keyword_impacts(df, keyword_sentiments):
    print("\nAnalyzing keyword impacts...")
    
    save_dir = "Macro/Events/Visualizations/Keyword_Impacts"
    os.makedirs(save_dir, exist_ok=True)
    
    tools = ['finbert', 'roberta']
    
    for tool in tools:
        print(f"\nAnalyzing {tool.upper()} keywords...")
        
        for noun_type in ['common', 'proper']:
            print(f"Analyzing {noun_type.capitalize()} nouns...")
            
            total_keywords = len([k for k, v in keyword_sentiments[noun_type].items() if tool in v and len(v[tool]) > 0])
            with tqdm(total=total_keywords, desc=f"{tool.upper()} {noun_type} keyword processing") as pbar:
                tool_keywords = {}
                for keyword, sentiments in keyword_sentiments[noun_type].items():
                    if tool in sentiments and len(sentiments[tool]) > 0:
                        avg_sentiment = np.mean(sentiments[tool])
                        sentiment_strength = abs(avg_sentiment)
                        frequency = len(sentiments[tool])
                        
                        if frequency >= 2:
                            tool_keywords[keyword] = {
                                'sentiment': avg_sentiment,
                                'strength': sentiment_strength,
                                'frequency': frequency
                            }
                    pbar.update(1)
            
            if not tool_keywords:
                print(f"No {noun_type} nouns found for {tool}")
                continue
            
            sorted_keywords = sorted(
                tool_keywords.items(),
                key=lambda x: x[1]['strength'],
                reverse=True
            )
            
            if not sorted_keywords:
                print(f"No sorted keywords for {tool} {noun_type}")
                continue
            
            n_keywords = min(30, len(sorted_keywords))
            top_keywords = sorted_keywords[:n_keywords]
            bottom_keywords = sorted_keywords[-n_keywords:]
            
            plt.figure(figsize=(20, 10))
            
            plt.subplot(1, 2, 1)
            values = [v['strength'] for k, v in top_keywords]
            labels = [k for k, v in top_keywords]
            colors = ['green' if v['sentiment'] > 0 else 'red' for k, v in top_keywords]
            
            plt.bar(range(len(values)), values, color=colors, alpha=0.6)
            plt.xticks(range(len(labels)), labels, rotation=90)
            plt.title(f'Top {n_keywords} Most Impactful {noun_type.capitalize()} Nouns ({tool.capitalize()})')
            
            plt.subplot(1, 2, 2)
            values = [v['strength'] for k, v in bottom_keywords]
            labels = [k for k, v in bottom_keywords]
            colors = ['green' if v['sentiment'] > 0 else 'red' for k, v in bottom_keywords]
            
            plt.bar(range(len(values)), values, color=colors, alpha=0.6)
            plt.xticks(range(len(labels)), labels, rotation=90)
            plt.title(f'Bottom {n_keywords} Least Impactful {noun_type.capitalize()} Nouns ({tool.capitalize()})')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{tool}_{noun_type}_nouns.png'))
            plt.close()
    
    return sorted_keywords

def visualize_top_impact_events(df, keyword_sentiments, top_n=5):
    save_dir = "Macro/Events/Visualizations/Top_Impact_Events"
    os.makedirs(save_dir, exist_ok=True)
    
    tools = ['finbert', 'roberta']
    
    for country in df['country'].unique():
        country_df = df[df['country'] == country]
        country_keywords = keyword_sentiments[country]
        
        event_impacts = []
        for idx, row in country_df.iterrows():
            impact = 0
            for tool in tools:
                for noun_type in ['common', 'proper']:
                    for keyword, data in country_keywords[noun_type].items():
                        if tool in data and keyword in row['description'].lower():
                            impact += abs(np.mean(data[tool]))
            
            event_impacts.append({
                'title': row['title'],
                'description': row['description'],
                'impact': impact,
                'category': row['category']
            })
        
        event_impacts.sort(key=lambda x: x['impact'], reverse=True)
        top_events = event_impacts[:top_n]
        
        plt.figure(figsize=(15, 8))
        events = [f"{event['title']}\n({event['category']})" for event in top_events]
        impacts = [event['impact'] for event in top_events]
        
        plt.bar(events, impacts, color='skyblue')
        plt.title(f'Top {top_n} Most Impactful Events in {country}')
        plt.xlabel('Event Title')
        plt.ylabel('Impact Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, f'{country}_top_impact_events.png'))
        plt.close()
        
        print(f"\n=== Top {top_n} Impactful Events in {country} ===")
        for i, event in enumerate(top_events, 1):
            print(f"\n{i}. {event['title']}")
            print(f"Category: {event['category']}")
            print(f"Impact Score: {event['impact']:.2f}")
            print(f"Description: {event['description']}")
            print("-" * 50)

class CFG: 
    force_update = False
    save_dir = "Macro/Events/Saves"

def analyze_global_events(df):
    print("\n=== Starting Global Events Analysis ===")
    
    save_dir = "Macro/Events/Saves"
    os.makedirs(save_dir, exist_ok=True)
    
    global_df = df[df['country'].isin(['Global', 'Industrial/Scientific'])].copy()
    print(f"Number of Global events: {len(global_df)}")
    
    tools = {
        'finbert': (process_finbert_with_device, "global_finbert_keywords.pkl"),
        'roberta': (process_roberta_with_device, "global_roberta_keywords.pkl")
    }
    
    global_keywords = {'common': {}, 'proper': {}}
    
    for tool_name, (process_func, save_file) in tools.items():
        save_path = os.path.join(save_dir, save_file)
        
        if not CFG.force_update and os.path.exists(save_path):
            print(f"Loading saved {tool_name.upper()} results...")
            with open(save_path, 'rb') as f:
                results = pickle.load(f)
        else:
            print(f"\nAnalyzing {tool_name.upper()} Global events...")
            start_time = time.time()
            
            batch_size = 32
            texts = global_df['description'].tolist()
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            results = {}
            with tqdm(total=total_batches, desc=f"{tool_name.upper()} batch processing") as pbar:
                for batch_idx in range(total_batches):
                    try:
                        batch_texts = texts[batch_idx * batch_size:min((batch_idx + 1) * batch_size, len(texts))]
                        batch_result = process_func(batch_texts)
                        
                        for noun, data in batch_result.items():
                            if noun not in results:
                                results[noun] = {'type': data['type']}
                            if tool_name not in results[noun]:
                                results[noun][tool_name] = []
                            results[noun][tool_name].extend(data[tool_name])
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\nError processing batch {batch_idx + 1}: {str(e)}")
                        continue
            
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            
            elapsed_time = time.time() - start_time
            print(f"{tool_name.upper()} analysis completed: {elapsed_time:.2f} seconds")
        
        for noun, data in results.items():
            noun_type = data['type']
            if noun not in global_keywords[noun_type]:
                global_keywords[noun_type][noun] = {}
            global_keywords[noun_type][noun].update({k: v for k, v in data.items() if k != 'type'})
    
    return global_keywords

def analyze_country_events(df, country, global_keywords):
    print(f"\n=== Starting {country} Events Analysis ===")
    
    save_dir = "Macro/Events/Saves"
    os.makedirs(save_dir, exist_ok=True)
    
    country_df = df[df['country'] == country].copy()
    print(f"Number of {country} events: {len(country_df)}")
    
    print("Copying Global keyword results...")
    country_keywords = {
        'common': {k: v.copy() for k, v in global_keywords['common'].items()},
        'proper': {k: v.copy() for k, v in global_keywords['proper'].items()}
    }
    
    tools = {
        'finbert': (process_finbert_with_device, f"{country}_finbert_keywords.pkl"),
        'roberta': (process_roberta_with_device, f"{country}_roberta_keywords.pkl")
    }
    
    for tool_name, (process_func, save_file) in tools.items():
        save_path = os.path.join(save_dir, save_file)
        
        if not CFG.force_update and os.path.exists(save_path):
            print(f"Loading saved {tool_name.upper()} results...")
            with open(save_path, 'rb') as f:
                results = pickle.load(f)
        else:
            print(f"\nAnalyzing {tool_name.upper()} {country} events...")
            start_time = time.time()
            
            batch_size = 32
            texts = country_df['description'].tolist()
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            results = {}
            with tqdm(total=total_batches, desc=f"{tool_name.upper()} batch processing") as pbar:
                for batch_idx in range(total_batches):
                    try:
                        batch_texts = texts[batch_idx * batch_size:min((batch_idx + 1) * batch_size, len(texts))]
                        batch_result = process_func(batch_texts)
                        
                        for noun, data in batch_result.items():
                            if noun not in results:
                                results[noun] = {'type': data['type']}
                            if tool_name not in results[noun]:
                                results[noun][tool_name] = []
                            results[noun][tool_name].extend(data[tool_name])
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\nError processing batch {batch_idx + 1}: {str(e)}")
                        continue
            
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            
            elapsed_time = time.time() - start_time
            print(f"{tool_name.upper()} analysis completed: {elapsed_time:.2f} seconds")
        
        for noun, data in results.items():
            noun_type = data['type']
            if noun not in country_keywords[noun_type]:
                country_keywords[noun_type][noun] = {}
            country_keywords[noun_type][noun].update({k: v for k, v in data.items() if k != 'type'})
    
    return country_keywords

def visualize_country_analysis(country, country_keywords, df, save_dir="Macro/Events/Visualizations/Country_Analysis"):
    print(f"\n=== Starting {country} Analysis Visualization ===")
    os.makedirs(save_dir, exist_ok=True)
    
    print("Visualizing event impact...")
    country_df = df[df['country'] == country].copy()
    global_df = df[df['country'].isin(['Global', 'Industrial/Scientific'])].copy()
    combined_df = pd.concat([country_df, global_df])
    
    event_impacts = []
    seen_titles = set()
    
    total_events = len(combined_df)
    with tqdm(total=total_events, desc="Calculating event impact") as pbar:
        for idx, row in combined_df.iterrows():
            if row['title'] in seen_titles:
                pbar.update(1)
                continue
            
            seen_titles.add(row['title'])
            
            impact = 0
            sentiment = 0
            for tool in ['finbert_sentiment', 'roberta_sentiment']:
                if tool in row:
                    tool_impact = abs(row[tool])
                    impact += tool_impact
                    sentiment += row[tool]
            
            event_impacts.append({
                'title': row['title'],
                'description': row['description'],
                'country': row['country'],
                'category': row['category'],
                'impact': impact,
                'sentiment': sentiment
            })
            pbar.update(1)
    
    sorted_events = sorted(event_impacts, key=lambda x: x['impact'], reverse=True)
    
    plt.figure(figsize=(28, 14))
    
    plt.subplot(1, 2, 1)
    events = [f"{event['title']}\n({event['category']})" for event in sorted_events[:20]]
    impacts = [event['impact'] for event in sorted_events[:20]]
    colors = ['green' if event['sentiment'] > 0 else 'red' for event in sorted_events[:20]]
    
    y_pos = np.arange(len(events))
    bars = plt.barh(y_pos, impacts, color=colors, alpha=0.6)
    
    min_impact = min(impacts)
    max_impact = max(impacts)
    impact_range = max_impact - min_impact
    plt.xlim(min_impact - impact_range * 0.05, max_impact + impact_range * 0.1)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}',
                ha='left', va='center', fontsize=10)
    
    plt.yticks(y_pos, events, fontsize=12)
    plt.title('Top 20 Most Impactful Events', fontsize=14)
    plt.xlabel('Impact Score (Green: Positive, Red: Negative)', fontsize=12)
    
    plt.subplot(1, 2, 2)
    events = [f"{event['title']}\n({event['category']})" for event in sorted_events[-20:]]
    impacts = [event['impact'] for event in sorted_events[-20:]]
    colors = ['green' if event['sentiment'] > 0 else 'red' for event in sorted_events[-20:]]
    
    y_pos = np.arange(len(events))
    bars = plt.barh(y_pos, impacts, color=colors, alpha=0.6)
    
    min_impact = min(impacts)
    max_impact = max(impacts)
    impact_range = max_impact - min_impact
    plt.xlim(min_impact - impact_range * 0.05, max_impact + impact_range * 0.1)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.6f}',
                ha='left', va='center', fontsize=10)
    
    plt.yticks(y_pos, events, fontsize=12)
    plt.title('Bottom 20 Least Impactful Events', fontsize=14)
    plt.xlabel('Impact Score (Green: Positive, Red: Negative)', fontsize=12)
    
    plt.suptitle(f'{country} - Event Impact Analysis', y=1.02, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{country}_events.png'), bbox_inches='tight')
    plt.close()
    
    return sorted_events

def analyze_market_sentiment_returns(df, country, sorted_events, save_dir="Macro/Events/Visualizations/Market_Sentiment_Returns"):
    print(f"\n=== Starting {country} Sentiment-Returns Analysis ===")
    os.makedirs(save_dir, exist_ok=True)
    
    country_df = df[df['country'] == country].copy()
    global_df = df[df['country'].isin(['Global', 'Industrial/Scientific'])].copy()
    combined_df = pd.concat([country_df, global_df])
    
    daily_data = combined_df.groupby('timestamp').agg({
        'finbert_sentiment': 'mean',
        'roberta_sentiment': 'mean',
        'market_returns': 'first'
    }).reset_index()
    
    daily_data = daily_data.replace([np.inf, -np.inf], np.nan)
    daily_data = daily_data.ffill().bfill()
    
    finbert_data = daily_data['finbert_sentiment'].values
    roberta_data = daily_data['roberta_sentiment'].values
    returns_data = daily_data['market_returns'].values
    
    window = 30
    ma_finbert = pd.Series(finbert_data).rolling(window=window).mean()
    ma_roberta = pd.Series(roberta_data).rolling(window=window).mean()
    ma_returns = pd.Series(returns_data).rolling(window=window).mean()
    
    finbert_volatility = pd.Series(finbert_data).rolling(window=window).std()
    roberta_volatility = pd.Series(roberta_data).rolling(window=window).std()
    returns_volatility = pd.Series(returns_data).rolling(window=window).std()
    
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    finbert_corr, finbert_p = pearsonr(finbert_data, returns_data)
    roberta_corr, roberta_p = pearsonr(roberta_data, returns_data)
    
    ax1.scatter(finbert_data, returns_data, alpha=0.5, color='blue', label=f'FinBERT (corr: {finbert_corr:.3f})')
    ax1.scatter(roberta_data, returns_data, alpha=0.5, color='red', label=f'RoBERTa (corr: {roberta_corr:.3f})')
    ax1.set_xlabel('Sentiment Score')
    ax1.set_ylabel('Market Returns')
    ax1.set_title('Raw Sentiment vs Market Returns')
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    mask = ~np.isnan(ma_finbert) & ~np.isnan(ma_returns)
    ma_finbert_corr, ma_finbert_p = pearsonr(ma_finbert[mask], ma_returns[mask])
    ma_roberta_corr, ma_roberta_p = pearsonr(ma_roberta[mask], ma_returns[mask])
    
    ax2.scatter(ma_finbert[mask], ma_returns[mask], alpha=0.5, color='blue', 
                label=f'FinBERT MA (corr: {ma_finbert_corr:.3f})')
    ax2.scatter(ma_roberta[mask], ma_returns[mask], alpha=0.5, color='red',
                label=f'RoBERTa MA (corr: {ma_roberta_corr:.3f})')
    ax2.set_xlabel(f'{window}-day MA Sentiment')
    ax2.set_ylabel(f'{window}-day MA Returns')
    ax2.set_title('Moving Average Analysis')
    ax2.legend()
    
    ax3 = fig.add_subplot(gs[1, 0])
    vol_finbert_corr, vol_finbert_p = pearsonr(finbert_volatility[mask], returns_volatility[mask])
    vol_roberta_corr, vol_roberta_p = pearsonr(roberta_volatility[mask], returns_volatility[mask])
    
    ax3.scatter(finbert_volatility, returns_volatility, alpha=0.5, color='blue',
                label=f'FinBERT Vol (corr: {vol_finbert_corr:.3f})')
    ax3.scatter(roberta_volatility, returns_volatility, alpha=0.5, color='red',
                label=f'RoBERTa Vol (corr: {vol_roberta_corr:.3f})')
    ax3.set_xlabel('Sentiment Volatility')
    ax3.set_ylabel('Returns Volatility')
    ax3.set_title('Volatility Analysis')
    ax3.legend()
    
    ax4 = fig.add_subplot(gs[1, 1])
    lags = range(-10, 11)
    finbert_lag_corrs = [pearsonr(np.roll(finbert_data, lag), returns_data)[0] for lag in lags]
    roberta_lag_corrs = [pearsonr(np.roll(roberta_data, lag), returns_data)[0] for lag in lags]
    
    ax4.plot(lags, finbert_lag_corrs, marker='o', color='blue', label='FinBERT')
    ax4.plot(lags, roberta_lag_corrs, marker='o', color='red', label='RoBERTa')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Lag (days)')
    ax4.set_ylabel('Correlation')
    ax4.set_title('Lag Analysis')
    ax4.legend()
    
    plt.suptitle(f'{country} - Market Returns Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{country}_market_analysis.png'), bbox_inches='tight')
    plt.close()

def analyze_optimal_weights(df, country, save_dir="Macro/Events/Visualizations/Weight_Analysis"):
    print(f"\n=== Starting {country} Event Type and Sentiment Tool Weight Analysis ===")
    os.makedirs(save_dir, exist_ok=True)
    
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    country_df = df[df['country'] == country].copy()
    global_df = df[df['country'] == 'Global'].copy()
    tech_df = df[df['country'] == 'Industrial/Scientific'].copy()
    
    sentiment_tools = ['finbert', 'roberta']
    daily_data = {}
    
    for tool in sentiment_tools:
        daily_national = country_df.groupby('timestamp')[f'{tool}_sentiment'].mean()
        daily_global = global_df.groupby('timestamp')[f'{tool}_sentiment'].mean()
        daily_tech = tech_df.groupby('timestamp')[f'{tool}_sentiment'].mean()
        daily_returns = country_df.groupby('timestamp')['market_returns'].first()
        
        daily_data[tool] = pd.DataFrame({
            'national': daily_national,
            'global': daily_global,
            'tech': daily_tech,
            'returns': daily_returns
        }).fillna(0)

    def safe_correlation(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return 0, 1
        mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        if sum(mask) < 2:
            return 0, 1
        try:
            return pearsonr(x[mask], y[mask])
        except:
            return 0, 1

    tool_results = {}
    
    for tool in sentiment_tools:
        results = {
            'granger_causality': {},
            'correlation': {},
            'correlation_pvalues': {},
            'regression': {},
            'grid_search': {}
        }
        
        analysis_df = daily_data[tool]
        
        for event_type in ['national', 'global', 'tech']:
            try:
                data = analysis_df[[event_type, 'returns']].copy()
                data = data.replace([np.inf, -np.inf], np.nan)
                data = data.dropna()
                if data[event_type].std() == 0 or data['returns'].std() == 0:
                    results['granger_causality'][event_type] = {
                        'f_stats': [0] * 5,
                        'p_values': [1] * 5,
                        'max_f_stat': 0,
                        'min_p_value': 1
                    }
                    continue
                
                if len(data) > 5:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        granger_test = grangercausalitytests(data, maxlag=5, verbose=False)
                    f_stats = []
                    p_values = []
                    for lag in range(1, 6):
                        f_stat = granger_test[lag][0]['ssr_ftest'][0]
                        p_value = granger_test[lag][0]['ssr_ftest'][1]
                        f_stats.append(f_stat)
                        p_values.append(p_value)
                    results['granger_causality'][event_type] = {
                        'f_stats': f_stats,
                        'p_values': p_values,
                        'max_f_stat': max(f_stats),
                        'min_p_value': min(p_values)
                    }
                else:
                    results['granger_causality'][event_type] = {
                        'f_stats': [0] * 5,
                        'p_values': [1] * 5,
                        'max_f_stat': 0,
                        'min_p_value': 1
                    }
            except Exception as e:
                print(f"Error during Granger analysis ({event_type}): {str(e)}")
                results['granger_causality'][event_type] = {
                    'f_stats': [0] * 5,
                    'p_values': [1] * 5,
                    'max_f_stat': 0,
                    'min_p_value': 1
                }
        
        for event_type in ['national', 'global', 'tech']:
            try:
                data = analysis_df[[event_type, 'returns']].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) > 0:
                    corr, p_value = safe_correlation(data[event_type].values, data['returns'].values)
                    results['correlation'][event_type] = corr
                    results['correlation_pvalues'][event_type] = p_value
                else:
                    results['correlation'][event_type] = 0
                    results['correlation_pvalues'][event_type] = 1
            except:
                results['correlation'][event_type] = 0
                results['correlation_pvalues'][event_type] = 1
        
        try:
            data = analysis_df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) > 0:
                reg = LinearRegression()
                X = data[['national', 'global', 'tech']]
                y = data['returns']
                reg.fit(X, y)
                
                coef_sum = sum(abs(reg.coef_))
                if coef_sum > 0:
                    results['regression'] = {
                        'national': abs(reg.coef_[0])/coef_sum,
                        'global': abs(reg.coef_[1])/coef_sum,
                        'tech': abs(reg.coef_[2])/coef_sum
                    }
                else:
                    results['regression'] = {'national': 1/3, 'global': 1/3, 'tech': 1/3}
            else:
                results['regression'] = {'national': 1/3, 'global': 1/3, 'tech': 1/3}
        except:
            results['regression'] = {'national': 1/3, 'global': 1/3, 'tech': 1/3}
        
        best_corr = -1
        best_weights = None
        
        weights = np.arange(0, 1.1, 0.1)
        for w1 in weights:
            for w2 in weights:
                w3 = 1 - w1 - w2
                if w3 >= 0:
                    weighted_sentiment = (
                        w1 * analysis_df['national'] +
                        w2 * analysis_df['global'] +
                        w3 * analysis_df['tech']
                    )
                    corr = weighted_sentiment.corr(analysis_df['returns'])
                    if abs(corr) > best_corr:
                        best_corr = abs(corr)
                        best_weights = {'national': w1, 'global': w2, 'tech': w3}
        
        results['grid_search'] = best_weights
        
        method_weights = {
            'granger_causality': 0.3,
            'correlation': 0.3,
            'regression': 0.2,
            'grid_search': 0.2
        }
        
        final_weights = {'national': 0, 'global': 0, 'tech': 0}
        for method, weight in method_weights.items():
            method_result = results[method]
            for event_type in final_weights.keys():
                if method == 'granger_causality':
                    final_weights[event_type] += method_result[event_type]['max_f_stat'] * weight
                else:
                    final_weights[event_type] += method_result[event_type] * weight
        
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v/total_weight for k, v in final_weights.items()}
        
        tool_results[tool] = {
            'methods': results,
            'final_weights': final_weights
        }
    
    fig = plt.figure(figsize=(20, 10))
    
    ax1 = plt.subplot(1, 2, 1)
    
    category_mapping = {
        'National': 'national',
        'Global': 'global',
        'Industrial/Technological': 'tech'
    }
    categories = ['National', 'Global', 'Industrial/Technological']
    y = np.arange(len(categories))
    width = 0.35
    
    finbert_weights = [tool_results['finbert']['final_weights'][category_mapping[cat]] for cat in categories]
    ax1.barh(y - width/2, finbert_weights, width, label='FinBERT', color=['royalblue', 'cornflowerblue', 'lightsteelblue'], alpha=0.8)
    
    roberta_weights = [tool_results['roberta']['final_weights'][category_mapping[cat]] for cat in categories]
    ax1.barh(y + width/2, roberta_weights, width, label='RoBERTa', color=['darkred', 'red', 'lightcoral'], alpha=0.8)
    
    for i, v in enumerate(finbert_weights):
        ax1.text(v, i - width/2, f'{v:.2f}', ha='left', va='center', fontsize=10)
    for i, v in enumerate(roberta_weights):
        ax1.text(v, i + width/2, f'{v:.2f}', ha='left', va='center', fontsize=10)
    
    ax1.set_xlabel('Optimal Weight', fontsize=12)
    ax1.set_title(f'{country} - Optimal Event Type Weights', fontsize=14, pad=20)
    ax1.set_yticks(y)
    ax1.set_yticklabels(categories)
    ax1.legend()
    
    ax2 = plt.subplot(1, 2, 2)
    
    methods = ['granger_causality', 'correlation', 'regression', 'grid_search']
    method_names = ['Granger', 'Correlation', 'Regression', 'Grid Search']
    
    finbert_weights = []
    for method in methods:
        if method == 'granger_causality':
            weights = [max(0, tool_results['finbert']['methods'][method][cat]['max_f_stat']) for cat in ['national', 'global', 'tech']]
        else:
            weights = [max(0, tool_results['finbert']['methods'][method][cat]) for cat in ['national', 'global', 'tech']]
        finbert_weights.append(weights)
    
    roberta_weights = []
    for method in methods:
        if method == 'granger_causality':
            weights = [max(0, tool_results['roberta']['methods'][method][cat]['max_f_stat']) for cat in ['national', 'global', 'tech']]
        else:
            weights = [max(0, tool_results['roberta']['methods'][method][cat]) for cat in ['national', 'global', 'tech']]
        roberta_weights.append(weights)
    
    x = np.arange(len(methods))
    width = 0.12
    
    for i, method in enumerate(methods):
        stars = ''
        if method == 'granger_causality':
            min_p = min(min(tool_results['finbert']['methods'][method][cat]['min_p_value'] 
                          for cat in ['national', 'global', 'tech']),
                       min(tool_results['roberta']['methods'][method][cat]['min_p_value']
                          for cat in ['national', 'global', 'tech']))
        elif method == 'correlation':
            min_p = min(min(tool_results['finbert']['methods']['correlation_pvalues'][cat]
                          for cat in ['national', 'global', 'tech']),
                       min(tool_results['roberta']['methods']['correlation_pvalues'][cat]
                          for cat in ['national', 'global', 'tech']))
        else:
            min_p = 1
        
        if min_p < 0.01:
            stars = ' ***'
        elif min_p < 0.05:
            stars = ' **'
        elif min_p < 0.1:
            stars = ' *'
        
        positions = [x[i] - width*2.5, x[i] - width*1.5, x[i] - width*0.5]
        colors = ['royalblue', 'cornflowerblue', 'lightsteelblue']
        labels = ['FinBERT National', 'FinBERT Global', 'FinBERT Tech'] if i == 0 else ["", "", ""]
        
        for pos, color, label, w in zip(positions, colors, labels, finbert_weights[i]):
            ax2.barh(pos, w, width, color=color, alpha=0.8, label=label)
        
        positions = [x[i] + width*0.5, x[i] + width*1.5, x[i] + width*2.5]
        colors = ['darkred', 'red', 'lightcoral']
        labels = ['RoBERTa National', 'RoBERTa Global', 'RoBERTa Tech'] if i == 0 else ["", "", ""]
        
        for pos, color, label, w in zip(positions, colors, labels, roberta_weights[i]):
            ax2.barh(pos, w, width, color=color, alpha=0.8, label=label)
    
    method_labels = []
    for i, method in enumerate(methods):
        stars = ''
        if method == 'granger_causality':
            min_p = min(min(tool_results['finbert']['methods'][method][cat]['min_p_value'] 
                          for cat in ['national', 'global', 'tech']),
                       min(tool_results['roberta']['methods'][method][cat]['min_p_value']
                          for cat in ['national', 'global', 'tech']))
        elif method == 'correlation':
            min_p = min(min(tool_results['finbert']['methods']['correlation_pvalues'][cat]
                          for cat in ['national', 'global', 'tech']),
                       min(tool_results['roberta']['methods']['correlation_pvalues'][cat]
                          for cat in ['national', 'global', 'tech']))
        else:
            min_p = 1
        
        if min_p < 0.01:
            stars = ' $\\mathbf{***}$'
        elif min_p < 0.05:
            stars = ' $\\mathbf{**}$'
        elif min_p < 0.1:
            stars = ' $\\mathbf{*}$'
        
        method_labels.append(f"{method_names[i]}{stars}")
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(method_labels)
    
    ax2.set_xlabel('Weight', fontsize=12)
    ax2.set_title(f'Method-wise Analysis\n(*** p<0.01, ** p<0.05, * p<0.1)', fontsize=14, pad=20)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle(f'{country} - Event Weight Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f'{country}_analysis.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    return tool_results

def visualize_optimal_weights(tool_results, country, save_dir="Macro/Events/Visualizations/Weight_Analysis"):
    plt.figure(figsize=(12, 6))
    
    tools = ['finbert', 'roberta']
    categories = ['national', 'global', 'tech']
    x = np.arange(len(categories))
    width = 0.35
    
    for i, tool in enumerate(tools):
        weights = [tool_results[tool]['final_weights'][cat] for cat in categories]
        plt.bar(x + i*width, weights, width, label=tool.upper())
    
    plt.xlabel('Event Type')
    plt.ylabel('Weight')
    plt.title(f'{country} - Final Optimal Weights')
    plt.xticks(x + width/2, ['National', 'Global', 'Tech/Industrial'])
    plt.legend()
    
    for i, tool in enumerate(tools):
        weights = [tool_results[tool]['final_weights'][cat] for cat in categories]
        for j, v in enumerate(weights):
            plt.text(x[j] + i*width, v, f'{v:.2f}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{country}_optimal_weights_combined.png'))
    plt.close()

def main():
    try:
        df = parse_jsonl("Macro/Events/BatchAggreg.jsonl")
        df = classify_unknowns(df)
        
        df = analyze_sentiments(df, save_dir=CFG.save_dir)
        
        df = download_stock_data(df)
        
        global_keywords = analyze_global_events(df)
        
        market_results = {}
        weight_results = {}
        market_event_mapping = {
            '^HSI': {
                'national': ['Hong Kong'],
                'global': ['Global', 'Industrial/Scientific']
            },
            '^KS11': {
                'national': ['South Korea'],
                'global': ['Global', 'Industrial/Scientific']
            },
            '^GSPC': {
                'national': ['U.S'],
                'global': ['Global', 'Industrial/Scientific']
            },
            '^N225': {
                'national': ['Japan'],
                'global': ['Global', 'Industrial/Scientific']
            }
        }
        for market_index, mapping in market_event_mapping.items():
            country = mapping['national'][0]
            print(f"\n=== {market_index} ({country}) analysis started ===")
            
            country_keywords = analyze_country_events(df, country, global_keywords)
            
            sorted_events = visualize_country_analysis(country, country_keywords, df)
            
            analyze_market_sentiment_returns(df, country, sorted_events)
            
            weight_results[country] = analyze_optimal_weights(df, country)
            
            market_results[market_index] = (country_keywords, sorted_events)
            print(f"=== {market_index} ({country}) analysis completed ===\n")
        
        return df, market_results, weight_results
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    df, market_results, weight_results = main()
