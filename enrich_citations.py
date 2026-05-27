import pandas as pd
import requests
import time
from tqdm import tqdm

def get_citations_and_reads(arxiv_url, retry=0):
    max_retries = 5
    arxiv_id = arxiv_url.split("/")[-1].split("v")[0]
    url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
    params = {"fields": "citationCount,readersCount"}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get("citationCount", 0), data.get("readersCount", 0)
        elif response.status_code == 429:
            if retry >= max_retries:
                print(f"Max retries exceeded for {arxiv_id}")
                return None, None
            wait_time = 2 ** retry * 5
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            return get_citations_and_reads(arxiv_url, retry+1)
        else:
            print(f"Failed for {arxiv_id}: Status {response.status_code}")
            return None, None
    except Exception as e:
        print(f"Exception for {arxiv_id}: {e}")
        return None, None

def enrich_csv(in_path="papers.csv", out_path="papers_with_citations.csv"):
    df = pd.read_csv(in_path)
    citations, reads = [], []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        cit, read = get_citations_and_reads(row['URL'])
        if cit is None:
            cit = 0
        if read is None:
            read = 0
        citations.append(cit)
        reads.append(read)
        print(f"Processed {i+1}/{len(df)}: {row['Title']} - Citations: {cit}, Reads: {read}")
        time.sleep(3)  # To respect API rate limits
    df['Citations'] = citations
    df['Estimated_Reads'] = reads
    df.to_csv(out_path, index=False)
    print(f"Enriched CSV saved to {out_path}")

if __name__ == "__main__":
    enrich_csv()
