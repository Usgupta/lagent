import asyncio
import json
import logging
import random
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Type, Union

import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache, cached
from duckduckgo_search import DDGS

from lagent.actions import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser

from pathlib import Path
import os
from PyPDF2 import PdfReader

from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Tuple
import json
import io
from io import BufferedReader


# Initialize the embedding model and tokenizer
model_name = "thenlper/gte-base"  # Change to your chosen model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def extract_text_from_pdf(file) -> List[Tuple[int, str]]:
    pages_text = []        

    file_content = file.file.read()
    pdf_file = io.BytesIO(file_content)
   
    
    reader = PdfReader(pdf_file)
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ''
        pages_text.append((page_num + 1, text))  # Store (page_number, text)
    return pages_text

class PdfSearch:
    def __init__(self,
                 api_key: str,
                 region: str = 'zh-CN',
                 topk: int = 3,
                 black_list: List[str] = [
                     'enoN',
                     'youtube.com',
                     'bilibili.com',
                     'researchgate.net',
                 ],
                 files = List[Tuple[str, BufferedReader]],
                 ):
        self.topk = topk
        
        #CREATE YOUR OWN QDRANT client URL 
        
        self.qdrant_client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "pdf_search_collection"
        self.idx = 0
        self.files = files
        self._initialize_qdrant()
        

    def _initialize_qdrant(self):
        self.qdrant_client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),  # Adjust size based on your model
        )
        
        for file in self.files:
            
            print(file)
            
            print(type(file))
            
            self._index_pdf(file)

    def _index_pdf(self, pdf_file):
        pages_text = extract_text_from_pdf(pdf_file)
        
        for page_number, pdf_text in pages_text:
            inputs = tokenizer(pdf_text, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()  # Get embeddings
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[{
                    "id": self.idx,
                    "vector": embeddings.tolist(),  # Ensure it's a list for Qdrant
                    "payload": {"path": pdf_file.filename, "content": pdf_text, "page_number": page_number}
                }]
            )
            self.idx += 1

    def search(self, query: str) -> dict:
        inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            query_embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
        
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=self.topk
        )
        
        filtered_results = self._filter_results(search_results)
        return filtered_results

    def _filter_results(self, results) -> dict:
        filtered_results = {}
        count = 0
        
        for result in results:
            filtered_results[count] = {
                'filename': result.payload["path"],
                # you may replace this with an AI summary instead of first 200 characters
                'summ': result.payload["content"][:200], 
                'title': result.payload["path"].split("/")[-1],
                'page_number': result.payload["page_number"]  
            }
            count += 1
            
            if count >= self.topk:
                break
                
        return filtered_results
    
    
class PdfBrowser(BaseAction):
    """Wrapper around the PDF Browser Tool.
    """

    def __init__(self,
                 searcher_type: str = 'PdfSearch',
                 timeout: int = 5,
                 black_list: Optional[List[str]] = None,
                 topk: int = 20,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True,
                 **kwargs):
        self.searcher = eval(searcher_type)(
            black_list=black_list, topk=topk, **kwargs)
        self.fetcher = ContentFetcher(timeout=timeout)
        self.search_results = None
        super().__init__(description, parser, enable)

    @tool_api
    def search(self, query: Union[str, List[str]]) -> dict:
        """PDF search API
        Args:
            query (List[str]): list of search query strings
        """
        queries = query if isinstance(query, list) else [query]
        search_results = {}

        with ThreadPoolExecutor() as executor:
            future_to_query = {
                executor.submit(self.searcher.search, q): q
                for q in queries
            }

            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                except Exception as exc:
                    warnings.warn(f'{query} generated an exception: {exc}')
                else:
                    for result in results.values():
                        if result['filename'] not in search_results:
                            search_results[result['filename']] = result
                        else:
                            search_results[
                                result['filename']]['summ'] += f"\n{result['summ']}"

        self.search_results = {
            idx: result
            for idx, result in enumerate(search_results.values())
        }
        return self.search_results

    @tool_api
    def select(self, select_ids: List[int]) -> dict:
        """Get the detailed content on the selected pages.

        Args:
            select_ids (List[int]): list of index to select. Max number of index to be selected is no more than 4.
        """
        if not self.search_results:
            raise ValueError('No search results to select from.')

        new_search_results = {}
        with ThreadPoolExecutor() as executor:
            future_to_id = {
                executor.submit(self.fetcher.fetch,
                                self.search_results[select_id]['filename']):
                select_id
                for select_id in select_ids if select_id in self.search_results
            }

            for future in as_completed(future_to_id):
                select_id = future_to_id[future]
                try:
                    web_success, web_content = future.result()
                except Exception as exc:
                    warnings.warn(f'{select_id} generated an exception: {exc}')
                else:
                    if web_success:
                        self.search_results[select_id][
                            'content'] = web_content[:8192]
                        new_search_results[select_id] = self.search_results[
                            select_id].copy()
                        new_search_results[select_id].pop('summ')

        return new_search_results

    @tool_api
    def open_url(self, filename: str) -> dict:
        print(f'Start Browsing: {filename}')
        web_success, web_content = self.fetcher.fetch(filename)
        if web_success:
            return {'type': 'text', 'content': web_content}
        else:
            return {'error': web_content}

class ContentFetcher:

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    @cached(cache=TTLCache(maxsize=100, ttl=600))
    def fetch(self, url: str) -> Tuple[bool, str]:
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            html = response.content
        except requests.RequestException as e:
            return False, str(e)

        text = BeautifulSoup(html, 'html.parser').get_text()
        cleaned_text = re.sub(r'\n+', '\n', text)
        return True, cleaned_text

class PdfBrowser(BaseAction):
    """Wrapper around the PDF Browser Tool.
    """

    def __init__(self,
                 searcher_type: str = 'PdfSearch',
                 timeout: int = 5,
                 black_list: Optional[List[str]] = None,
                 topk: int = 20,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True,
                 files = List[Tuple[str, BufferedReader]],
                 **kwargs):
        self.searcher = eval(searcher_type)(
            black_list=black_list,
            topk=topk,
            files=files,
            **kwargs)
        self.fetcher = ContentFetcher(timeout=timeout)
        self.search_results = None
        super().__init__(description, parser, enable)

    @tool_api
    def search(self, query: Union[str, List[str]]) -> dict:
        """PDF search API
        Args:
            query (List[str]): list of search query strings
        """
        queries = query if isinstance(query, list) else [query]
        search_results = {}

        with ThreadPoolExecutor() as executor:
            future_to_query = {
                executor.submit(self.searcher.search, q): q
                for q in queries
            }

            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                except Exception as exc:
                    warnings.warn(f'{query} generated an exception: {exc}')
                else:
                    for result in results.values():
                        if result['filename'] not in search_results:
                            search_results[result['filename']] = result
                        else:
                            search_results[
                                result['filename']]['summ'] += f"\n{result['summ']}"

        self.search_results = {
            idx: result
            for idx, result in enumerate(search_results.values())
        }
        return self.search_results

    @tool_api
    def select(self, select_ids: List[int]) -> dict:
        """Get the detailed content on the selected pages.

        Args:
            select_ids (List[int]): list of index to select. Max number of index to be selected is no more than 4.
        """
        if not self.search_results:
            raise ValueError('No search results to select from.')

        new_search_results = {}
        with ThreadPoolExecutor() as executor:
            future_to_id = {
                executor.submit(self.fetcher.fetch,
                                self.search_results[select_id]['filename']):
                select_id
                for select_id in select_ids if select_id in self.search_results
            }

            for future in as_completed(future_to_id):
                select_id = future_to_id[future]
                try:
                    web_success, web_content = future.result()
                except Exception as exc:
                    warnings.warn(f'{select_id} generated an exception: {exc}')
                else:
                    if web_success:
                        self.search_results[select_id][
                            'content'] = web_content[:8192]
                        new_search_results[select_id] = self.search_results[
                            select_id].copy()
                        new_search_results[select_id].pop('summ')

        return new_search_results

    @tool_api
    def open_url(self, filename: str) -> dict:
        print(f'Start Browsing: {filename}')
        web_success, web_content = self.fetcher.fetch(filename)
        if web_success:
            return {'type': 'text', 'content': web_content}
        else:
            return {'error': web_content}