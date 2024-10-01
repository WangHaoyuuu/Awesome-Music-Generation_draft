import numpy as np
import faiss
import time
from itertools import product
from tqdm import tqdm
from collections import defaultdict
import json
import sys
import numpy as np
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class FaissDatasetBuilder:
    """
    A class for building and managing Faiss indexes for dataset searching.
    Supports Flat, IVF, and HNSW index types.
    """
    def __init__(self, data, index_type='IVF', nlist=50, nprobe=10, M=16, efConstruction=100, efSearch=10, batch_size=500):
        self.data = data
        self.dimension = self.data.shape[1]
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.M = M
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.index = None
        self.batch_size = batch_size

    def build_index(self):
        """
        Build the Faiss index based on the specified index type.
        Supports Flat, IVF, and HNSW indexes.
        """
        if self.index_type == 'Flat':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == 'IVF':
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        elif self.index_type == 'HNSW':
            self.index = faiss.IndexHNSWFlat(self.dimension, self.M)
            self.index.hnsw.efConstruction = self.efConstruction
            self.index.hnsw.efSearch = self.efSearch
        
        if self.index_type == 'IVF':
            # Train in batches for IVF index
            for i in range(0, len(self.data), self.batch_size):
                batch = self.data[i:i+self.batch_size].astype(np.float32)
                self.index.train(batch)
            
        # Add data in batches for all index types
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i+self.batch_size].astype(np.float32)
            self.index.add(batch)
        
        if self.index_type == 'IVF':
            self.index.nprobe = self.nprobe

    def search(self, query, k=5):
        """
        Perform a search on the built index.
        """
        if self.index is None:
            raise ValueError("Index not built. Please call build_index() method first.")
        
        if isinstance(query, np.ndarray):
            query = query.astype(np.float32)
        
        start_time = time.time()
        distances, indices = self.index.search(query.reshape(1, -1), k)
        end_time = time.time()

        return {
            'indices': indices.squeeze(),
            'distances': distances.squeeze(),
            'search_time': end_time - start_time
        }

    def batch_search(self, queries, k=5):
        """
        Perform a batch search on the built index.
        """
        if self.index is None:
            raise ValueError("Index not built. Please call build_index() method first.")
        
        if isinstance(queries, np.ndarray):
            queries = queries.astype(np.float32)
        
        start_time = time.time()
        distances, indices = self.index.search(queries, k)
        end_time = time.time()

        return {
            'indices': indices,
            'distances': distances,
            'search_time': end_time - start_time
        }

    def save_index(self, filename):
        """
        Save the built index to a file.
        """
        if self.index is None:
            raise ValueError("Index not built. Please call build_index() method first.")
        faiss.write_index(self.index, filename)

    def load_index(self, filename):
        """
        Load a previously saved index from a file.
        """
        self.index = faiss.read_index(filename)
        if self.index_type == 'IVF':
            self.index.nprobe = self.nprobe

def calculate_nlist_values(num_vectors):
    """
    Calculate a range of nlist values for IVF index based on the number of vectors.
    """
    base_nlist = int(np.sqrt(num_vectors))
    return [
        max(50, base_nlist // 8),
        max(50, base_nlist // 4),
        max(50, base_nlist // 2),
        base_nlist,
        min(base_nlist * 2, num_vectors // 10),
        min(base_nlist * 4, num_vectors // 5),
        min(base_nlist * 8, num_vectors // 2),
        min(base_nlist * 16, num_vectors)
    ]

def calculate_nprobe_values(num_vectors, nlist):
    """
    Calculate a range of nprobe values for IVF index based on the number of vectors and nlist.
    """
    base_nprobe = max(1, int(np.sqrt(nlist)))
    
    if num_vectors < 10000:
        nprobe_values = [1, base_nprobe, min(base_nprobe * 2, nlist), 
                         min(base_nprobe * 4, nlist), min(base_nprobe * 8, nlist)]
    elif num_vectors < 100000:
        nprobe_values = [1, base_nprobe, min(base_nprobe * 2, nlist), 
                         min(base_nprobe * 4, nlist), min(base_nprobe * 8, nlist),
                         min(base_nprobe * 16, nlist)]
    else:
        nprobe_values = [1, base_nprobe, min(base_nprobe * 2, nlist), 
                         min(base_nprobe * 4, nlist), min(base_nprobe * 8, nlist),
                         min(base_nprobe * 16, nlist), min(base_nprobe * 32, nlist),
                         min(base_nprobe * 64, nlist)]
    
    return sorted(list(set(nprobe_values)))

def calculate_hnsw_params(num_vectors):
    """
    Calculate ranges for HNSW parameters (M and efConstruction) based on the number of vectors.
    """
    base_m = 16
    
    if num_vectors < 10000:
        m_values = [4, 8, base_m, 32]
    elif num_vectors < 100000:
        m_values = [4, 8, base_m, 32, 48, 64]
    else:
        m_values = [4, 8, base_m, 32, 48, 64, 96]
    
    m_values = [min(m, 96) for m in m_values]
    
    ef_construction_values = [max(m * 2, 40) for m in m_values]
    ef_construction_values += [min(v * 2, 800) for v in ef_construction_values]
    ef_construction_values += [min(v * 4, 1600) for v in ef_construction_values]
    ef_construction_values += [min(v * 8, 3200) for v in ef_construction_values]
    
    m_values = sorted(list(set(m_values)))
    ef_construction_values = sorted(list(set(ef_construction_values)))
    
    return m_values, ef_construction_values

def calculate_ef_search_values(m_values):
    """
    Calculate a range of efSearch values for HNSW index based on M values.
    """
    ef_search_values = []
    for m in m_values:
        ef_search_values.extend([m, m * 2, m * 4, m * 8, m * 16, m * 32])
    return sorted(list(set(ef_search_values)))

def evaluate_faiss_params(audio_data, melody_data, audio_queries, k_values):
    """
    Evaluate different Faiss index types and parameters on the given dataset.
    """
    index_types = ['Flat', 'IVF', 'HNSW']
    nlist_values = calculate_nlist_values(len(audio_data))
    d = audio_data.shape[1]

    results = []
    total_iterations = sum([
        len(k_values) * 2,  # Flat (audio and melody)
        len(nlist_values) * len(calculate_nprobe_values(len(audio_data), nlist_values[0])) * len(k_values) * 2,  # IVF
        len(calculate_hnsw_params(len(audio_data))[0]) * len(calculate_hnsw_params(len(audio_data))[1]) * len(calculate_ef_search_values(calculate_hnsw_params(len(audio_data))[0])) * len(k_values) * 2  # HNSW
    ])

    with tqdm(total=total_iterations, desc="Evaluation Progress") as pbar:
        for index_type in index_types:
            if index_type == 'Flat':
                # Evaluate Flat index
                audio_index = faiss.IndexFlatL2(d)
                audio_index.add(audio_data)
                melody_index = faiss.IndexFlatL2(d)
                melody_index.add(melody_data)
                for k in k_values:
                    result = evaluate_index(audio_index, melody_index, audio_queries, k)
                    result['index_type'] = 'Flat'
                    results.append(result)
                    pbar.update(2)
            
            elif index_type == 'IVF':
                # Evaluate IVF index with different nlist and nprobe values
                for nlist in nlist_values:
                    quantizer = faiss.IndexFlatL2(d)
                    audio_index = faiss.IndexIVFFlat(quantizer, d, nlist)
                    audio_index.train(audio_data)
                    audio_index.add(audio_data)
                    melody_index = faiss.IndexIVFFlat(quantizer, d, nlist)
                    melody_index.train(melody_data)
                    melody_index.add(melody_data)
                    nprobe_values = calculate_nprobe_values(len(audio_data), nlist)
                    for nprobe in nprobe_values:
                        audio_index.nprobe = nprobe
                        melody_index.nprobe = nprobe
                        for k in k_values:
                            result = evaluate_index(audio_index, melody_index, audio_queries, k)
                            result.update({
                                'index_type': 'IVF',
                                'nlist': nlist,
                                'nprobe': nprobe
                            })
                            results.append(result)
                            pbar.update(2)

            elif index_type == 'HNSW':
                # Evaluate HNSW index with different M, efConstruction, and efSearch values
                m_values, efConstruction_values = calculate_hnsw_params(len(audio_data))
                ef_search_values = calculate_ef_search_values(m_values)
                for m_value, efConstruction in product(m_values, efConstruction_values):
                    audio_index = faiss.IndexHNSWFlat(d, m_value)
                    audio_index.hnsw.efConstruction = efConstruction
                    audio_index.add(audio_data)
                    melody_index = faiss.IndexHNSWFlat(d, m_value)
                    melody_index.hnsw.efConstruction = efConstruction
                    melody_index.add(melody_data)
                    for efSearch in ef_search_values:
                        audio_index.hnsw.efSearch = efSearch
                        melody_index.hnsw.efSearch = efSearch
                        for k in k_values:
                            result = evaluate_index(audio_index, melody_index, audio_queries, k)
                            result.update({
                                'index_type': 'HNSW',
                                'M': m_value,
                                'efConstruction': efConstruction,
                                'efSearch': efSearch
                            })
                            results.append(result)
                            pbar.update(2)

    return results

def evaluate_index(audio_index, melody_index, audio_queries, k):
    """
    Evaluate the performance of a given index configuration.
    """
    # Evaluate audio to audio query
    audio_to_audio_start_time = time.time()
    audio_to_audio_distances, audio_to_audio_indices = audio_index.search(audio_queries, k)
    audio_to_audio_search_time = time.time() - audio_to_audio_start_time

    # Evaluate audio to melody query
    audio_to_melody_start_time = time.time()
    audio_to_melody_distances, audio_to_melody_indices = melody_index.search(audio_queries, k)
    audio_to_melody_search_time = time.time() - audio_to_melody_start_time

    intersection_rates = []
    for i, (audio_audio_idx, audio_melody_idx) in enumerate(zip(audio_to_audio_indices, audio_to_melody_indices)):
        # Check if the query vector index is included in the results
        audio_match = i in audio_audio_idx
        melody_match = i in audio_melody_idx
        
        # If both results contain the query vector index, consider it as an intersection
        intersection_rate = 1 if melody_match else 0
        intersection_rates.append(intersection_rate)

    average_intersection_rate = np.mean(intersection_rates)

    return {
        'k': k,
        'audio_to_audio_search_time': audio_to_audio_search_time,
        'audio_to_melody_search_time': audio_to_melody_search_time,
        'average_intersection_rate': average_intersection_rate,
    }

def save_results_to_json(results, save_path, filename='find_best_faiss_params.json'):
    """
    Save the evaluation results to a JSON file.
    """
    serializable_results = []
    for result in results:
        serializable_result = {
            'index_type': result['index_type'],
            'k': result['k'],
            'average_intersection_rate': float(result['average_intersection_rate']),
            'audio_to_audio_search_time': float(result['audio_to_audio_search_time']),
            'audio_to_melody_search_time': float(result['audio_to_melody_search_time'])
        }
        if 'nlist' in result:
            serializable_result['nlist'] = result['nlist']
            serializable_result['nprobe'] = result['nprobe']
        if 'M' in result:
            serializable_result['M'] = result['M']
            serializable_result['efConstruction'] = result['efConstruction']
            serializable_result['efSearch'] = result['efSearch']
        serializable_results.append(serializable_result)

    serializable_results.sort(key=lambda x: x['average_intersection_rate'], reverse=True)

    with open(save_path + '/' + filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {filename}")

if __name__ == "__main__":
    # Load data
    melody_data = np.load('./Multimodal_Alignment_npy/musiccaps_melody_362_trimmed.npy')
    audio_data = np.load('./Multimodal_Alignment_npy/musiccaps_audio_362_trimmed.npy')
    audio_queries = np.load('./Multimodal_Alignment_npy/musiccaps_audio_362_trimmed.npy')

    # Use all data
    audio_data = audio_data[:]
    melody_data = melody_data[:]
    audio_queries = audio_queries[:]

    # Set k values for nearest neighbor search
    k_values = [1]

    # Run evaluation
    results = evaluate_faiss_params(audio_data, melody_data, audio_queries, k_values)

    # Sort results by intersection rate and print
    sorted_results = sorted(results, key=lambda x: x['average_intersection_rate'], reverse=True)
    
    print("Parameter evaluation results (sorted by intersection rate in descending order):")
    for result in sorted_results[:30]:  # Print only the top 30 results
        print(f"Index type: {result['index_type']}, k: {result['k']}")
        if 'nlist' in result:
            print(f"nlist: {result['nlist']}, nprobe: {result['nprobe']}")
        if 'M' in result:
            print(f"M: {result['M']}, efConstruction: {result['efConstruction']}, efSearch: {result['efSearch']}")
        print(f"Average intersection rate: {result['average_intersection_rate']:.4f}")
        print(f"Audio to audio search time: {result['audio_to_audio_search_time']:.6f} seconds")
        print(f"Audio to melody search time: {result['audio_to_melody_search_time']:.6f} seconds")
        print("-" * 80)
    save_path = './'
    # Save results to JSON file
    save_results_to_json(results, save_path)