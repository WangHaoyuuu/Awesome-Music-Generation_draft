import numpy as np
import faiss
import time
import os
import json

def build_hnsw_index(data, M=32, efConstruction=128):
    """
    Build HNSW index.

    Args:
        data (np.ndarray): Dataset with shape (n_samples, d).
        M (int): Number of connections for HNSW.
        efConstruction (int): ef parameter for index construction.

    Returns:
        faiss.IndexHNSWFlat: Constructed HNSW index.
    """
    d = data.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction

    start_time = time.time()
    index.add(data)
    end_time = time.time()

    print(f"HNSW index construction time: {end_time - start_time:.2f} seconds")
    return index

def build_ivf_index(data, nlist=50):
    """
    Build IVF index.

    Args:
        data (np.ndarray): Dataset with shape (n_samples, d).
        nlist (int): Number of centroids for IVF.

    Returns:
        faiss.IndexIVFFlat: Constructed IVF index.
    """
    d = data.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    start_time = time.time()
    index.train(data)
    index.add(data)
    end_time = time.time()

    print(f"IVF index construction time: {end_time - start_time:.2f} seconds")
    return index

def evaluate_index(audio_index, melody_index, audio_queries, k=1):
    """
    Evaluate index performance.

    Args:
        audio_index (faiss.Index): Audio index.
        melody_index (faiss.Index): Melody index.
        audio_queries (np.ndarray): Query vectors with shape (n_queries, d).
        k (int): Number of nearest neighbors.

    Returns:
        dict: Evaluation results.
    """
    # Evaluate audio to audio query
    audio_to_audio_start_time = time.time()
    audio_to_audio_distances, audio_to_audio_indices = audio_index.search(audio_queries, k)
    audio_to_audio_search_time = time.time() - audio_to_audio_start_time

    # Evaluate audio to melody query
    audio_to_melody_start_time = time.time()
    audio_to_melody_distances, audio_to_melody_indices = melody_index.search(audio_queries, k)
    audio_to_melody_search_time = time.time() - audio_to_melody_start_time

    # Calculate intersection rate
    intersection_rates = []
    for i, (audio_audio_idx, audio_melody_idx) in enumerate(zip(audio_to_audio_indices, audio_to_melody_indices)):
        # Check if query vector index is included in results
        audio_match = i in audio_audio_idx
        melody_match = i in audio_melody_idx

        # Consider it an intersection if both results contain the query vector index
        intersection_rate = 1 if melody_match else 0
        intersection_rates.append(intersection_rate)

    average_intersection_rate = np.mean(intersection_rates)

    return {
        'k': k,
        'audio_to_audio_search_time': audio_to_audio_search_time,
        'audio_to_melody_search_time': audio_to_melody_search_time,
        'average_intersection_rate': average_intersection_rate,
    }

def save_index(index, index_path):
    """
    Save FAISS index to specified path.

    Args:
        index (faiss.Index): Index to be saved.
        index_path (str): File path to save the index.
    """
    faiss.write_index(index, index_path)
    print(f"Index saved to: {index_path}")

if __name__ == "__main__":
    # Load data
    melody_data = np.load('./Multimodal_Alignment_npy/musiccaps_melody_362_trimmed.npy')
    audio_data = np.load('./Multimodal_Alignment_npy/musiccaps_audio_362_trimmed.npy')
    audio_queries = np.load('./Multimodal_Alignment_npy/musiccaps_audio_362_trimmed.npy')

    # Read JSON parameter file
    with open('./save_faiss_params.json', 'r') as f:
        params_list = json.load(f)

    # Create a list to store index information
    index_info_list = []

    for i, params in enumerate(params_list, 1):
        index_type = params['index_type'].upper()
        k = int(params['k'])

        if index_type == "HNSW":
            M = int(params['M'])
            efConstruction = int(params['efConstruction'])
            efSearch = int(params['efSearch'])

            print(f"Building melody HNSW index (parameter set {i})...")
            melody_index = build_hnsw_index(melody_data, M=M, efConstruction=efConstruction)
            melody_index.hnsw.efSearch = efSearch

            index_info = f"HNSW, M: {M}, efConstruction: {efConstruction}, efSearch: {efSearch}"
            file_suffix = f"hnsw_M{M}_efC{efConstruction}_efS{efSearch}"

        elif index_type == "IVF":
            nlist = int(params['nlist'])
            nprobe = int(params['nprobe'])

            print(f"Building melody IVF index (parameter set {i})...")
            melody_index = build_ivf_index(melody_data, nlist=nlist)
            melody_index.nprobe = nprobe

            index_info = f"IVF, nlist: {nlist}, nprobe: {nprobe}"
            file_suffix = f"ivf_nlist{nlist}_nprobe{nprobe}"

        else:
            print(f"Invalid index type: {index_type}. Skipping parameter set {i}.")
            continue

        # Perform validation
        print(f"Performing validation with k={k}...")
        validation_result = evaluate_index(melody_index, melody_index, audio_queries, k=k)

        # Output validation results
        print(f"\nIndex type: {index_info}, k: {k}")
        print(f"Average intersection rate: {validation_result['average_intersection_rate']:.4f}")
        print(f"Audio to audio search time: {validation_result['audio_to_audio_search_time']:.6f} seconds")
        print(f"Audio to melody search time: {validation_result['audio_to_melody_search_time']:.6f} seconds")

        # Save index
        save_path = './OverlapRate_Experiments_result'
        os.makedirs(save_path, exist_ok=True)

        melody_index_path = os.path.join(save_path, f'musiccaps_melody_{file_suffix}.faiss')

        print(f"Saving melody {index_type} index...")
        save_index(melody_index, melody_index_path)

        # Add index information to list
        index_info_list.append({
            "index_type": index_type,
            "file_name": f'musiccaps_melody_{file_suffix}.faiss',
            "average_intersection_rate": validation_result['average_intersection_rate'],
            "parameters": params
        })

        print(f"Parameter set {i} processing completed.\n")

    # Save index information to JSON file
    index_info_json_path = os.path.join(save_path, 'index_info.json')
    with open(index_info_json_path, 'w') as f:
        json.dump(index_info_list, f, indent=2)

    print(f"Index information saved to: {index_info_json_path}")