# Awesome-Music-Generation
This repository is the implementation of music generation model MMGen.


## FAISS Experiments

This module optimizes FAISS (Facebook AI Similarity Search) indexes for audio and melody data in the MMGen music generation model.

### Scripts

1. `find_best_faiss_params.py`: Evaluates FAISS index types (Flat, IVF, HNSW) and their parameters.
   - Calculates intersection rates and search times
   - Outputs: `find_best_faiss_params.json`

2. `save_faiss_params.py`: Builds and saves FAISS indexes using selected parameters.
   - Constructs HNSW and IVF indexes
   - Outputs: 
     - Indexes in `./OverlapRate_Experiments_result/`
     - `index_info.json` with performance metrics

### Data

Required files (in `./Multimodal_Alignment_npy/`):  
example
- `musiccaps_melody_362_trimmed.npy`
- `musiccaps_audio_362_trimmed.npy`

### Usage

1. Run `python find_best_faiss_params.py` to evaluate parameters
2. **Manually select** optimal parameters from `find_best_faiss_params.json`
3. Create `save_faiss_params.json` with the selected parameters
4. Run `python save_faiss_params.py` to build and save indexes
