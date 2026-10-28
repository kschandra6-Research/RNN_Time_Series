"""
Backend Package for NeuroCast
Concept & Design by K.S.Chandra 2026
"""
from .engine import train_rnn_engine, generate_synthetic_series, PredictionResult, NUMPY_AVAILABLE, TORCH_AVAILABLE
from .parser import inspect_file_columns, find_workspace_data_files
