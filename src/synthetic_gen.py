import pandas as pd
import warnings
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

warnings.filterwarnings('ignore')

def generate_synthetic_data(real_data, num_rows=None, epochs=10):
    """
    Trains a CTGAN model on the provided real data and generates a synthetic equivalent.
    """
    if num_rows is None:
        num_rows = len(real_data)
        
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    
    # Using low epochs default for faster execution during development
    synthesizer = CTGANSynthesizer(metadata, epochs=epochs)
    synthesizer.fit(real_data)
    
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    return synthetic_data
