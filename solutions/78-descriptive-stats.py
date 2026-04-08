import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    data = np.array(data)
    stats = {}
    stats["mean"] = np.mean(data)
    stats["median"] = np.median(data)
    
    values, counts = np.unique(data, return_counts=True)
    stats["mode"] = values[np.argmax(counts)]
    stats["variance"] = np.std(data)**2
    stats["standard_deviation"] = np.std(data)
    stats["25th_percentile"] = np.quantile(data, 0.25)
    stats["50th_percentile"] = np.quantile(data, 0.5)
    stats["75th_percentile"] = np.quantile(data, 0.75)
    stats["interquartile_range"] = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    return stats