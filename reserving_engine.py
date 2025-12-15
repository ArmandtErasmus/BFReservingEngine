import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from abc import ABC, abstractmethod
import warnings


class LossTriangle:
    """
    Represents a loss triangle with cumulative or incremental claims data.
    
    This class provides functionality to work with loss triangles, including
    conversion between cumulative and incremental formats, and validation
    of triangle structure.
    """
    
    def __init__(self, data: np.ndarray, is_cumulative: bool = True):
        """
        Initialise loss triangle.
        
        Args:
            data: Triangular array of claims data (n_periods x n_periods)
            is_cumulative: Whether data is cumulative (True) or incremental (False)
        
        Raises:
            ValueError: If data is not a valid triangular array
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError("Data must be a square array")
        
        self.data = data.astype(float)
        self.is_cumulative = is_cumulative
        self.n_periods = data.shape[0]
        
        # Validate triangle structure
        self._validate_triangle()
    
    def _validate_triangle(self) -> None:
        """Validate that the triangle has the correct structure."""
        for i in range(self.n_periods):
            for j in range(self.n_periods - i):
                if np.isnan(self.data[i, j]) and j < self.n_periods - i - 1:
                    warnings.warn(
                        f"Missing data at position ({i}, {j}) in observed triangle"
                    )
    
    def to_cumulative(self) -> np.ndarray:
        """
        Convert incremental triangle to cumulative.
        
        Returns:
            Cumulative loss triangle as numpy array
        """
        if self.is_cumulative:
            return self.data.copy()
        
        cumulative = np.zeros_like(self.data)
        for i in range(self.n_periods):
            for j in range(self.n_periods - i):
                if j == 0:
                    cumulative[i, j] = self.data[i, j]
                else:
                    cumulative[i, j] = cumulative[i, j-1] + self.data[i, j]
        
        return cumulative
    
    def to_incremental(self) -> np.ndarray:
        """
        Convert cumulative triangle to incremental.
        
        Returns:
            Incremental loss triangle as numpy array
        """
        if not self.is_cumulative:
            return self.data.copy()
        
        incremental = np.zeros_like(self.data)
        for i in range(self.n_periods):
            for j in range(self.n_periods - i):
                if j == 0:
                    incremental[i, j] = self.data[i, j]
                else:
                    incremental[i, j] = self.data[i, j] - self.data[i, j-1]
        
        return incremental
    
    def get_latest_observed(self) -> np.ndarray:
        """
        Get the latest observed cumulative claims for each accident period.
        
        Returns:
            Array of latest observed claims for each accident period
        """
        cumulative = self.to_cumulative()
        latest = np.array([
            cumulative[i, self.n_periods - i - 1] 
            for i in range(self.n_periods)
        ])
        return latest
    
    def to_dataframe(self, 
                     acc_period_labels: Optional[List[str]] = None,
                     dev_period_labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert triangle to pandas DataFrame for easier visualisation.
        
        Args:
            acc_period_labels: Labels for accident periods
            dev_period_labels: Labels for development periods
        
        Returns:
            DataFrame representation of the triangle
        """
        if acc_period_labels is None:
            acc_period_labels = [f"Acc {i}" for i in range(self.n_periods)]
        if dev_period_labels is None:
            dev_period_labels = [f"Dev {j}" for j in range(self.n_periods)]
        
        return pd.DataFrame(
            self.data,
            columns=dev_period_labels,
            index=acc_period_labels
        )


class ReservingMethod(ABC):
    """
    Abstract base class for reserving methods.
    
    This class defines the interface that all reserving methods must implement,
    ensuring consistency and enabling polymorphic usage.
    """
    
    @abstractmethod
    def calculate_reserves(self, triangle: LossTriangle) -> Dict:
        """
        Calculate reserves using the specific method.
        
        Args:
            triangle: LossTriangle object containing claims data
        
        Returns:
            Dictionary containing reserve estimates and related metrics
        """
        pass


class ChainLadderMethod(ReservingMethod):
    """
    Implementation of the Chain Ladder reserving method.
    
    The Chain Ladder method projects future claims based on observed
    development patterns, assuming stability of development factors
    across accident periods.
    """
    
    def __init__(self):
        """Initialise Chain Ladder method."""
        self.development_factors = None
        self.projected_triangle = None
        self.incremental_projected = None
    
    def calculate_development_factors(self, triangle: LossTriangle) -> np.ndarray:
        """
        Calculate age-to-age development factors.
        
        Development factors are computed as weighted averages across
        all available accident periods.
        
        Args:
            triangle: LossTriangle object
        
        Returns:
            Array of development factors for each development period
        """
        cumulative = triangle.to_cumulative()
        n = triangle.n_periods
        factors = np.ones(n - 1)  # Initialise to 1.0
        
        for j in range(n - 1):
            numerator = 0.0
            denominator = 0.0
            
            for i in range(n - j - 1):
                if cumulative[i, j] > 0 and not np.isnan(cumulative[i, j]):
                    if not np.isnan(cumulative[i, j + 1]):
                        numerator += cumulative[i, j + 1]
                        denominator += cumulative[i, j]
            
            if denominator > 0:
                factors[j] = numerator / denominator
            else:
                # If no data available, use 1.0 (no development)
                factors[j] = 1.0
        
        return factors
    
    def project_triangle(self, triangle: LossTriangle) -> np.ndarray:
        """
        Project the lower triangle using development factors.
        
        Args:
            triangle: LossTriangle object
        
        Returns:
            Projected cumulative triangle including lower triangle
        """
        cumulative = triangle.to_cumulative()
        n = triangle.n_periods
        projected = cumulative.copy()
        factors = self.calculate_development_factors(triangle)
        
        for i in range(n):
            latest_observed_idx = n - i - 1
            
            if latest_observed_idx < n - 1:
                current_value = cumulative[i, latest_observed_idx]
                
                # Project forward using development factors
                for j in range(latest_observed_idx + 1, n):
                    if j - 1 < len(factors):
                        current_value *= factors[j - 1]
                        projected[i, j] = current_value
        
        return projected
    
    def calculate_reserves(self, triangle: LossTriangle) -> Dict:
        """
        Calculate Chain Ladder reserves.
        
        Args:
            triangle: LossTriangle object
        
        Returns:
            Dictionary containing:
                - ultimates: Ultimate claims for each accident period
                - reserves: Reserve estimates for each accident period
                - total_reserve: Total reserve across all accident periods
                - development_factors: Age-to-age development factors
                - projected_triangle: Full projected cumulative triangle
                - incremental_projected: Projected incremental triangle
        """
        cumulative = triangle.to_cumulative()
        projected = self.project_triangle(triangle)
        self.development_factors = self.calculate_development_factors(triangle)
        self.projected_triangle = projected
        
        n = triangle.n_periods
        ultimates = projected[:, -1]
        latest_observed = triangle.get_latest_observed()
        reserves = ultimates - latest_observed
        
        # Calculate incremental projected triangle
        incremental_projected = np.zeros_like(projected)
        for i in range(n):
            for j in range(n):
                if j == 0:
                    incremental_projected[i, j] = projected[i, j]
                else:
                    incremental_projected[i, j] = (
                        projected[i, j] - projected[i, j-1]
                    )
        
        self.incremental_projected = incremental_projected
        
        return {
            'ultimates': ultimates,
            'reserves': reserves,
            'total_reserve': np.sum(reserves),
            'development_factors': self.development_factors,
            'projected_triangle': projected,
            'incremental_projected': incremental_projected,
            'latest_observed': latest_observed
        }


class BornhuetterFergusonMethod(ReservingMethod):
    """
    Implementation of the Bornhuetter-Ferguson reserving method.
    
    The Bornhuetter-Ferguson method combines prior expectations of
    ultimate claims with observed development patterns, providing
    a compromise between Chain Ladder and prior estimates.
    """
    
    def __init__(self, prior_ultimates: np.ndarray):
        """
        Initialise Bornhuetter-Ferguson method.
        
        Args:
            prior_ultimates: Array of prior ultimate estimates for each accident period
        
        Raises:
            ValueError: If prior_ultimates length doesn't match triangle dimensions
        """
        self.prior_ultimates = np.array(prior_ultimates, dtype=float)
        self.chain_ladder = ChainLadderMethod()
    
    def calculate_reserves(self, triangle: LossTriangle) -> Dict:
        """
        Calculate Bornhuetter-Ferguson reserves.
        
        Args:
            triangle: LossTriangle object
        
        Returns:
            Dictionary containing:
                - ultimates: BF ultimate claims for each accident period
                - reserves: BF reserve estimates for each accident period
                - total_reserve: Total BF reserve
                - prior_ultimates: Prior ultimate estimates used
                - chain_ladder_ultimates: CL ultimates for comparison
                - development_ratios: Ratio of observed to CL ultimate
        """
        if len(self.prior_ultimates) != triangle.n_periods:
            raise ValueError(
                f"Prior ultimates length ({len(self.prior_ultimates)}) "
                f"must match triangle periods ({triangle.n_periods})"
            )
        
        # Get Chain Ladder results for development pattern
        cl_result = self.chain_ladder.calculate_reserves(triangle)
        cl_ultimates = cl_result['ultimates']
        latest_observed = triangle.get_latest_observed()
        
        n = triangle.n_periods
        bf_reserves = np.zeros(n)
        bf_ultimates = np.zeros(n)
        development_ratios = np.zeros(n)
        
        for i in range(n):
            if cl_ultimates[i] > 0:
                # Development ratio: proportion of ultimate already observed
                development_ratio = latest_observed[i] / cl_ultimates[i]
                development_ratios[i] = development_ratio
                
                # BF reserve = prior * (1 - development_ratio)
                bf_reserves[i] = self.prior_ultimates[i] * (1 - development_ratio)
                bf_ultimates[i] = latest_observed[i] + bf_reserves[i]
            else:
                # If CL ultimate is zero, use prior directly
                bf_reserves[i] = self.prior_ultimates[i]
                bf_ultimates[i] = self.prior_ultimates[i]
                development_ratios[i] = 0.0
        
        return {
            'ultimates': bf_ultimates,
            'reserves': bf_reserves,
            'total_reserve': np.sum(bf_reserves),
            'prior_ultimates': self.prior_ultimates,
            'chain_ladder_ultimates': cl_ultimates,
            'development_ratios': development_ratios,
            'latest_observed': latest_observed
        }


class ReservingAnalyser:
    """
    Main class for performing comprehensive reserving analysis.
    
    This class provides a unified interface for running multiple
    reserving methods and comparing results.
    """
    
    def __init__(self, triangle: LossTriangle):
        """
        Initialise reserving analyser.
        
        Args:
            triangle: LossTriangle object to analyse
        """
        self.triangle = triangle
        self.results = {}
    
    def run_chain_ladder(self) -> Dict:
        """
        Run Chain Ladder analysis.
        
        Returns:
            Dictionary of Chain Ladder results
        """
        method = ChainLadderMethod()
        self.results['chain_ladder'] = method.calculate_reserves(self.triangle)
        return self.results['chain_ladder']
    
    def run_bornhuetter_ferguson(self, prior_ultimates: np.ndarray) -> Dict:
        """
        Run Bornhuetter-Ferguson analysis.
        
        Args:
            prior_ultimates: Prior ultimate estimates for each accident period
        
        Returns:
            Dictionary of Bornhuetter-Ferguson results
        """
        method = BornhuetterFergusonMethod(prior_ultimates)
        self.results['bornhuetter_ferguson'] = method.calculate_reserves(self.triangle)
        return self.results['bornhuetter_ferguson']
    
    def calculate_incremental_factors(self, ultimates: np.ndarray) -> np.ndarray:
        """
        Calculate incremental development factors.
        
        Incremental factors show the proportion of ultimate claims
        that emerge in each development period.
        
        Args:
            ultimates: Ultimate claims for each accident period
        
        Returns:
            Array of incremental development factors
        """
        incremental = self.triangle.to_incremental()
        n = self.triangle.n_periods
        factors = np.zeros(n)
        
        for j in range(n):
            numerator = 0.0
            denominator = 0.0
            
            for i in range(n - j):
                if not np.isnan(incremental[i, j]):
                    numerator += incremental[i, j]
                if ultimates[i] > 0:
                    denominator += ultimates[i]
            
            if denominator > 0:
                factors[j] = numerator / denominator
        
        return factors
    
    def get_comparison_summary(self) -> pd.DataFrame:
        """
        Get a summary comparison of all methods run.
        
        Returns:
            DataFrame comparing results across methods
        """
        if not self.results:
            return pd.DataFrame()
        
        n = self.triangle.n_periods
        summary_data = {
            'Accident Period': [f"Acc {i}" for i in range(n)],
            'Latest Observed': self.triangle.get_latest_observed()
        }
        
        if 'chain_ladder' in self.results:
            cl = self.results['chain_ladder']
            summary_data['CL Ultimate'] = cl['ultimates']
            summary_data['CL Reserve'] = cl['reserves']
        
        if 'bornhuetter_ferguson' in self.results:
            bf = self.results['bornhuetter_ferguson']
            summary_data['BF Ultimate'] = bf['ultimates']
            summary_data['BF Reserve'] = bf['reserves']
            summary_data['Prior Ultimate'] = bf['prior_ultimates']
        
        return pd.DataFrame(summary_data)


def generate_sample_triangle(n_periods: int = 10,
                            base_claims: float = 10000.0,
                            volatility: float = 0.2,
                            random_seed: Optional[int] = 42) -> np.ndarray:
    """
    Generate sample loss triangle for demonstration purposes.
    
    Args:
        n_periods: Number of accident and development periods
        base_claims: Base level of claims
        volatility: Volatility in claim development
        random_seed: Random seed for reproducibility
    
    Returns:
        Sample cumulative loss triangle
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    triangle = np.zeros((n_periods, n_periods))
    # Typical development factors (decreasing over time)
    development_factors = [
        1.5, 1.3, 1.2, 1.15, 1.1, 1.08, 1.05, 1.03, 1.02, 1.01
    ]
    
    for i in range(n_periods):
        # Initial claims with some randomness
        initial = base_claims * (1 + np.random.normal(0, volatility))
        triangle[i, 0] = max(initial, 0)
        
        # Develop claims over time
        for j in range(1, min(n_periods - i, len(development_factors) + 1)):
            factor = (
                development_factors[j-1] 
                if j-1 < len(development_factors) 
                else 1.01
            )
            # Add some randomness to development
            factor *= (1 + np.random.normal(0, volatility * 0.3))
            triangle[i, j] = triangle[i, j-1] * max(factor, 1.0)
    
    return triangle

