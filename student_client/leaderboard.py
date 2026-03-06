"""
Leaderboard Functions for Student Client

Provides functions to retrieve leaderboard data from the gym server.
"""

import logging
from typing import Dict, Any, List, Union
import pandas as pd
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_leaderboard_score(
    user_token: str,
    server_url: str = "http://rlchallenge.orailix.com",
    limit: int = 100,
    return_dataframe: bool = True
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Retrieve the user's leaderboard score as a pandas DataFrame or dictionary.
    
    This function returns the user's aggregated leaderboard statistics in the same format
    as the global leaderboard, but only for the specified user. It shows the user's
    performance over their last 100 completed episodes (or fewer if they have less).
    
    Args:
        user_token: User authentication token to retrieve score for
        server_url: Gym server URL
        limit: Maximum number of episodes to consider for calculations (default: 100)
        return_dataframe: If True, return as DataFrame. If False, return as dict (default: False)
        
    Returns:
        Dict[str, Any] or pd.DataFrame: User's leaderboard score with keys/columns:
            - user_token: User identifier
            - total_episodes: Total number of episodes considered
            - total_reward: Total cumulative reward
            - avg_reward: Average reward per episode
            - best_reward: Best single episode reward
            - maintenance_actions: Total maintenance actions taken
            - avg_episode_length: Average episode length in steps
            - failure_rate: Rate of episode failures (0.0 to 1.0)
            - last_episode_date: Date of last episode
            
    Raises:
        ValueError: If the request fails or data cannot be retrieved
        ConnectionError: If unable to connect to the server
        
    Example:
        >>> from student_client.leaderboard import get_leaderboard_score
        >>> 
        >>> # Get as dictionary (default) - recommended for most use cases
        >>> score = get_leaderboard_score(user_token="student_user")
        >>> print(f"Total reward: {score['total_reward']}")
        >>> print(f"Average reward: {score['avg_reward']}")
        >>> 
        >>> # Get as DataFrame for compatibility with existing code
        >>> df = get_leaderboard_score(user_token="student_user", return_dataframe=True)
        >>> print(df)
        
        Note: This function returns the user's aggregated leaderboard score,
        similar to what appears in the global leaderboard but for a single user.
        Users can only see their own performance data.
    """
    try:
        # Create HTTP client
        client = httpx.Client(
            base_url=server_url.rstrip('/'),
            timeout=30.0
        )
        
        # Call user-specific score endpoint to get the metrics
        response = client.get(
            f"/api/v1/user/{user_token}/score",
            params={
                "limit": limit
            }
        )
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse the JSON response
        score_data = response.json()
        
        # Check if we got valid data with metrics
        if not score_data or 'metrics' not in score_data or not score_data['metrics']:
            logger.warning("Score data is empty or invalid format")
            if return_dataframe:
                return pd.DataFrame()
            else:
                return {}
        
        # Extract the metrics and format as a leaderboard entry
        metrics = score_data['metrics']
        
        # Create a leaderboard-style entry for this user
        leaderboard_entry = {
            'user_token': user_token,
            'total_episodes': metrics.get('total_episodes', 0),
            'total_reward': metrics.get('total_reward', 0.0),
            'avg_reward': metrics.get('avg_reward', 0.0),
            'best_reward': metrics.get('best_reward', 0.0),
            'maintenance_actions': metrics.get('total_maintenance', 0),
            'avg_episode_length': metrics.get('avg_steps', 0.0),
            'failure_rate': metrics.get('failure_rate', 0.0),
            'last_episode_date': metrics.get('last_episode_date', None)
        }
        
        logger.info(f"Retrieved leaderboard score for user {user_token}: {leaderboard_entry}")
        
        # Return in requested format
        if return_dataframe:
            return pd.DataFrame([leaderboard_entry])
        else:
            return leaderboard_entry
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to retrieve user score: HTTP {e.response.status_code}")
        error_msg = f"Could not retrieve user score: HTTP {e.response.status_code}"
        if e.response.status_code == 401:
            error_msg += " - Unauthorized: Invalid user token"
        elif e.response.status_code == 404:
            error_msg += " - User not found or no data available"
        raise ValueError(error_msg)
        
    except httpx.ConnectError as e:
        logger.error(f"Failed to connect to server: {e}")
        raise ConnectionError(f"Could not connect to server at {server_url}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Failed to process score data: {str(e)}")
        raise ValueError(f"Score retrieval failed: {str(e)}")

