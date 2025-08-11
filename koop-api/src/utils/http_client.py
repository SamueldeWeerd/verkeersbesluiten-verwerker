import time
import logging
from typing import Optional, Dict, Any
import requests
from requests import Response

class RateLimitedClient:
    """
    HTTP client with built-in rate limiting and retry functionality.
    Handles 429 responses adaptively and implements exponential backoff.
    """
    
    def __init__(
        self,
        settings = None  # Will be injected
    ):
        """
        Initialize the rate-limited client with settings.
        If no settings provided, will use get_settings() to load them.
        """
        from src.config.settings import get_settings
        self._settings = settings or get_settings()
        self._request_delay = self._settings.rate_limit.request_delay
        self._max_retries = self._settings.rate_limit.max_retries
        self._retry_delay_multiplier = self._settings.rate_limit.delay_multiplier
        self._successful_requests_to_reset = self._settings.rate_limit.successful_requests_to_reset
        self._timeout = self._settings.rate_limit.request_timeout
        self._connect_timeout = self._settings.rate_limit.connect_timeout
        self._max_retry_delay = self._settings.rate_limit.max_retry_delay
        
        # Rate limiting state
        self._rate_limited = False
        self._last_request_time = 0
        self._successful_requests = 0
        
    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Optional[Response]:
        """
        Make a rate-limited GET request.
        
        Args:
            url: The URL to request
            params: Optional query parameters
            timeout: Optional request timeout (overrides default)
            **kwargs: Additional arguments passed to requests.get()
            
        Returns:
            Response object if successful, None if all retries failed
        """
        return self._make_request(url, params, timeout, **kwargs)
    
    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Optional[Response]:
        """Internal method to make the actual HTTP request with rate limiting."""
        for attempt in range(self._max_retries + 1):
            try:
                # Apply rate limiting delay if needed
                self._apply_rate_limiting_delay(attempt)
                
                # Make the request
                logging.debug(f"🌐 Requesting: {url}")
                response = requests.get(
                    url,
                    params=params,
                    timeout=(self._connect_timeout, timeout or self._timeout),  # (connect_timeout, read_timeout)
                    **kwargs
                )
                self._last_request_time = time.time()
                
                # Handle rate limiting response
                if response.status_code == 429:
                    self._handle_rate_limit(response)
                    continue
                
                # Handle successful response
                if response.ok:
                    self._handle_success()
                else:
                    self._handle_failure(response)
                
                return response
                
            except requests.RequestException as e:
                self._handle_error(e, attempt, url)
                if attempt == self._max_retries:
                    logging.error(f"❌ All {self._max_retries + 1} retries failed for {url}")
                    return None
        
        return None
    
    def _apply_rate_limiting_delay(self, attempt: int) -> None:
        """Apply appropriate delays for rate limiting and retries."""
        if attempt > 0:
            # Exponential backoff for retries with maximum delay cap
            delay = self._request_delay * (self._retry_delay_multiplier ** (attempt - 1))
            delay = min(delay, self._max_retry_delay)
            logging.info(f"⏳ Retry attempt {attempt}: waiting {delay:.1f} seconds...")
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                logging.info("⚠️ Retry interrupted by user")
                raise
        elif self._rate_limited:
            # Regular rate limiting delay
            time_since_last = time.time() - self._last_request_time
            if time_since_last < self._request_delay:
                sleep_time = self._request_delay - time_since_last
                logging.info(f"⏳ Rate limiting active: waiting {sleep_time:.1f} seconds...")
                try:
                    time.sleep(sleep_time)
                except KeyboardInterrupt:
                    logging.info("⚠️ Rate limiting interrupted by user")
                    raise
    
    def _handle_rate_limit(self, response: Response) -> None:
        """Handle 429 Too Many Requests response."""
        if not self._rate_limited:
            logging.warning("⚠️ First 429 error detected - rate limiting now active")
            self._rate_limited = True
        self._successful_requests = 0
        
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            wait_time = int(retry_after)
            logging.warning(f"⚠️ Rate limited (429). Waiting {wait_time} seconds as per Retry-After header...")
            time.sleep(wait_time)
        else:
            logging.warning("⚠️ Rate limited (429). Using exponential backoff...")
    
    def _handle_success(self) -> None:
        """Handle successful response."""
        logging.info("✅ Request successful")
        self._successful_requests += 1
        
        if (self._rate_limited and 
            self._successful_requests >= self._successful_requests_to_reset):
            logging.info(f"🚀 Rate limiting disabled after {self._successful_requests_to_reset} successful requests")
            self._rate_limited = False
            self._successful_requests = 0
    
    def _handle_failure(self, response: Response) -> None:
        """Handle non-ok response."""
        logging.warning(f"⚠️ Request failed: {response.status_code}")
        self._successful_requests = 0
    
    def _handle_error(self, error: Exception, attempt: int, url: str) -> None:
        """Handle request exception."""
        if isinstance(error, requests.exceptions.Timeout):
            logging.warning(f"⏰ Request timeout (attempt {attempt + 1}/{self._max_retries + 1}): {error}")
        elif isinstance(error, requests.exceptions.ConnectionError):
            logging.warning(f"🔌 Connection error (attempt {attempt + 1}/{self._max_retries + 1}): {error}")
        elif "redirect" in str(error).lower():
            logging.warning(f"🔄 Redirect error (attempt {attempt + 1}/{self._max_retries + 1}): {error}")
        else:
            logging.error(f"❌ Request error (attempt {attempt + 1}/{self._max_retries + 1}): {error}")
        
        self._successful_requests = 0