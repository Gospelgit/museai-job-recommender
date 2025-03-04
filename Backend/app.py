from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback
import importlib.util
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('museai_job_recommender')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Import the job recommender module
def load_job_recommender():
    """Load the job recommender module dynamically."""
    try:
        # Path to your original job recommender module
        module_path = "job_recommender.py"
        
        if not os.path.exists(module_path):
            logger.error(f"Job recommender module not found at {module_path}")
            return None
            
        spec = importlib.util.spec_from_file_location("job_recommender", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check if it has the required function
        if not hasattr(module, "handle_web_request"):
            logger.error("Module does not have handle_web_request function")
            return None
            
        logger.info("Job recommender module loaded successfully")
        return module
    except Exception as e:
        logger.error(f"Error loading job recommender module: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# More API code...
