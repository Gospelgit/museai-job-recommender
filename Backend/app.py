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
        
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in directory: {os.listdir('.')}")
        
        if not os.path.exists(module_path):
            logger.error(f"Job recommender module not found at {module_path}")
            return None
            
        logger.info(f"Loading module from {module_path}")
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

# Load the module when the app starts
job_recommender = load_job_recommender()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if job_recommender is None:
        return jsonify({"status": "error", "message": "Job recommender module not loaded"}), 500
    
    return jsonify({"status": "ok", "message": "MuseAI Job Recommender service is healthy"})

@app.route('/api/search', methods=['POST'])
def search_jobs():
    """Endpoint to search for jobs based on user input."""
    if job_recommender is None:
        return jsonify({"error": "Job recommender service is not available"}), 500
    
    try:
        # Get request data
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        job_title = data.get('job_title')
        skills = data.get('skills')
        experience = data.get('experience', 0)
        
        # Validate required parameters
        if not job_title or not skills:
            return jsonify({"error": "Job title and skills are required"}), 400
        
        # Log the request
        logger.info(f"Processing job search request - Title: {job_title}, Experience: {experience} years")
        
        # Measure execution time
        start_time = time.time()
        
        # Call the job recommender function
        result = job_recommender.handle_web_request(job_title, skills, experience)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"Job search completed in {execution_time:.2f} seconds")
        
        # Add execution time to result
        if isinstance(result, dict) and "error" not in result:
            result["processing_time"] = f"{execution_time:.2f} seconds"
        
        # Return the results with CORS headers
        response = jsonify(result)
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "An error occurred while processing your request",
            "details": str(e)
        }), 500

@app.route('/api/info', methods=['GET'])
def get_info():
    """Return information about the API."""
    return jsonify({
        "name": "MuseAI Global Job Matching Tool API",
        "version": "1.0.0",
        "description": "API for finding jobs that match your skills and experience",
        "endpoints": [
            {
                "path": "/api/health",
                "method": "GET",
                "description": "Health check endpoint"
            },
            {
                "path": "/api/search",
                "method": "POST",
                "description": "Search for job recommendations",
                "parameters": {
                    "job_title": "Title of the job you're looking for",
                    "skills": "Your skills and qualifications",
                    "experience": "Years of experience (optional, default: 0)"
                }
            },
            {
                "path": "/api/info",
                "method": "GET",
                "description": "Get API information"
            }
        ]
    })

if __name__ == '__main__':
    # Get port from environment variable for Render compatibility
    port = int(os.environ.get('PORT', 5000))
    # Make sure to bind to 0.0.0.0 for Render
    app.run(host='0.0.0.0', port=port)
