# Importing the patch module first - before any other imports
import patch_imports

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
CORS(app, 
     resources={r"/api/*": {"origins": ["https://aimuse.netlify.app", "http://localhost:3000"]}}, 
     supports_credentials=True)

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

# Add this after the job_recommender = load_job_recommender() line in your app.py

# Simple fallback if job_recommender fails to load
if job_recommender is None:
    @app.route('/api/search', methods=['POST'])
    def search_jobs_fallback():
        """Fallback endpoint that works without the job recommender module."""
        try:
            # Get request data
            data = request.json
            
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            # Extract parameters for logging
            job_title = data.get('job_title', 'Unknown')
            skills = data.get('skills', [])
            experience = data.get('experience', 0)
            
            # Log the request
            logger.info(f"Processing job search request (FALLBACK MODE) - Title: {job_title}, Experience: {experience} years")
            
            # Return mock data for testing frontend-backend connection
            return jsonify({
                "status": "connected",
                "message": "Job recommender module not available, but API is working - connection successful!",
                "request_received": {
                    "job_title": job_title,
                    "skills": skills,
                    "experience": experience
                },
                "recommendations": [
                    {
                        "jobTitle": "Test Software Developer",
                        "company": "Example Tech Inc",
                        "location": "Remote",
                        "salary": "$80,000 - $120,000",
                        "experience": 2,
                        "score": 95,
                        "url": "https://example.com/job1",
                        "responsibilities": "Building and maintaining web applications. Testing and debugging. Collaborating with team members."
                    },
                    {
                        "jobTitle": "Test Data Analyst",
                        "company": "Data Insights LLC",
                        "location": "New York",
                        "salary": "$70,000 - $90,000",
                        "experience": 1,
                        "score": 85,
                        "url": "https://example.com/job2",
                        "responsibilities": "Analyzing data sets. Creating visualizations. Generating reports and insights."
                    },
                    {
                        "jobTitle": "Test Marketing Specialist",
                        "company": "Global Marketing Group",
                        "location": "Chicago",
                        "salary": "$65,000 - $85,000",
                        "experience": 3,
                        "score": 75,
                        "url": "https://example.com/job3",
                        "responsibilities": "Developing marketing campaigns. Managing social media. Analyzing campaign performance."
                    }
                ]
            })
            
        except Exception as e:
            logger.error(f"Error processing request in fallback handler: {str(e)}")
            return jsonify({
                "error": "An error occurred while processing your request",
                "details": str(e)
            }), 500

# Also add this route to explicitly check connection status
@app.route('/api/connection-check', methods=['GET'])
def check_connection():
    """Simple endpoint to verify frontend-backend connection."""
    return jsonify({
        "status": "connected",
        "message": "Backend API is reachable and responding correctly"
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if job_recommender is None:
        return jsonify({"status": "warning", "message": "Job recommender module not loaded, but service is running"}), 200
    
    return jsonify({"status": "ok", "message": "MuseAI Job Recommender service is healthy"})

@app.route('/api/search', methods=['POST'])
def search_jobs():
    """Endpoint to search for jobs based on user input."""
    if job_recommender is None:
        # Instead of returning an error, use the fallback implementation
        logger.warning("Using fallback job search implementation - job_recommender module not available")
        return search_jobs_fallback()
    
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
    port = int(os.environ.get('PORT', 10000)) 
    app.run(host='0.0.0.0', port=port)
