# Importing the patch module first - before any other imports
import patch_imports

from flask import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
import io
import tempfile
from PIL import Image
import re
import json
from datetime import datetime

# Add the current directory to the path so we can import the module
sys.path.append(os.path.abspath("."))

# Use st.cache_resource to ensure the Job_recommend module is only loaded once
@st.cache_resource
def load_job_recommend_module():
    try:
        import Job_recommend
        return Job_recommend
    except ImportError as e:
        st.error(f"Could not import Job_recommend module: {str(e)}")
        st.error("Make sure Job_recommend.py is in the current directory.")
        return None

# Set page config
st.set_page_config(
    page_title="MuseAI Job Recommender",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use st.cache_data for the recommendation function to avoid duplicate processing
@st.cache_data(ttl=3600)  # Cache results for 1 hour
def get_job_recommendations(desired_role, skills, experience, uploaded_file=None):
    """Get job recommendations with caching"""
    Job_recommend = load_job_recommend_module()
    if not Job_recommend:
        return None
    
    # Create user profile
    user_profile = {
        "skills": skills,
        "experience": experience,
        "desired_role": desired_role
    }
    
    # Call the recommendation function
    return Job_recommend.streamlit_recommendation(desired_role, skills, experience, uploaded_file)

# Function to generate a downloadable CSV of results
def generate_csv(recommendations):
    csv = recommendations.to_csv(index=False).encode('utf-8')
    return csv

# App title and description
st.title("MuseAI Job Recommender")
st.markdown("### Find your perfect job match with AI-powered recommendations")

# Sidebar for inputs
with st.sidebar:
    st.image("https://raw.githubusercontent.com/Gospelgit/museai-job-recommender/main/logo.png", use_column_width=True)
    st.title("Your Profile")

    # User inputs in a form
    with st.form("user_profile_form"):
        # Desired job title or role
        desired_role = st.text_input(
            "Wetin be the job you dey find?",
            placeholder="e.g., Data Scientist, Software Engineer",
            help="Enter the job title you want"
        )
        
        # Skills input
        skills = st.text_area(
            "Your skills (separate am with comma)",
            placeholder="e.g., Python, Data Analysis, Communication",
            help="Enter your skills separated by commas"
        )
        
        # Experience level
        experience = st.slider(
            "How many years you don work?",
            min_value=0,
            max_value=20,
            value=2,
            help="Select your years of professional experience"
        )
        
        # File uploader for custom job listings (optional)
        uploaded_file = st.file_uploader(
            "Upload your own job listings Excel file (optional)",
            type=["xlsx", "xls"],
            help="Optional: Upload your own Excel file with job listings"
        )
        
        # Advanced options expander
        with st.expander("Advanced Options", expanded=False):
            use_large_model = st.checkbox(
                "Use larger model (slower but more accurate)",
                value=False,
                help="Enable to use a larger, more accurate AI model"
            )
            
            # Set environment variable based on checkbox
            if use_large_model:
                os.environ['USE_LARGE_MODEL'] = 'true'
            else:
                os.environ['USE_LARGE_MODEL'] = 'false'
        
        # Submit button
        submit_button = st.form_submit_button("Find Jobs")

# Main content area
if submit_button:
    if not skills or not desired_role:
        st.warning("Abeg, put the job you dey find and at least one skill make we fit recommend job for you")
    else:
        with st.spinner("Oya make, make we do small witchy witchy for you..."):
            # Track processing time
            start_time = time.time()
            
            # Process inputs
            skills_list = [skill.strip() for skill in skills.split(",") if skill.strip()]
            
            # Get file path from uploaded file if provided
            upload_path = None
            if uploaded_file:
                # Save uploaded file to temp location
                with open("temp_job_listings.xlsx", "wb") as f:
                    f.write(uploaded_file.getvalue())
                upload_path = "temp_job_listings.xlsx"
            
            # Get recommendations
            recommendations = get_job_recommendations(
                desired_role, 
                skills,
                experience,
                upload_path
            )
            
            processing_time = time.time() - start_time
            
            # Display recommendations
            if recommendations is None:
                st.error("Something don happen. Make sure the Job_recommend module dey properly set up.")
            elif recommendations.empty:
                st.info("We no see any job wey match. Try change your skills small.")
            else:
                # Store recommendations in session state for analytics
                st.session_state.recommendations = recommendations
                
                st.success(f"We don find {len(recommendations)} jobs for you in {processing_time:.1f} seconds!")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Match Score", f"{recommendations['match_score'].mean():.1f}%")
                with col2:
                    max_salary = recommendations['salary'].max() if 'salary' in recommendations.columns else 0
                    st.metric("Highest Salary", f"${max_salary:,.0f}" if max_salary > 0 else "N/A")
                with col3:
                    st.metric("Jobs Found", len(recommendations))
                
                # Add model info
                model_used = "Standard Model"
                if os.environ.get('USE_LARGE_MODEL') == 'true':
                    model_used = "Enhanced Accuracy Model"
                st.caption(f"Powered by: {model_used} • Processing time: {processing_time:.2f}s")
                
                # Allow downloading results as CSV
                st.download_button(
                    "Download Results as CSV",
                    generate_csv(recommendations),
                    "job_recommendations.csv",
                    "text/csv",
                    key="download-csv"
                )
                
                # Add export formats
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Export as JSON"):
                        import json
                        json_str = recommendations.to_json(orient="records")
                        st.download_button(
                            "Download JSON",
                            json_str,
                            "job_recommendations.json",
                            "application/json",
                            key="download-json"
                        )
                with col2:
                    if st.button("Export as Excel"):
                        buffer = pd.ExcelWriter("temp.xlsx", engine='xlsxwriter')
                        recommendations.to_excel(buffer, index=False)
                        buffer.close()
                        
                        with open("temp.xlsx", "rb") as f:
                            excel_data = f.read()
                            
                        st.download_button(
                            "Download Excel",
                            excel_data,
                            "job_recommendations.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download-excel"
                        )
                        
                        # Clean up temp file
                        try:
                            os.remove("temp.xlsx")
                        except:
                            pass
                
                # Display each job
                st.markdown("## Recommended Jobs")
                for i, (_, job) in enumerate(recommendations.iterrows(), 1):
                    # Create expandable card for each job
                    with st.expander(f"{i}. {job['title']} for {job.get('company', 'Unknown')} - {job['match_score']:.1f}% Match"):
                        # Two columns - job details and match info
                        col1, col2 = st.columns([7, 3])
                        
                        with col1:
                            st.markdown(f"### {job['title']}")
                            st.markdown(f"**Company**: {job.get('company', 'Unknown Company')}")
                            st.markdown(f"**Location**: {job.get('location', 'Not specified')}")
                            
                            # Display salary info if available
                            salary_display = "Not specified"
                            if 'salary_info' in job and job['salary_info'] != "Not specified":
                                salary_display = job['salary_info']
                            st.markdown(f"**Salary/Compensation**: {salary_display}")
                            
                            # Display description preview if available
                            if 'description' in job and job['description']:
                                st.markdown("**Description**:")
                                description_preview = job['description'][:500] + "..." if len(job['description']) > 500 else job['description']
                                st.markdown(description_preview)
                            
                            # Link to apply
                            if 'job_url' in job and job['job_url']:
                                st.markdown(f"[Apply on Original Website]({job['job_url']})")
                        
                        with col2:
                            st.markdown("**Match Details**")
                            st.markdown(f"**Overall Match**: {job['match_score']:.1f}%")
                            
                            # Skills match visualization
                            if 'required_skills' in job and job['required_skills']:
                                st.markdown("**Skills Match**:")
                                user_skills_set = {skill.strip().lower() for skill in skills_list}
                                
                                if isinstance(job['required_skills'], list):
                                    job_skills = job['required_skills']
                                else:
                                    # Handle if it's a string
                                    job_skills = [s.strip() for s in str(job['required_skills']).split(',')]
                                
                                for skill in job_skills:
                                    if any(user_skill in skill.lower() for user_skill in user_skills_set):
                                        st.markdown(f"✅ {skill}")
                                    else:
                                        st.markdown(f"❌ {skill}")
                            
                            # Experience match
                            required_exp = job.get('required_experience', 0)
                            if required_exp > 0:
                                exp_match = min(experience / required_exp * 100 if required_exp > 0 else 100, 100)
                                st.markdown(f"**Experience**: {exp_match:.0f}% match")
                                st.progress(exp_match / 100)
                            else:
                                st.markdown("**Experience**: No specific requirement")

else:
    # Welcome content when the app first loads
    st.markdown("""
    ## We sha welcome you to the MuseAI Job Recommender!
    
    This tool na just AI wer fit match your skills and preferences with job opportunities globally.
    
    **If you wan start:**
    1. Fill your profile for the sidebar
    2. Click "Find Jobs" make you see your personalized recommendations
    3. Explore job details and apply to positions that interest you
    
    MuseAI dey analyzes thousands of job listings to find the one wer fit you well well.
    """)
    
    # Placeholder visualization or statistics
    st.image("https://raw.githubusercontent.com/Gospelgit/museai-job-recommender/main/dashboard_preview.png", 
             caption="Sample recommendation dashboard preview", use_column_width=True)

# Footer
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.markdown("Powered by MuseAI - Transforming job search with AI")
with col2:
    st.markdown("[GitHub Repository](https://github.com/Gospelgit/museai-job-recommender)")

# Add analytics section if we have recommendations
if 'recommendations' in st.session_state and not st.session_state.recommendations.empty:
    with st.expander("📊 Job Market Analytics", expanded=False):
        recommendations = st.session_state.recommendations
        
        st.markdown("### Job Market Insights")
        
        # Location analysis
        if 'location' in recommendations.columns:
            st.subheader("Location Distribution")
            location_counts = recommendations['location'].value_counts()
            
            # Create a bar chart for locations
            location_data = pd.DataFrame({
                'Location': location_counts.index,
                'Count': location_counts.values
            })
            st.bar_chart(location_data.set_index('Location'))
        
        # Experience requirements analysis
        if 'required_experience' in recommendations.columns:
            st.subheader("Experience Requirements")
            avg_experience = recommendations['required_experience'].mean()
            st.metric("Average Years Required", f"{avg_experience:.1f} years")
            
            # Create histogram of experience requirements
            st.bar_chart(recommendations['required_experience'].value_counts().sort_index())
        
        # Salary analysis if available
        if 'salary' in recommendations.columns and recommendations['salary'].sum() > 0:
            st.subheader("Salary Distribution")
            
            # Filter out zero values
            valid_salaries = recommendations[recommendations['salary'] > 0]['salary']
            if not valid_salaries.empty:
                st.metric("Average Salary", f"${valid_salaries.mean():,.2f}")
                st.metric("Salary Range", f"${valid_salaries.min():,.0f} - ${valid_salaries.max():,.0f}")
                
                # Create histogram of salaries
                st.bar_chart(
                    pd.cut(valid_salaries, bins=5).value_counts().sort_index()
                )

# Skills gap analysis
if submit_button and 'recommendations' in st.session_state and not st.session_state.recommendations.empty:
    with st.expander("🔍 Skills Gap Analysis", expanded=False):
        recommendations = st.session_state.recommendations
        
        st.markdown("### Skills Gap Analysis")
        st.markdown("This analysis shows skills commonly required for this role that you may want to develop:")
        
        # Extract all skills from job listings
        all_job_skills = []
        if 'required_skills' in recommendations.columns:
            for skills_list in recommendations['required_skills']:
                if isinstance(skills_list, list):
                    all_job_skills.extend(skills_list)
                elif isinstance(skills_list, str):
                    all_job_skills.extend([s.strip() for s in skills_list.split(',')])
        
        # Count skill frequencies
        from collections import Counter
        skill_counter = Counter([skill.lower() for skill in all_job_skills if skill])
        
        # User skills for comparison
        user_skills_set = {skill.strip().lower() for skill in skills.split(',') if skill.strip()}
        
        # Find missing skills (skills in demand but not in user profile)
        missing_skills = {}
        for skill, count in skill_counter.items():
            if not any(user_skill in skill for user_skill in user_skills_set):
                missing_skills[skill] = count
        
        # Show top missing skills
        if missing_skills:
            missing_df = pd.DataFrame({
                'Skill': list(missing_skills.keys()),
                'Frequency': list(missing_skills.values())
            }).sort_values('Frequency', ascending=False).head(10)
            
            st.table(missing_df)
            
            st.markdown("### Recommendations")
            st.markdown("Consider adding these skills to your profile to increase your match rate:")
            for skill in missing_df['Skill'].head(5):
                st.markdown(f"- **{skill.title()}**")
        else:
            st.success("Great job! Your skill set already covers the main requirements for this role.")

# Cleanup temp files on session end
if 'temp_job_listings.xlsx' in os.listdir('.'):
    try:
        if st.session_state.get('_is_running', True):
            # Only set this once
            st.session_state['_is_running'] = False
            
            def cleanup():
                if os.path.exists('temp_job_listings.xlsx'):
                    os.remove('temp_job_listings.xlsx')
            
            import atexit
            atexit.register(cleanup)
    except:
        pass
        
# Add a feedback section
with st.expander("💬 Feedback", expanded=False):
    st.markdown("### Help us improve!")
    st.markdown("Your feedback go help us make this tool better.")
    
    feedback_type = st.selectbox(
        "Feedback Type",
        ["General Feedback", "Bug Report", "Feature Request", "Other"]
    )
    
    feedback_text = st.text_area("Your Feedback")
    
    if st.button("Submit Feedback"):
        if feedback_text:
            # In a real app, you would save this to a database or send via email
            st.success("Thank you for your feedback! We go look into am.")
            
            # For demo purposes, just save to a local file
            try:
                with open("feedback.txt", "a") as f:
                    f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                    f.write(f"Type: {feedback_type}\n")
                    f.write(f"Feedback: {feedback_text}\n")
            except:
                pass
        else:
            st.error("Abeg, write something for the feedback field.")
import Flask, request, jsonify
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
CORS(app)

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
