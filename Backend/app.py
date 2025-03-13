import streamlit as st
import pandas as pd
import sys
import os
import time
import tempfile
import re
import Job_recommend

# Set page config first (should be at the very top)
st.set_page_config(
    page_title="MuseAI Job Recommender",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the current directory to the path so we can import modules
sys.path.append(os.path.abspath("."))

# Cache the job recommendation function to avoid redundant processing
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_job_recommendations(job_title, skills, experience, uploaded_file=None):
    """Get job recommendations with caching"""
    try:
        # Import the job recommender module
        from Job_recommend import streamlit_recommendation
        
        # Call the recommendation function
        return streamlit_recommendation(job_title, skills, experience, uploaded_file)
    except ImportError as e:
        st.error(f"Error importing Job_recommend module: {str(e)}")
        st.info("Make sure the Job_recommend.py file is in the same directory as app.py")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return pd.DataFrame()

# App title and description
st.title("MuseAI Job Recommender")
st.markdown("### Come make we help you find global jobs using AI")

# Sidebar for inputs - simplified to just job title, skills, and experience
with st.sidebar:
    try:
        st.image("https://raw.githubusercontent.com/Gospelgit/museai-job-recommender/main/logo.png", use_column_width=True)
    except:
        st.title("MuseAI")
    
    st.title("Your Profile")
    
    # User inputs in a form
    with st.form("user_profile_form"):
        # Desired job title or role
        desired_role = st.text_input(
            "Wetin be the job you dey find?",
            placeholder="e.g., Data Scientist, Software Engineer",
            help="Wetin be the job you dey find?"
        )
        
        # Skills input
        skills = st.text_area(
            "Your skills (separate am with comma)",
            placeholder="e.g., Python, Data Analysis, Communication",
            help="Wetin be your skills, separate am with commas"
        )
        
        # Experience level
        experience = st.slider(
            "How many years you don work?",
            min_value=0,
            max_value=20,
            value=2,
            help="How many years experience you get"
        )
        
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
            
            # Get recommendations
            recommendations = get_job_recommendations(
                desired_role, 
                skills,
                experience
            )
            
            processing_time = time.time() - start_time
            
            # Display recommendations
            if recommendations is None or recommendations.empty:
                st.info("We no see any job wey match. Try change your skills or job title small.")
            else:
                # Store recommendations in session state for persistence
                st.session_state.recommendations = recommendations
                
                st.success(f"We don find {len(recommendations)} jobs for you in {processing_time:.1f} seconds!")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Match Score", f"{recommendations['match_score'].mean():.1f}%")
                with col2:
                    st.metric("Jobs wer we find", len(recommendations))
                with col3:
                    st.metric("Time wer e take us to find the jobs", f"{processing_time:.1f}s")
                
                # Display each job
                st.markdown("## Recommended Jobs")
                for i, (_, job) in enumerate(recommendations.iterrows(), 1):
                    # Create expandable card for each job
                    with st.expander(f"{i}. {job['title']} - {job['match_score']:.1f}% Match"):
                        # Two columns - job details and match info
                        col1, col2 = st.columns([7, 3])
                        
                        with col1:
                            st.markdown(f"### {job['title']}")
                            st.markdown(f"**Company**: {job.get('company', 'Unknown Company')}")
                            st.markdown(f"**Location**: {job.get('location', 'Not specified')}")
                            
                            # Display salary info if available
                            salary_display = job.get('salary', 'Not specified')
                            st.markdown(f"**Salary/Compensation**: {salary_display}")
                            
                            # Display description preview if available
                            if 'description' in job and job['description']:
                                st.markdown("**Description**:")
                                description_preview = job['description'][:500] + "..." if len(job['description']) > 500 else job['description']
                                st.markdown(description_preview)
                            
                            # Link to apply
                            if 'url' in job and job['url']:
                                st.markdown(f"[Apply on Original Website]({job['url']})")
                        
                        with col2:
                            st.markdown("**Match Details**")
                            st.markdown(f"**Overall Match**: {job['match_score']:.1f}%")
                            
                            # Skills match visualization
                            if 'required_skills' in job and isinstance(job['required_skills'], list) and job['required_skills']:
                                st.markdown("**Skills Match**:")
                                user_skills_set = {skill.strip().lower() for skill in skills.split(',') if skill.strip()}
                                
                                for skill in job['required_skills']:
                                    skill_lower = skill.lower()
                                    if any(user_skill in skill_lower for user_skill in user_skills_set):
                                        st.markdown(f"✅ {skill}")
                                    else:
                                        st.markdown(f"❌ {skill}")
                            
                            # Experience match
                            required_exp = job.get('experience', 0)
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
    try:
        st.image("https://raw.githubusercontent.com/Gospelgit/museai-job-recommender/main/dashboard_preview.png", 
                caption="Sample recommendation dashboard preview", use_column_width=True)
    except:
        # If image fails to load, just show a placeholder
        st.info("Fill in your details in the sidebar to get started!")

# Footer
st.markdown("---")
st.markdown("Powered by MuseAI - Transforming job search with AI")
st.markdown("[GitHub Repository](https://github.com/Gospelgit/museai-job-recommender)")
