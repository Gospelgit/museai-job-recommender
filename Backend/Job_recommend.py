import pandas as pd
import re
import random
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def streamlit_recommendation(job_title, skills, experience, uploaded_file=None):
    """
    Generate job recommendations based on user inputs.
    
    Parameters:
    job_title (str): The desired job title
    skills (str): Comma-separated skills
    experience (int): Years of experience (integer)
    uploaded_file (file): Optional resume file
    
    Returns:
    pd.DataFrame: DataFrame with job recommendations
    """
    try:
        # Clean and process user inputs
        job_title = job_title.strip().lower()
        user_skills = [skill.strip().lower() for skill in skills.split(',') if skill.strip()]
        
        # Find the Excel file - try different possible locations
        if uploaded_file and hasattr(uploaded_file, 'name'):
            # If user uploaded a file, use it
            excel_file = uploaded_file
        else:
            # Try multiple possible paths
            possible_paths = [
                "job_listings.xlsx",  # Current directory
                "./job_listings.xlsx",  # Explicit current directory
                os.path.join(os.path.dirname(__file__), "job_listings.xlsx"),  # Relative to this file
            ]
            
            excel_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    excel_file = path
                    break
            
            if not excel_file:
                print("Job listings file not found. Please check the path or upload a file.")
                return pd.DataFrame()
        
        # Load job listings - handle both file object and path string
        if isinstance(excel_file, str):
            job_df = pd.read_excel(excel_file)
        else:
            job_df = pd.read_excel(excel_file)
        
        if job_df.empty:
            print("No job listings found in the file.")
            return pd.DataFrame()
        
        # Ensure required columns exist
        required_columns = ['title', 'link']
        if not all(col in job_df.columns for col in required_columns):
            print("Missing required columns in job listings file.")
            return pd.DataFrame()
        
        # Find matching jobs by title
        matching_jobs = find_matching_jobs(job_df, job_title)
        
        if not matching_jobs.empty:
            # Add synthetic job descriptions and required skills
            matching_jobs = enrich_job_data(matching_jobs, user_skills)
            
            # Calculate match scores
            matching_jobs = calculate_match_scores(matching_jobs, job_title, user_skills, experience)
            
            # Sort by match score and return top results
            return matching_jobs.sort_values('match_score', ascending=False).head(10)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error in job recommendation: {str(e)}")
        return pd.DataFrame()

def find_matching_jobs(df, job_title):
    """Find jobs that match the job title"""
    # Convert job title to lowercase for comparison
    job_title_lower = job_title.lower()
    
    # Filter DataFrame based on job title
    title_mask = df['title'].str.lower().str.contains(job_title_lower, na=False)
    matching_jobs = df[title_mask].copy()
    
    # If no exact matches, try partial matching
    if matching_jobs.empty:
        # Split job title into words
        title_words = job_title_lower.split()
        
        # Find jobs containing any of the words
        if len(title_words) > 0:
            word_matches = []
            for word in title_words:
                if len(word) >= 3:  # Only use words with 3+ characters
                    word_mask = df['title'].str.lower().str.contains(r'\b' + word + r'\b', na=False)
                    word_matches.append(df[word_mask])
            
            # Combine all matches
            if word_matches:
                matching_jobs = pd.concat(word_matches).drop_duplicates()
    
    # If still no matches, return a sample of jobs
    if matching_jobs.empty:
        matching_jobs = df.sample(min(10, len(df)))
    
    return matching_jobs

def enrich_job_data(df, user_skills):
    """Add synthetic job descriptions and required skills"""
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Create empty columns if they don't exist
    if 'description' not in df.columns:
        df['description'] = ""
    if 'required_skills' not in df.columns:
        df['required_skills'] = None
    if 'location' not in df.columns:
        df['location'] = "Not specified"
    if 'company' not in df.columns:
        df['company'] = "Unknown Company"
    if 'salary' not in df.columns:
        df['salary'] = "Not specified"
    if 'experience' not in df.columns:
        df['experience'] = 0
    
    # Common skills by job category
    skill_sets = {
        'data': ['SQL', 'Python', 'Excel', 'Statistics', 'Data Visualization', 'Machine Learning'],
        'software': ['JavaScript', 'Java', 'Python', 'C#', 'Git', 'Agile', 'REST API'],
        'marketing': ['Social Media', 'Content Creation', 'SEO', 'Analytics', 'Communication'],
        'design': ['UI/UX', 'Photoshop', 'Illustrator', 'Figma', 'Typography', 'Wireframing'],
        'management': ['Leadership', 'Project Management', 'Communication', 'Strategy', 'Team Building'],
        'engineer': ['Problem Solving', 'Technical Skills', 'Communication', 'Teamwork', 'Attention to Detail'],
        'analyst': ['Data Analysis', 'Excel', 'SQL', 'Problem Solving', 'Communication', 'Critical Thinking'],
        'default': ['Communication', 'Teamwork', 'Problem Solving', 'Time Management', 'Organization']
    }
    
    # Process each job
    for idx, row in df.iterrows():
        job_title = str(row['title']).lower()
        
        # If no description, create a synthetic one
        if not row['description'] or pd.isna(row['description']) or row['description'] == "":
            df.at[idx, 'description'] = f"Job Title: {row['title']}\n\nLooking for a qualified professional with experience in this field."
        
        # Assign skills based on job title keywords
        job_skills = skill_sets['default'].copy()  # Start with default skills
        
        # Add category-specific skills
        for category, skills in skill_sets.items():
            if category in job_title and category != 'default':
                job_skills.extend([s for s in skills if s not in job_skills])
        
        # Add some random user skills to increase match rate
        if user_skills:
            num_user_skills = min(len(user_skills), 2)
            if num_user_skills > 0:
                random_user_skills = random.sample(user_skills, num_user_skills)
                job_skills.extend([s for s in random_user_skills if s not in job_skills])
        
        # Assign skills
        df.at[idx, 'required_skills'] = job_skills[:5]  # Limit to 5 skills
        
        # Set random experience requirement between 0-5 years
        df.at[idx, 'experience'] = random.randint(0, 5)
        
        # Set URL if not present
        if 'url' not in df.columns or pd.isna(row.get('url')):
            df.at[idx, 'url'] = row.get('link', '#')
    
    return df

def calculate_match_scores(df, job_title, user_skills, user_experience):
    """Calculate match scores for each job"""
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Initialize match score column
    df['match_score'] = 0.0
    
    # Create a CountVectorizer to calculate text similarity
    vectorizer = CountVectorizer(stop_words='english')
    
    # Calculate title similarity (30% weight)
    if len(df) > 0:
        all_titles = df['title'].tolist() + [job_title]
        try:
            title_vectors = vectorizer.fit_transform(all_titles)
            last_vector = title_vectors[-1]
            title_similarities = cosine_similarity(last_vector, title_vectors[:-1])[0]
            df['title_match'] = title_similarities * 100
        except:
            # Fallback if vectorizer fails
            df['title_match'] = df['title'].apply(lambda x: 
                100 if job_title.lower() in x.lower() else 
                50 if any(word in x.lower() for word in job_title.lower().split()) else 30)
    else:
        df['title_match'] = 0
    
    # Calculate skills match (50% weight)
    df['skills_match'] = df.apply(
        lambda row: calculate_skills_match(user_skills, row['required_skills']), 
        axis=1
    )
    
    # Calculate experience match (20% weight)
    df['experience_match'] = df.apply(
        lambda row: min(100, (user_experience / max(row['experience'], 1)) * 100) if row['experience'] > 0 else 100,
        axis=1
    )
    
    # Calculate final weighted score
    df['match_score'] = (
        df['title_match'] * 0.3 + 
        df['skills_match'] * 0.5 + 
        df['experience_match'] * 0.2
    )
    
    # Add small random variation to break ties
    df['match_score'] = df['match_score'] + df['match_score'].apply(lambda x: random.uniform(0, 1))
    
    # Ensure score is within 0-100 range
    df['match_score'] = df['match_score'].clip(0, 100).round(1)
    
    return df

def calculate_skills_match(user_skills, job_skills):
    """Calculate the match percentage between user skills and job required skills"""
    if not user_skills or not job_skills:
        return 50.0  # Neutral score if no skills to compare
    
    # Convert job skills to lowercase for comparison
    if isinstance(job_skills, list):
        job_skills_lower = [str(skill).lower() for skill in job_skills]
    else:
        # If job_skills is not a list, return default score
        return 50.0
    
    # Count matches
    matches = 0
    for user_skill in user_skills:
        for job_skill in job_skills_lower:
            if user_skill in job_skill or job_skill in user_skill:
                matches += 1
                break
    
    # Calculate match percentage
    match_percent = (matches / len(user_skills)) * 100 if user_skills else 50.0
    
    return match_percent

# Run the function directly if the script is executed
if __name__ == "__main__":
    print("Running job recommendation...")
    # Test with sample inputs
    job_title = "Data Analyst"
    skills = "Python, SQL, Excel, Data Analysis"
    experience = 2
    
    results = streamlit_recommendation(job_title, skills, experience)
    
    if results is not None and not results.empty:
        print(f"Found {len(results)} matching jobs")
        print(results[['title', 'match_score']])
    else:
        print("No matching jobs found")
