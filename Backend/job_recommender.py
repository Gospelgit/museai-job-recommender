def log_memory_usage(log_point=""):
    """Log current memory usage."""
    # Force garbage collection
    gc.collect()
    
    try:
        # Get process memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Log memory usage
        print(f"MEMORY USAGE [{log_point}]: {memory_info.rss / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"Error logging memory: {e}")

def emergency_memory_cleanup():
    """Emergency cleanup when memory is low."""
    global embedding_cache, embedding_cache_timestamps, _batch_results
    
    print("Performing emergency memory cleanup...")
    
    # Clear caches
    if 'embedding_cache' in globals():
        embedding_cache.clear()
    if 'embedding_cache_timestamps' in globals():
        embedding_cache_timestamps.clear()
    if '_batch_results' in globals():
        _batch_results.clear()
    
    # Force garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Emergency memory cleanup completed")
# Global configurations for performance
MAX_CACHE_ENTRIES = 100  # Limit total entries
MAX_JOBS_TO_PROCESS = 15  # Reduced from 25 to 15
MAX_PARALLEL_WORKERS = 3  # Reduced from 5 to 3
EMBEDDING_DIMENSION = 1024  # Mini model embedding dimension
MODEL_LOADING_TIMEOUT = 45  # Maximum time to wait for model loading in seconds

# Use a model loading mechanism with timeout
_embedding_model = None
_embedding_model_lock = threading.Lock()
_model_loading_event = threading.Event()

def get_embedding_model():
    """Lazily initialize the embedding model with timeout mechanism."""
    global _embedding_model
    
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                # Start model loading in a separate thread
                loading_thread = threading.Thread(target=_load_model)
                loading_thread.start()
                
                # Wait for model loading with timeout
                if not _model_loading_event.wait(MODEL_LOADING_TIMEOUT):
                    # Use minimal model if loading takes too long
                    print("Model loading timeout. Using minimal configuration.")
                    _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    return _embedding_model

def _load_model():
    """Load a small efficient model and set the event when done."""
    global _embedding_model
    try:
        # Use the smallest viable model by default - no fallback needed
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Optimize model for inference
        _embedding_model.eval()  # Set to evaluation mode
        
        # Quantize model to reduce memory usage
        _embedding_model = _embedding_model.half()  # Use FP16 for all cases
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    finally:
        _model_loading_event.set()  # Signal that loading is complete

# Initialize embedding batching system
_batch_queue = []
_batch_queue_lock = threading.Lock()
_batch_results = {}
_batch_processing_thread = None
_batch_processing_event = threading.Event()
_stop_batch_processing = False

def start_batch_processing():
    """Start the background thread for batch processing embeddings."""
    global _batch_processing_thread, _stop_batch_processing
    
    if _batch_processing_thread is None or not _batch_processing_thread.is_alive():
        _stop_batch_processing = False
        _batch_processing_thread = threading.Thread(target=_process_embedding_batches)
        _batch_processing_thread.daemon = True
        _batch_processing_thread.start()

def stop_batch_processing():
    """Stop the background batch processing thread."""
    global _stop_batch_processing
    _stop_batch_processing = True
    _batch_processing_event.set()
    if _batch_processing_thread and _batch_processing_thread.is_alive():
        _batch_processing_thread.join(timeout=2)

def _process_embedding_batches():
    """Background thread to process embedding batches efficiently."""
    batch_size = 16  # Process 16 texts at once for efficiency
    
    while not _stop_batch_processing:
        current_batch = []
        current_keys = []
        
        # queuing batches size
        with _batch_queue_lock:
            while _batch_queue and len(current_batch) < batch_size:
                key, text = _batch_queue.pop(0)
                current_batch.append(text)
                current_keys.append(key)
        
        if current_batch:
            try:
                #computing embeddings in batch
                model = get_embedding_model()
                with torch.no_grad():  # Disable gradient calculation for inference
                    embeddings = model.encode(current_batch, normalize_embeddings=True)
                
                # Storing results
                for i, key in enumerate(current_keys):
                    _batch_results[key] = embeddings[i].tolist()
            except Exception as e:
                print(f"Error in batch embedding: {str(e)}")
                # Put empty embeddings as fallback
                for key in current_keys:
                    _batch_results[key] = [0.0] * EMBEDDING_DIMENSION
        
        # Wait for new items or timeout
        if not _batch_queue:
            _batch_processing_event.wait(timeout=0.5)
            _batch_processing_event.clear()

# Enhanced embedding cache with timeout-based invalidation
embedding_cache = {}
embedding_cache_timestamps = {}
CACHE_EXPIRY_SECONDS = 3600  # 1 hour cache validity


def clean_old_cache_entries():
    current_time = time.time()
    keys_to_remove = []
    
    # Removing expired entries
    for key, timestamp in list(embedding_cache_timestamps.items()):
        if current_time - timestamp > CACHE_EXPIRY_SECONDS:
            keys_to_remove.append(key)
    
   
    if len(embedding_cache) - len(keys_to_remove) > MAX_CACHE_ENTRIES:
        # Sort by timestamp and keep only newest MAX_CACHE_ENTRIES
        sorted_items = sorted(
            [(k, v) for k, v in embedding_cache_timestamps.items() if k not in keys_to_remove],
            key=lambda x: x[1], 
            reverse=True
        )
        
      
        to_keep = [k for k, _ in sorted_items[:MAX_CACHE_ENTRIES]]
        
        # Mark rest for removal
        for key in list(embedding_cache.keys()):
            if key not in to_keep and key not in keys_to_remove:
                keys_to_remove.append(key)
    
    # Remove all marked entries
    for key in keys_to_remove:
        embedding_cache.pop(key, None)
        embedding_cache_timestamps.pop(key, None)
    
  
    clean_batch_results()

@functools.lru_cache(maxsize=500)
def generate_embedding(text):
    """Generate embedding for text with batching support and caching."""
    if not text or not isinstance(text, str) or not text.strip():
        # Return zero vector for empty text
        return [0.0] * EMBEDDING_DIMENSION
    
    # Cleaning and normalizing the text
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    if len(cleaned_text) > 512:  # Limit text length for performance
        cleaned_text = cleaned_text[:512]
    
    # Check cache first
    cache_key = hash(cleaned_text)
    
    if cache_key in embedding_cache:
        embedding_cache_timestamps[cache_key] = time.time()  # Update timestamp
        return embedding_cache[cache_key]
    
    # If batch processing thread not started, start it
    start_batch_processing()
    
    # Adding to batch queue with unique key
    request_key = f"{cache_key}_{time.time()}"
    with _batch_queue_lock:
        _batch_queue.append((request_key, cleaned_text))
        _batch_processing_event.set()  # Signal new item
    
    # Wait for result with timeout
    max_wait = 5  # seconds
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if request_key in _batch_results:
            embedding = _batch_results.pop(request_key)
            # Cache the result
            embedding_cache[cache_key] = embedding
            embedding_cache_timestamps[cache_key] = time.time()
            return embedding
        time.sleep(0.1)
    
    # If timeout, generate embedding directly
    try:
        model = get_embedding_model()
        with torch.no_grad():
            embedding = model.encode(cleaned_text, normalize_embeddings=True).tolist()
        
        # Cache the result
        embedding_cache[cache_key] = embedding
        embedding_cache_timestamps[cache_key] = time.time()
        return embedding
    except Exception:
        # Return zeros on error
        return [0.0] * EMBEDDING_DIMENSION

# Downloading nltk resources
def initialize_nltk():
    """Initialize NLTK resources if needed."""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

# Start NLTK initialization in background
threading.Thread(target=initialize_nltk, daemon=True).start()

# Optimization: Pre-loading stopwords and lemmatizer
_stopwords = None
_lemmatizer = None

def get_stopwords():
    """Get stopwords with lazy loading."""
    global _stopwords
    if _stopwords is None:
        try:
            _stopwords = set(stopwords.words('english'))
        except:
            _stopwords = set()  # Fallback
    return _stopwords

def get_lemmatizer():
    """Get lemmatizer with lazy loading."""
    global _lemmatizer
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer

def extract_job_details(job_description):
    """
    Extract job title, company name, responsibilities, qualifications, years of experience,
    location, salary/pay information, and the entity needing the work done from a job description.
    Optimized version with cleaner regex and early returns.
    """
    # Initialize with defaults
    job_details = {
        "job_title": "Unknown Job Title",
        "company_name": "Unknown Company",
        "job_responsibilities": "",
        "job_qualifications": "",
        "job_location": "Not specified",
        "salary_info": "Not specified",
        "entity_info": "Not specified",
        "experience_years": 0
    }
    
    # Return early if the input is empty
    if not job_description or not isinstance(job_description, str):
        return tuple(job_details.values())
    
    # Extract job title
    title_match = re.search(r"\*\*Job Title:\*\*\s*(.+)", job_description, re.IGNORECASE)
    if title_match:
        job_details["job_title"] = title_match.group(1).strip()

    # Extract company name
    company_match = re.search(r"\*\*Company:\*\*\s*(.+)", job_description, re.IGNORECASE)
    if company_match:
        job_details["company_name"] = company_match.group(1).strip()

    # Extract job responsibilities
    responsibilities_match = re.search(r"\*\*Job Responsibilities:\*\*\n(.*?)(?=\n\*\*|\Z)", job_description, re.DOTALL)
    if responsibilities_match:
        job_details["job_responsibilities"] = responsibilities_match.group(1).strip()

    # Extract job qualifications
    qualifications_match = re.search(r"\*\*Required Skills & Qualifications:\*\*\n(.*?)(?=\n\*\*|\Z)", job_description, re.DOTALL)
    if qualifications_match:
        job_details["job_qualifications"] = qualifications_match.group(1).strip()
        # Extract years of experience
        experience_match = re.search(r"(\d+)\+\s*years?", job_details["job_qualifications"])
        if experience_match:
            job_details["experience_years"] = int(experience_match.group(1))
        
    # Extract location information
    location_match = re.search(r"\*\*Location:\*\*\s*(.+?)(?=\n\*\*|\Z)", job_description, re.IGNORECASE | re.DOTALL)
    if location_match:
        job_details["job_location"] = location_match.group(1).strip()
    else:
        # Try alternative patterns for location
        for pattern in [r"Location:?\s*(.+?)(?=\n|\.|\,)", r"(?:based in|located in|position in|job in|work in)\s+(.+?)(?=\n|\.|\,)"]:
            location_match = re.search(pattern, job_description, re.IGNORECASE)
            if location_match:
                job_details["job_location"] = location_match.group(1).strip()
                break
        # Check for remote work
        if "remote" in job_description.lower() or "work from home" in job_description.lower() or "wfh" in job_description.lower():
            job_details["job_location"] = "Remote"
    
    # Extract salary/pay information
    for pattern in [
        r"\*\*Salary:\*\*\s*(.+?)(?=\n\*\*|\Z)",
        r"\*\*Pay:\*\*\s*(.+?)(?=\n\*\*|\Z)",
        r"\*\*Compensation:\*\*\s*(.+?)(?=\n\*\*|\Z)",
        r"(?:salary|pay|compensation)(?:\s+is|\s+range|\s*:)?\s+(?:[\$€£]?\s*\d+(?:[,\.]\d+)*(?:\s*[kK])?(?:\s*-\s*[\$€£]?\s*\d+(?:[,\.]\d+)*(?:\s*[kK])?)?)",
    ]:
        salary_match = re.search(pattern, job_description, re.IGNORECASE)
        if salary_match:
            matched_group = 1 if "**" in pattern else 0
            if matched_group == 0:
                job_details["salary_info"] = salary_match.group(0).strip()
            else:
                job_details["salary_info"] = salary_match.group(matched_group).strip()
            break
    
    # Extract entity info
    for pattern in [
        r"(?:our|the)\s+(client|company|organization|business|firm|employer|team|department|agency|startup|enterprise)",
        r"(?:joining|work with|work for|be part of)\s+(?:our|the|a)\s+(team|organization|company|business|startup|firm|corporation|agency)",
    ]:
        entity_match = re.search(pattern, job_description, re.IGNORECASE)
        if entity_match:
            job_details["entity_info"] = entity_match.group(1).strip()
            break

    return (
        job_details["job_title"], 
        job_details["company_name"], 
        job_details["job_responsibilities"], 
        job_details["job_qualifications"], 
        job_details["experience_years"], 
        job_details["job_location"], 
        job_details["salary_info"], 
        job_details["entity_info"]
    )

# Use LRU cache for extract_key_terms to avoid recomputation
@functools.lru_cache(maxsize=300)
def extract_key_terms(text):
    """
    Extract important keywords and phrases from text using NLP techniques.
    Cached version with efficiency improvements.
    """
    if not text or not isinstance(text, str):
        return []
        
    try:
        # Get lemmatizer and stopwords lazily
        lemmatizer = get_lemmatizer()
        stop_words = get_stopwords()
        
        # Tokenize and convert to lowercase
        tokens = word_tokenize(text.lower())
        
        # Filter tokens more efficiently
        filtered_tokens = []
        for token in tokens:
            if token not in stop_words and len(token) > 2 and token.isalpha():
                filtered_tokens.append(lemmatizer.lemmatize(token))
        
        # Use Counter for term frequency
        term_counter = Counter(filtered_tokens)
        
        # Get most common terms (reduced for performance)
        return [term for term, _ in term_counter.most_common(40)]  
    except Exception:
        # Fallback to simple word extraction if NLP processing fails
        words = text.lower().split()
        return [w for w in words if len(w) > 2][:40]

@functools.lru_cache(maxsize=500)
def calculate_term_overlap(user_terms_tuple, job_terms_tuple):
    """
    Calculate the overlap between user's terms and job terms.
    Cached version using immutable tuples.
    """
    if not user_terms_tuple or not job_terms_tuple:
        return 0.0
    
    # Convert tuples to sets
    user_set = set(user_terms_tuple)
    job_set = set(job_terms_tuple)
    
    # Calculate Jaccard similarity
    intersection = len(user_set.intersection(job_set))
    union = len(user_set.union(job_set))
    
    return intersection / union if union > 0 else 0.0

@functools.lru_cache(maxsize=500)
def calculate_title_relevance(user_input, job_title):
    """
    Calculate how relevant the job title is to the user's search query.
    Cached version with improved scoring.
    """
    if not user_input or not job_title:
        return 0.5  # Neutral score for empty inputs
        
    # Clean and normalize both inputs
    user_input_clean = user_input.lower()
    job_title_clean = job_title.lower()
    
    # Split into terms and create sets
    user_terms = set(re.findall(r'\b\w+\b', user_input_clean))
    title_terms = set(re.findall(r'\b\w+\b', job_title_clean))
    
    # Calculate term overlap
    if not user_terms:
        return 0.5  # Neutral score if no user terms
        
    intersection = len(user_terms.intersection(title_terms))
    overlap_ratio = intersection / len(user_terms)
    
    # Improve match scoring with sequence matching
    exact_phrase_bonus = 0
    for i in range(len(user_terms)):
        if user_input_clean in job_title_clean:
            exact_phrase_bonus = 0.2
            break
    
    # Give extra weight to matches in sequence
    extra_score = 0
    for user_word in user_terms:
        if len(user_word) >= 4 and user_word in job_title_clean:
            extra_score += 0.1
    
    return min(1.0, overlap_ratio + extra_score + exact_phrase_bonus)

def vector_search(user_profile_embedding, user_job_title_embedding, user_experience, user_location, 
                  job_descriptions, job_urls, job_names, user_input, user_skills, top_k=10):
    """
    Perform a vector search on job descriptions based on the user profile and job title preference.
    Optimized version with faster processing and improved scoring.
    """
    # Extract key terms only once
    user_key_terms = tuple(extract_key_terms(user_input))
    user_skill_terms = tuple(extract_key_terms(user_skills))
    
    # Prepare a variation of user inputs for better matching
    user_input_embedding = generate_embedding(user_input)
    
    # Process jobs in parallel - with reduced parallelism
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        futures = []
        for job_desc, job_url, job_name in zip(job_descriptions, job_urls, job_names):
            futures.append(executor.submit(
                process_job_for_search,
                job_desc, job_url, job_name,
                user_profile_embedding, user_job_title_embedding, user_input_embedding,
                user_experience, user_location, user_input,
                user_key_terms, user_skill_terms
            ))
        
        # Collect results as they complete
        scored_jobs = []
        for future in concurrent.futures.as_completed(futures):
            job_result = future.result()
            if job_result:
                scored_jobs.append(job_result)
                
                # Early termination if we have enough high-scoring results
                if len(scored_jobs) >= top_k * 1.5 and any(job["score"] > 80 for job in scored_jobs):
                    # Cancel remaining futures if we have good matches
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
    
    # Force garbage collection after parallel processing
    import gc
    gc.collect()
    
    # Sort by score in descending order
    scored_jobs.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top-k results
    return scored_jobs[:top_k]

def process_job_for_search(job_desc, job_url, job_name, 
                          user_profile_embedding, user_job_title_embedding, user_input_embedding,
                          user_experience, user_location, user_input,
                          user_key_terms, user_skill_terms):
    """Process a single job for the vector search function with improved scoring."""
    try:
        # Extract job details
        job_title, company_name, job_responsibilities, job_qualifications, job_experience, job_location, salary_info, entity_info = extract_job_details(job_desc)
        
        # Create a simplified job text representation - focus on most relevant parts
        job_text = f"{job_title}. {job_responsibilities[:300]}. {job_qualifications[:300]}"
        
        # Generate job embeddings
        job_embedding = generate_embedding(job_text)
        job_title_embedding = generate_embedding(job_title)
        
        # Compute similarity scores 
        def cosine_similarity(a, b):
            """Calculate cosine similarity between two vectors"""
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
            
        skill_similarity_score = float(cosine_similarity(user_profile_embedding, job_embedding))
        title_similarity_score = float(cosine_similarity(user_job_title_embedding, job_title_embedding))
        query_match_score = float(cosine_similarity(user_input_embedding, job_embedding))
        
        # Extract key terms from job
        job_key_terms = tuple(extract_key_terms(job_desc[:1000]))  # Limit processing to first 1000 chars
        
        # Calculate overlaps
        title_term_overlap = calculate_term_overlap(user_key_terms, job_key_terms)
        skill_term_overlap = calculate_term_overlap(user_skill_terms, job_key_terms)
        
        # Calculate title relevance with improved method
        title_relevance = calculate_title_relevance(user_input, job_title)
        
        # New scoring weights for BGE model - adjusted for better matching
        combined_score = (
            skill_similarity_score * 0.30 +     # Skills match importance
            title_similarity_score * 0.20 +     # Job title match
            query_match_score * 0.15 +          # Overall query match
            skill_term_overlap * 0.15 +         # Explicit skill terms
            title_term_overlap * 0.10 +         # Title term matching
            title_relevance * 0.10              # Direct title relevance
        )
        
        # Exact title match bonus
        if user_input.lower() in job_title.lower():
            combined_score += 0.15
        
        # Experience adjustment - more nuanced
        if job_experience > 0 and 'intern' not in job_title.lower():
            if user_experience < job_experience:
                # Smaller penalty for being close
                penalty = min(0.3, (job_experience - user_experience) * 0.05)
                combined_score -= penalty
            elif user_experience >= job_experience:
                # Bonus for having enough experience
                bonus = min(0.15, 0.05 + (user_experience - job_experience) * 0.02)
                combined_score += bonus
        
        # Location adjustment
        user_location_str = str(user_location).lower() if user_location else ""
        job_location_str = str(job_location).lower() if job_location else ""
        
        if user_location_str and user_location_str not in ["remote", ""]:
            if any(loc in job_location_str for loc in [user_location_str, "remote", "anywhere"]):
                combined_score += 0.1
            else:
                combined_score -= 0.1
        elif user_location_str and user_location_str == "remote":
            if any(loc in job_location_str for loc in ["remote", "anywhere", "work from home", "wfh"]):
                combined_score += 0.1
        
        # Small random variation to break ties
        combined_score += random.uniform(0, 0.003)
        
        # Normalize score to 0-100% with better distribution
        raw_score = combined_score * 100
        # Scale final score to have more meaningful distribution between 0-100
        final_score = max(0, min(100, raw_score))
        
        # Create job result - ensure all values are JSON serializable
        return {
            "jobTitle": job_name if job_name else job_title,
            "company": entity_info if entity_info != "Not specified" else company_name,
            "score": float(round(final_score, 1)),  # Ensure this is a standard Python float
            "location": job_location,
            "salary": salary_info,
            "experience": int(job_experience),  # Ensure this is a standard Python int
            "url": str(job_url),
            "responsibilities": job_responsibilities[:200] + "..." if len(job_responsibilities) > 200 else job_responsibilities,
            # Add key skills for better user understanding
            "key_skills": ", ".join(job_key_terms[:5]) if job_key_terms else "Not specified"
        }
    except Exception as e:
        # Skip jobs that cause errors
        print(f"Error processing job: {str(e)}")
        return None

def find_matching_jobs_by_title(df, user_input, min_consecutive_chars=3):
    """
    Find jobs in the DataFrame that match the user input.
    Optimized version with faster matching and smarter filtering.
    """
    if df.empty or not user_input:
        return []
        
    # Determine title column - use 'title' as it exists in your Excel
    title_column = 'title'
    if title_column not in df.columns:
        return []
    
    # Ensure the user input has lowercase for comparison
    user_input_lower = user_input.lower()
    # Extract meaningful words (3+ chars)
    user_input_words = [word.strip() for word in re.findall(r'\b\w+\b', user_input_lower) if len(word.strip()) >= 3]
    
    # Create a copy with lowercase titles
    df_filtered = df.copy()
    df_filtered['title_lower'] = df_filtered[title_column].str.lower()
    
    # Optimization: pre-filter DataFrame
    matches = []
    exact_match_indices = []
    
    # First look for exact phrase matches (highest priority)
    exact_match_mask = df_filtered['title_lower'].str.contains(user_input_lower, regex=False)
    exact_match_indices = df_filtered[exact_match_mask].index.tolist()
    matches.extend(exact_match_indices)
    
    # Then look for matches containing all words
    if len(matches) < 100 and len(user_input_words) > 0:
        all_words_mask = df_filtered['title_lower'].apply(
            lambda title: all(word in title for word in user_input_words)
        )
        all_words_indices = df_filtered[all_words_mask].index.difference(exact_match_indices).tolist()
        matches.extend(all_words_indices[:50])  # Limit to 50 for performance
    
    # Then match individual words
    if len(matches) < 100:
        for word in user_input_words:
            if len(matches) >= 100:
                break
                
            # Find rows containing this word as whole word
            word_mask = df_filtered['title_lower'].str.contains(r'\b' + re.escape(word) + r'\b', regex=True)
            word_indices = df_filtered[word_mask].index.difference(matches).tolist()
            matches.extend(word_indices[:50])  # Limit to 50 per word
    
    # If still not enough matches, try semantic matching with embeddings
    if len(matches) < 50:
        # Create embedding for user input
        user_input_embedding = generate_embedding(user_input)
        
        # Get remaining indices
        remaining_indices = list(set(df_filtered.index) - set(matches))
        
        # Only process a subset of remaining jobs for performance
        sample_size = min(200, len(remaining_indices))
        if sample_size > 0:
            remaining_sample = random.sample(remaining_indices, sample_size)
            
            # Get job titles
            remaining_titles = df_filtered.loc[remaining_sample, title_column].tolist()
            
            # Process in batches
            batch_size = 16
            all_similarities = []
            remaining_indices_processed = []
            
            for i in range(0, len(remaining_titles), batch_size):
                batch_titles = remaining_titles[i:i+batch_size]
                batch_indices = remaining_sample[i:i+batch_size]
                
                # Generate embeddings for this batch
                batch_embeddings = [generate_embedding(title) for title in batch_titles]
                
                # Calculate similarities
                batch_similarities = [float(cosine_similarity(user_input_embedding, job_emb))
                                    for job_emb in batch_embeddings]
                
                all_similarities.extend(batch_similarities)
                remaining_indices_processed.extend(batch_indices)
            
            # Get indices of top similarities
            if all_similarities:
                remaining_with_scores = list(zip(remaining_indices_processed, all_similarities))
                remaining_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Only add high-similarity matches
                semantic_matches = [(idx, score) for idx, score in remaining_with_scores if score > 0.5]
                matches.extend([idx for idx, _ in semantic_matches[:30]])
    
    # Convert matches to row dictionaries, limiting to top MAX_JOBS_TO_PROCESS
    matches = list(set(matches))  # Remove duplicates
    matching_rows = []
    for idx in matches[:MAX_JOBS_TO_PROCESS]:  # Limit for performance
        row_dict = df.loc[idx].to_dict()
        # Ensure the row dictionary has expected keys
        if 'title' in row_dict and 'link' in row_dict:
            matching_rows.append(row_dict)
    
    return matching_rows

# Selenium driver pool for reuse
_selenium_pool = []
_selenium_pool_lock = threading.Lock()

def get_selenium_driver():
    """Get a Selenium driver from the pool or create a new one."""
    with _selenium_pool_lock:
        if _selenium_pool:
            return _selenium_pool.pop()
        else:
            return initialize_selenium_driver()

def return_selenium_driver(driver):
    """Return a Selenium driver to the pool."""
    if driver:
        with _selenium_pool_lock:
            if len(_selenium_pool) < 2:  # Reduced pool size for memory efficiency
                _selenium_pool.append(driver)
            else:
                try:
                    driver.quit()
                except:
                    pass

def initialize_selenium_driver():
    """Initialize and return a configured Selenium WebDriver with optimized settings."""
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=800,600")  # Smaller size for faster loading
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-images")  # Disable images
        chrome_options.add_argument("--disable-javascript")  # Disable JavaScript where possible
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(10)  # Further reduced timeout
        return driver
    except Exception as e:
        print(f"Error initializing Selenium driver: {str(e)}")
        return None

def extract_job_description_from_url(url, driver=None):
    """
    Extract job description from a job listing URL.
    Optimized version with better error handling, timeouts, and fallbacks.
    """
    url = str(url) if url is not None else ""
    if not url:
        return "No URL provided"
    
    # First try with requests for speed
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)  # Reduced timeout
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find job description in common containers
            for selector in ['job-description', 'description', 'jobDescription', 'details', 'listing-desc',
                           'job-details', 'jobdetails', 'job_description', 'content', 'vacancy-details']:
                desc_div = soup.find(class_=re.compile(f'.*{selector}.*', re.IGNORECASE))
                if desc_div:
                    text = desc_div.get_text(separator=' ', strip=True)
                    if len(text) > 100:  # Only return if substantial content
                        return text
            
            # If no specific container found, look for main content tags
            for tag in ['main', 'article', 'section']:
                content = soup.find(tag)
                if content:
                    text = content.get_text(separator=' ', strip=True)
                    if len(text) > 200:
                        return text
            
            # If still not found, try body content
            if soup.body:
                body_text = soup.body.get_text(separator=' ', strip=True)
                if len(body_text) > 200:
                    return body_text[:5000]  # Limit length for processing efficiency
    except Exception as e:
        print(f"Request error for {url}: {str(e)}")
    
    # Fall back to Selenium only if needed and available
    if driver:
        try:
            driver.get(url)
            
            # Shorter wait time
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Try specific job description elements first
            for selector in [
                "div.description__text", "div#jobDescriptionText", "div.jobDescriptionContent",
                "[class*='jobDescription']", "[class*='job-description']", "[id*='jobDescription']"
            ]:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        description = " ".join([element.text for element in elements])
                        if description.strip():
                            return description
                except:
                    continue
            
            # If no specific container found, get all body text
            page_text = driver.find_element(By.TAG_NAME, "body").text
            return page_text[:5000]  # Limit length for processing efficiency
        except Exception as e:
            print(f"Selenium error for {url}: {str(e)}")
            return "Failed to extract job description"
    
    return "Could not extract job description"

def fetch_job_description(job):
    """Fetch job description for a single job with timeout and retries."""
    # Get job details
    job_title = job.get('title', '')
    job_link = job.get('link', '')
    
    # Skip if no link or title
    if not job_link or not job_title:
        return None
    
    # Use a simple retry mechanism
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Get Selenium driver only if needed
            driver = None
            if any(domain in job_link.lower() for domain in [
                'linkedin', 'indeed', 'glassdoor', 'monster', 'dice', 'greenhouse.io', 
                'lever.co', 'jobvite', 'workday', 'applytojob', 'smartrecruiters'
            ]):
                driver = get_selenium_driver()
            
            # Extract job description
            job_description = extract_job_description_from_url(job_link, driver)
            
            # Return driver to pool
            if driver:
                return_selenium_driver(driver)
            
            # Extract location from description
            _, _, _, _, _, extracted_location, salary, _ = extract_job_details(job_description)
            
            return {
                'description': job_description,
                'url': job_link,
                'title': job_title,
                'location': extracted_location,
                'salary': salary
            }
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {job_link}: {str(e)}")
            if driver:
                try:
                    driver.quit()
                except:
                    pass
            # Simple exponential backoff
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
    
    # Return partial data if we have a title and link
    return {
        'description': f"Job Title: {job_title}",
        'url': job_link,
        'title': job_title,
        'location': "Not specified",
        'salary': "Not specified"
    }

def handle_web_request(job_title, skills, experience, uploaded_file=None):
    """
    Process a web request for job recommendations with optimized performance.
    """
    start_time = time.time()
    
    try:
        # Track memory at start
        log_memory_usage("request_start")
        
        # Start batch processing for embeddings
        start_batch_processing()
        
        # Find the Excel file
        if uploaded_file and os.path.exists(uploaded_file):
            excel_file = uploaded_file
        else:
            # Try multiple possible paths
            possible_paths = [
                r"C:/Users/Gospel/Documents/Flask JR launch/job_listings.xlsx",
                r"job_listings.xlsx",  # Try current directory
                r"./job_listings.xlsx"  # Also try explicit current directory
            ]
            
            excel_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    excel_file = path
                    break
            
            if not excel_file:
                return {"error": "Job listings file not found. Please check the path or upload a file."}
        
        # Load job listings - Only read needed columns for efficiency
        print("Loading job listings from Excel...")
        job_df = pd.read_excel(
            excel_file, 
            usecols=['title', 'link'],  # Only use essential columns
            engine='openpyxl'  # Specify engine explicitly
        )
        
        log_memory_usage("after_excel_load")
        
        if job_df.empty:
            return {"error": "No job listings found in the Excel file. Please check your data."}
        
        # Generate embeddings for user input - do this early for better parallelization
        print("Processing your search query...")
        user_profile_embedding = generate_embedding(skills)
        user_job_title_embedding = generate_embedding(job_title)
        
        # Find matching jobs by title
        print("Finding matching jobs...")
        matching_jobs = find_matching_jobs_by_title(job_df, job_title)
        
        # Clean up dataframe as it's no longer needed
        del job_df
        gc.collect()
        
        if not matching_jobs:
            return {"error": "We no see any job wey match your search. Try another job title."}
        
        # Process matching jobs in parallel with reduced count for speed
        print(f"Processing {len(matching_jobs[:MAX_JOBS_TO_PROCESS])} job listings...")
        all_job_data = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
            futures = {executor.submit(fetch_job_description, job): job 
                      for job in matching_jobs[:MAX_JOBS_TO_PROCESS]}
            
            # Process results as they come in
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    all_job_data.append(result)
                    
                    # Early termination if we have enough jobs and have been processing for over 15 seconds
                    if len(all_job_data) >= 10 and time.time() - start_time > 15:
                        # Cancel remaining futures
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
        
        log_memory_usage("after_job_fetch")
        
        if not all_job_data:
            return {"error": "We no fit get any job descriptions. Please check your internet connection."}
        
        # Extract data into separate lists
        job_descriptions = [data['description'] for data in all_job_data]
        job_urls = [data['url'] for data in all_job_data]
        job_names = [data['title'] for data in all_job_data]
        
        # Get recommendations
        print("Finding the best matches for you...")
        recommendations = vector_search(
            user_profile_embedding,
            user_job_title_embedding,
            experience,
            "",  # Empty location preference
            job_descriptions,
            job_urls,
            job_names,
            job_title,
            skills,
            top_k=10  # Get top 10 results
        )
        
        # Clean up resources
        clean_old_cache_entries()  # Clean up stale cache entries
        stop_batch_processing()  # Stop background embedding processing
        
        # Final memory cleanup
        emergency_memory_cleanup()
        log_memory_usage("request_end")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        # Include processing time in response
        return {
            "recommendations": recommendations,
            "processing_time": f"{processing_time:.2f} seconds",
            "model_used": "all-MiniLM-L6-v2"  # Update to show correct model
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {"error": f"Error don happen: {str(e)}", "details": error_details}
    finally:
        # Ensure resources are cleaned up
        stop_batch_processing()
        emergency_memory_cleanup()

def main():
    """Command-line interface for the job recommender"""
    try:
        print("🚀 Starting Advanced Job Recommender with BGE model...")
        
        # Use the correct Excel file path
        excel_file = r"job_listings.xlsx"  # Try current directory first
        
        if not os.path.exists(excel_file):
            # Try alternative path
            excel_file = r"C:/Users/Gospel/Documents/Flask JR launch/job_listings.xlsx"
            if not os.path.exists(excel_file):
                print("We no fit find the job listings file. Please check the path.")
                return
        
        print("Loading job data...")
        # Load job listings from Excel - only needed columns
        job_df = pd.read_excel(excel_file, usecols=['title', 'link'])
        
        if job_df.empty:
            print("No job listings found. Please check your Excel file.")
            return
        
        # Ask for user input
        user_input = input("Abeg, which job you for like make we search internet, come recommend give you?: ")
        user_skills = input("Which skills or qualifications you get for this job?: ")
        user_experience = int(input("How many years of experience you get like this?: "))
        
        print("\nStarting search... This go take like 30-60 seconds with the BGE large model...")
        
        # Call the web handler with CLI inputs
        result = handle_web_request(user_input, user_skills, user_experience)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
            
        # Display the recommendations
        print("\n===== THE TOP JOBS WEY FIT YOU WELL =====\n")
        
        for i, job in enumerate(result["recommendations"], 1):
            print(f"#{i}. {job['jobTitle']}")
            print(f"Company: {job['company']}")
            print(f"Match Score: {40 + job['score']}%")
            print(f"Location: {job['location']}")
            if job.get('key_skills'):
                print(f"Key Skills: {job['key_skills']}")
            if job['salary'] != "Not specified":
                print(f"Salary: {job['salary']}")
            if job['experience'] > 0:
                print(f"Experience Required: {job['experience']}+ years")
            print(f"Job Link: {job['url']}")
            print("-" * 50)
        
        if "processing_time" in result:
            print(f"\nProcessing Time: {result['processing_time']}")
        
    except Exception as e:
        import traceback
        print(f"Error don happen: {str(e)}")
        print(traceback.format_exc())
    finally:
        # Clean up resources
        stop_batch_processing()

if __name__ == "__main__":
    main()
