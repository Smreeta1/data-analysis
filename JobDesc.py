import spacy
import re
from collections import defaultdict

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Define keywords
technical_skills = {'python', 'django', 'redis', 'aws', 'docker', 'kubernetes', 'fastapi', 'mysql', 'postgresql', 'java', 'ruby', 'javascript', 'c++', 'c#'}
qualification_keywords = {'bachelor', 'master', 'degree', 'certification', 'certificate', 'training','IT','CS','Engineering','Computer Science'}
soft_skills = {'communication', 'teamwork', 'problem-solving', 'leadership', 'time management', 'adaptability'}

def remove_stop_words(doc):
    """Remove stop words."""
    return [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]

def extract_years_of_experience(text):
    """Extract years of experience from text."""
    experience_pattern = re.compile(r'\b(\d+\s*(to|-|-)\s*\d+|\d+)\s*(years?|yr|y)\b', re.IGNORECASE)
    return [match[0] for match in experience_pattern.findall(text)]  # Convert to list of strings

def extract_information(text):
    """Extracts detailed keywords and sections from job description text."""
    doc = nlp(text)
    
    # Remove stop words
    tokens = remove_stop_words(doc)
    
    sections = defaultdict(list)
    current_section = 'General'
    
    # Extract sections and their content
    for line in text.split('\n'):
        line = line.strip()
        if line.endswith(':'):
            current_section = line[:-1]  # Remove the colon
        else:
            sections[current_section].append(line)
    
    # Initialize storage for categorized info
    categorized_info = defaultdict(lambda: {'Technical Skills': set(), 'Years of Experience': set(), 'Qualifications': set(), 'Soft Skills': set(), 'Responsibilities': set()})
    
    # Extract years of experience
    years_of_experience = extract_years_of_experience(text)
    categorized_info['General']['Years of Experience'].update(years_of_experience)
    
    # sections
    for section, lines in sections.items():
        section_text = ' '.join(lines)
        doc_section = nlp(section_text)
        
        # Remove stop words from the section text
        tokens_section = remove_stop_words(doc_section)
        
        # Identify technical skills
        for token in tokens_section:
            if token in technical_skills:
                categorized_info['General']['Technical Skills'].add(token)
        
        # Extract qualifications
        if section.lower() in {'requirements', 'qualifications'}:
            for token in tokens_section:
                if token in qualification_keywords:
                    categorized_info[section]['Qualifications'].add(token)
        
        # Extract soft skills
        if section.lower() in {'requirements', 'qualifications', 'nice to have'}:
            for token in tokens_section:
                if token in soft_skills:
                    categorized_info['General']['Soft Skills'].add(token)
        
        # Extract responsibilities with trigrams starting with verbs
        if section.lower() == 'responsibilities':
            responsibilities = set()
            tokens = [token.text for token in doc_section if token.text.lower() not in spacy.lang.en.stop_words.STOP_WORDS]
            for i in range(len(tokens) - 2):
                if doc_section[i].pos_ == 'VERB':
                    trigram = f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}"
                    responsibilities.add(trigram)
            categorized_info[section]['Responsibilities'] = responsibilities
    
    return categorized_info

def main():
    text = """Requirements:
Must complete Bachelor's degree in Computer Science or related field.
Must have 4-5 years of professional experience with Python, Django & Rest API.
Proficiency in Redis Implementation, Multitenant Architecture & Elastic Search.
Experience with signal, web sockets, celery, scheduler & crone jobs.
Experience with Google Cloud Platform and/or AWS.
Experience with GitLab or similar version control systems.
Experience with MySQL/PostgreSQL with optimization and handling big data.
Experience with Agile development methodologies.
Strong problem-solving and debugging skills.
Excellent communication and ability to work independently and as part of a team.
If you are passionate about back-end development and want to work on challenging projects in a fast-paced environment, we encourage you to apply for this exciting opportunity.

Responsibilities:
Develop and maintain the back end of our web applications using Python Django.
Collaborate with front-end developers to integrate the back-end with the front-end application.
Write efficient and maintainable code using best practices and design patterns.
Develop and maintain unit tests for back-end code.
Use related technologies to build server-side applications, APIs, and integrations.
Participate in code reviews and provide feedback to improve code quality.

Nice To Have:
Experience with other backend technologies, such as Node.js, Ruby on Rails, etc.
Experience building web applications using the FastAPI framework and understanding the features it offers for building web APIs like asynchronous support, dependency injection, and route generation.
Experience with containerization and orchestration technologies, such as Docker and Kubernetes.
Experience with APM tools like New Relic or similar will be an advantage."""

    categorized_info = extract_information(text)

    # Print results
    for section, info in categorized_info.items():
        print(f"--- {section} ---")
        for category, details in info.items():
            if details:
                # All details are strings,returbs tuple otherwise:error
                details_str = [str(detail) for detail in details]
                print(f"{category}: {', '.join(details_str)}")
        print() 

if __name__ == "__main__":
    main()
