"""
Simple Q&A system with zero dependencies and instant responses.
No training needed - just a clean lookup table with fuzzy matching.
"""

import re
from difflib import SequenceMatcher

# Comprehensive Q&A database
QA_DATABASE = {
    # USA/Geography
    "what is usa": "USA stands for the United States of America, a federal republic of 50 states in North America.",
    "full form of usa": "United States of America",
    "what does usa stand for": "United States of America",
    "capital of usa": "Washington, D.C. is the capital of the United States.",
    "capital of france": "Paris is the capital of France.",
    "largest ocean": "The Pacific Ocean is the largest ocean, covering about 165 million square kilometers.",
    "tallest mountain": "Mount Everest is the tallest mountain at 8,849 meters (29,032 feet) above sea level.",
    "how many continents": "There are seven continents: Africa, Antarctica, Asia, Europe, North America, Oceania, and South America.",
    
    # Science - Physics
    "what is gravity": "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it gives weight to physical objects.",
    "speed of light": "The speed of light in vacuum is approximately 299,792,458 meters per second (about 300,000 km/s).",
    "what is energy": "Energy is the capacity to do work or cause change. It exists in various forms like kinetic, potential, thermal, and chemical.",
    "what is force": "A force is a push or pull that can change an object's motion, direction, or shape. Measured in Newtons.",
    "what is velocity": "Velocity is the speed and direction of an object's motion. It's a vector quantity measured in meters per second.",
    "what is acceleration": "Acceleration is the rate of change of velocity over time. It's measured in meters per second squared.",
    
    # Science - Biology
    "what is photosynthesis": "Photosynthesis is the process by which plants convert light energy, water, and carbon dioxide into glucose and oxygen.",
    "what is dna": "DNA (Deoxyribonucleic Acid) is the molecule that carries genetic instructions for life. It has a double helix structure.",
    "what is a cell": "A cell is the basic structural and functional unit of all living organisms. It contains genetic material and organelles.",
    "what is evolution": "Evolution is the process by which species change over time through natural selection, genetic variation, and adaptation.",
    "what is the heart": "The heart is a muscular organ that pumps blood throughout the body, delivering oxygen and nutrients to tissues.",
    "what are chromosomes": "Chromosomes are thread-like structures in cells that contain DNA. Humans have 23 pairs (46 total).",
    
    # Science - Chemistry
    "what is water made of": "Water (H₂O) is made of two hydrogen atoms and one oxygen atom bonded together.",
    "what is an element": "An element is a pure substance made of only one type of atom. There are 118 known elements.",
    "what is oxygen": "Oxygen is a chemical element (symbol O, atomic number 8) essential for respiration in most living organisms.",
    "what is a molecule": "A molecule is two or more atoms bonded together. It's the smallest unit of a chemical compound.",
    "what is carbon": "Carbon is a chemical element (symbol C, atomic number 6) that forms the basis of all organic compounds.",
    
    # Computing/Technology
    "what is python": "Python is a high-level programming language known for its simplicity, readability, and extensive library support.",
    "what is a variable": "A variable is a named storage location in computer memory that holds a value which can change during program execution.",
    "what is an algorithm": "An algorithm is a step-by-step procedure or formula for solving a problem or completing a task.",
    "what is recursion": "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem.",
    "what is the internet": "The Internet is a global network connecting millions of computers, enabling communication and information sharing worldwide.",
    "what is ai": "Artificial Intelligence (AI) is technology that enables machines to perform tasks requiring human-like intelligence, such as learning and problem-solving.",
    "what is machine learning": "Machine learning is a subset of AI where computers learn from data to make predictions without being explicitly programmed.",
    "what is a computer": "A computer is an electronic device that processes data, performs calculations, and executes programs based on instructions.",
    "what is programming": "Programming is the process of writing instructions (code) that a computer can execute to perform specific tasks.",
    
    # Mathematics
    "what is pi": "Pi (π) is a mathematical constant approximately equal to 3.14159. It represents the ratio of a circle's circumference to its diameter.",
    "what is calculus": "Calculus is a branch of mathematics studying continuous change, including differentiation and integration.",
    "what is algebra": "Algebra is a branch of mathematics using symbols and letters to represent numbers and express mathematical relationships.",
    "what is geometry": "Geometry is the branch of mathematics concerned with properties of space, shapes, sizes, and relative positions of figures.",
    
    # Astronomy
    "what is the sun": "The Sun is a star at the center of our solar system. It's a massive ball of hot plasma providing light and heat to Earth.",
    "what is the moon": "The Moon is Earth's natural satellite that orbits our planet. It takes about 27.3 days to complete one orbit.",
    "what is a galaxy": "A galaxy is a massive system of stars, gas, dust, and dark matter bound together by gravity. Our galaxy is the Milky Way.",
    "what is a planet": "A planet is a large celestial body that orbits a star, is roughly spherical, and has cleared its orbital path.",
    "what is a star": "A star is a massive, luminous sphere of plasma held together by gravity, powered by nuclear fusion in its core.",
    
    # History
    "when was world war 2": "World War II was fought from 1939 to 1945. It was the deadliest conflict in human history.",
    "when was world war 1": "World War I was fought from 1914 to 1918. It involved most of the world's great powers.",
    "who was albert einstein": "Albert Einstein (1879-1955) was a theoretical physicist who developed the theory of relativity and won the Nobel Prize in 1921.",
    "what is the renaissance": "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th century, marking the transition to modernity.",
    
    # Philosophy/Society
    "what is ethics": "Ethics is the study of moral principles that govern behavior and decision-making. It examines concepts of right and wrong.",
    "what is logic": "Logic is the study of reasoning and argumentation. It provides principles for evaluating valid inferences and sound arguments.",
    "what is democracy": "Democracy is a system of government where citizens vote for their representatives and leaders through free and fair elections.",
    
    # Economics
    "what is inflation": "Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power.",
    "what is gdp": "GDP (Gross Domestic Product) is the total monetary value of all goods and services produced in a country during a specific period.",
    
    # Misc
    "what are atoms": "Atoms are the basic building blocks of matter. They consist of a nucleus (protons and neutrons) surrounded by electrons.",
    "what is temperature": "Temperature is a measure of the average kinetic energy of particles in a substance. It indicates how hot or cold something is.",
    "what is celsius": "Celsius is a temperature scale where water freezes at 0°C and boils at 100°C at standard atmospheric pressure.",
    "what is the periodic table": "The periodic table is a chart organizing all known chemical elements by atomic number, electron configuration, and recurring properties.",
}

def normalize_query(query):
    """Normalize a query for matching"""
    query = query.lower().strip()
    # Remove question marks and common prefixes, but keep the core question
    query = re.sub(r'\?+$', '', query)
    query = re.sub(r'^(what is|what are|what does|tell me about|explain|define|describe)\s+(the|a|an)?\s*', '', query)
    query = query.strip()
    
    # Remove duplicate consecutive words
    words = query.split()
    normalized_words = []
    prev_word = None
    for word in words:
        if word != prev_word:
            normalized_words.append(word)
            prev_word = word
    
    return ' '.join(normalized_words)

def similarity(a, b):
    """Calculate similarity between two strings (0-1)"""
    return SequenceMatcher(None, a, b).ratio()

def get_answer(query, threshold=0.8):
    """
    Get answer for a query with fuzzy matching.
    
    Args:
        query: User's question
        threshold: Minimum similarity score (0-1)
    
    Returns:
        Answer string or None if no match
    """
    if not query or len(query) < 3:
        return None
    
    normalized_query = normalize_query(query)
    
    # Try exact match with normalized keys
    for key, answer in QA_DATABASE.items():
        if normalize_query(key) == normalized_query:
            return answer
    
    # Try fuzzy matching with normalized keys
    best_match = None
    best_score = 0
    
    for key, answer in QA_DATABASE.items():
        normalized_key = normalize_query(key)
        score = similarity(normalized_query, normalized_key)
        if score > best_score:
            best_score = score
            best_match = answer
    
    if best_score >= threshold:
        return best_match
    
    return None

def is_available():
    """Check if simple Q&A is available (always true)"""
    return True

def list_topics():
    """Get list of available topics"""
    topics = set()
    for query in QA_DATABASE.keys():
        words = query.split()
        if len(words) > 2:
            # Extract meaningful topic (last 1-2 words)
            topics.add(' '.join(words[-2:]))
    return sorted(topics)
