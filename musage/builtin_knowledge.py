"""Built-in knowledge base for common queries — instant answers without web search"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# § 1  WORD-TO-NUMBER CONVERTER
# ═══════════════════════════════════════════════════════════════════════════════

_WORD_TO_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
    'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
    'million': 1000000, 'billion': 1000000000,
}

_OPERATORS = {
    'plus': '+', 'add': '+', 'added to': '+', 'sum of': '+',
    'minus': '-', 'subtract': '-', 'less': '-', 'difference': '-',
    'times': '*', 'multiply': '*', 'multiplied by': '*', 'product of': '*',
    'divided by': '/', 'divide': '/', 'over': '/',
}


def _words_to_number(words: str) -> Optional[int]:
    """Convert word numbers to integers. Returns None if not a valid word number."""
    words = words.lower().strip()
    if words in _WORD_TO_NUM:
        return _WORD_TO_NUM[words]
    
    # Handle compound numbers like "twenty five" or "ninety nine"
    parts = words.split()
    if len(parts) == 2:
        a, b = parts
        if a in _WORD_TO_NUM and b in _WORD_TO_NUM:
            val_a, val_b = _WORD_TO_NUM[a], _WORD_TO_NUM[b]
            if val_a >= 20 and val_a < 100 and val_b < 10:
                return val_a + val_b
    
    return None


def _parse_word_math(query: str) -> Optional[str]:
    """
    Parse word-based math queries like:
      - "five times two"
      - "what is ten plus three"
      - "twenty divided by four"
    
    Returns formatted result string or None if not a word math query.
    """
    q = query.lower().strip()
    
    # Remove common prefixes
    q = re.sub(r'^(what\s+is\s+|calculate\s+|compute\s+|what\'s\s+)', '', q).strip()
    q = re.sub(r'\?+$', '', q).strip()
    
    # Pattern: NUMBER OPERATOR NUMBER
    pattern = r'^([a-z]+(?:\s+[a-z]+)?)\s+(plus|add|minus|subtract|times|multiply|multiplied\s+by|divided\s+by|divide)\s+([a-z]+(?:\s+[a-z]+)?)$'
    m = re.match(pattern, q)
    
    if not m:
        return None
    
    num1_str, op_str, num2_str = m.groups()
    
    # Convert words to numbers
    num1 = _words_to_number(num1_str)
    num2 = _words_to_number(num2_str)
    
    if num1 is None or num2 is None:
        return None
    
    # Get operator
    op = _OPERATORS.get(op_str, _OPERATORS.get(op_str.replace(' by', ''), None))
    if op is None:
        return None
    
    # Calculate
    try:
        if op == '+':
            result = num1 + num2
        elif op == '-':
            result = num1 - num2
        elif op == '*':
            result = num1 * num2
        elif op == '/':
            if num2 == 0:
                return "Cannot divide by zero."
            result = num1 / num2
        else:
            return None
        
        # Format result
        if isinstance(result, float) and result == int(result):
            result = int(result)
        
        return f"{num1} {op} {num2} = {result}"
    
    except Exception as e:
        logger.warning(f"Word math calculation error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  COMMON KNOWLEDGE BASE (FAQ-style)
# ═══════════════════════════════════════════════════════════════════════════════

_KNOWLEDGE_BASE = {
    # ══════════════════════════════════════════════════════════════════════════
    # § MATHEMATICS & PHYSICS
    # ══════════════════════════════════════════════════════════════════════════
    'what is pi': (
        "Pi (π) is a mathematical constant representing the ratio of a circle's "
        "circumference to its diameter. It equals approximately 3.14159265359 and is an "
        "irrational number with infinite non-repeating decimal digits. Pi appears in many "
        "mathematical formulas across geometry, trigonometry, and calculus."
    ),
    
    'what is algebra': (
        "Algebra is a branch of mathematics dealing with symbols and the rules for "
        "manipulating those symbols. It involves solving equations and working with "
        "variables to represent unknown quantities. Algebra is fundamental to advanced "
        "mathematics and is widely used in science, engineering, and everyday problem-solving."
    ),
    
    'what is calculus': (
        "Calculus is a branch of mathematics focused on rates of change (differential calculus) "
        "and accumulation of quantities (integral calculus). Developed independently by Isaac "
        "Newton and Gottfried Leibniz in the 17th century, it's essential for physics, "
        "engineering, economics, and many scientific fields."
    ),
    
    'what is geometry': (
        "Geometry is a branch of mathematics concerned with shapes, sizes, positions, and "
        "properties of space. It studies points, lines, angles, surfaces, and solids. "
        "Euclidean geometry deals with flat surfaces, while non-Euclidean geometry explores "
        "curved spaces."
    ),
    
    'what is gravity': (
        "Gravity is a fundamental force of nature that attracts objects with mass toward "
        "one another. On Earth, it gives weight to physical objects and causes them to fall "
        "when dropped. Newton's law describes gravity as proportional to mass and inversely "
        "proportional to distance squared. Einstein's general relativity describes it as "
        "curvature of spacetime."
    ),
    
    'what is speed of light': (
        "The speed of light in vacuum is approximately 299,792,458 meters per second "
        "(about 186,282 miles per second). Denoted by 'c', it's a universal physical constant "
        "and the maximum speed at which all energy, matter, and information can travel. "
        "Einstein's theory of relativity is built on this constant."
    ),
    
    'what is energy': (
        "Energy is the capacity to do work or produce heat. It exists in various forms "
        "including kinetic (motion), potential (position), thermal (heat), electrical, "
        "chemical, and nuclear. The law of conservation of energy states that energy cannot "
        "be created or destroyed, only transformed from one form to another."
    ),
    
    'what is atom': (
        "An atom is the smallest unit of matter that retains the properties of an element. "
        "It consists of a nucleus (containing protons and neutrons) surrounded by electrons. "
        "Atoms are typically 0.1 to 0.5 nanometers in diameter and combine to form molecules "
        "and compounds."
    ),
    
    'what is quantum mechanics': (
        "Quantum mechanics is a fundamental theory in physics describing the behavior of "
        "matter and energy at atomic and subatomic scales. It introduces concepts like wave-"
        "particle duality, uncertainty principle, and quantum superposition, which differ "
        "significantly from classical physics."
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # § BIOLOGY & LIFE SCIENCES
    # ══════════════════════════════════════════════════════════════════════════
    'what is photosynthesis': (
        "Photosynthesis is the process by which plants, algae, and some bacteria convert "
        "light energy (usually from the sun) into chemical energy stored in glucose. "
        "The process occurs in chloroplasts using carbon dioxide and water, producing "
        "oxygen as a byproduct. The basic equation is: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂."
    ),
    
    'what is dna': (
        "DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions "
        "for the development, functioning, growth, and reproduction of all known organisms. "
        "It consists of two strands forming a double helix structure, with four chemical "
        "bases: adenine (A), thymine (T), guanine (G), and cytosine (C). DNA is replicated "
        "and passed from parents to offspring."
    ),
    
    'what is rna': (
        "RNA (ribonucleic acid) is a molecule similar to DNA that plays crucial roles in "
        "protein synthesis and gene regulation. Unlike DNA, RNA is typically single-stranded "
        "and uses uracil (U) instead of thymine (T). Main types include mRNA (messenger), "
        "tRNA (transfer), and rRNA (ribosomal)."
    ),
    
    'what is cell': (
        "A cell is the basic structural and functional unit of all living organisms. "
        "Cells can be prokaryotic (without a nucleus, like bacteria) or eukaryotic (with "
        "a nucleus, like plant and animal cells). They contain organelles that perform "
        "specific functions necessary for life."
    ),
    
    'what is evolution': (
        "Evolution is the process by which species of organisms change over time through "
        "variations in their genetic code. Charles Darwin's theory of natural selection "
        "explains how traits that enhance survival and reproduction become more common in "
        "populations over generations. Evidence comes from fossils, genetics, and comparative "
        "anatomy."
    ),
    
    'what is protein': (
        "Proteins are large, complex molecules made of amino acids that perform most of "
        "the work in cells. They serve as enzymes, structural components, antibodies, and "
        "hormones. The sequence of amino acids determines each protein's unique 3D structure "
        "and function. Proteins are essential for virtually all biological processes."
    ),
    
    'what is mitochondria': (
        "Mitochondria are organelles found in most eukaryotic cells, often called the "
        "'powerhouses of the cell.' They generate ATP (adenosine triphosphate) through "
        "cellular respiration, converting nutrients into usable energy. Mitochondria have "
        "their own DNA and are believed to have originated from ancient bacteria."
    ),
    
    'what is ecosystem': (
        "An ecosystem is a community of living organisms interacting with each other and "
        "their physical environment. It includes all plants, animals, microorganisms, soil, "
        "rocks, minerals, water, and air in a given area. Energy flows through ecosystems "
        "via food chains and food webs."
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # § CHEMISTRY
    # ══════════════════════════════════════════════════════════════════════════
    'what is periodic table': (
        "The periodic table is a tabular arrangement of chemical elements organized by "
        "atomic number, electron configuration, and recurring chemical properties. Created "
        "by Dmitri Mendeleev in 1869, it currently contains 118 confirmed elements. Elements "
        "are arranged in rows (periods) and columns (groups) based on their properties."
    ),
    
    'what is molecule': (
        "A molecule is a group of two or more atoms held together by chemical bonds. "
        "Molecules can consist of atoms of the same element (like O₂) or different elements "
        "(like H₂O). They are the smallest particles of a chemical compound that retain "
        "the chemical properties of that compound."
    ),
    
    'what is chemical reaction': (
        "A chemical reaction is a process where substances (reactants) are transformed into "
        "different substances (products) through the breaking and forming of chemical bonds. "
        "Reactions can release energy (exothermic) or absorb energy (endothermic). They "
        "follow the law of conservation of mass."
    ),
    
    'what is acid': (
        "An acid is a substance that donates protons (H⁺ ions) in solution or accepts "
        "electron pairs. Acids have a pH less than 7, taste sour, and turn blue litmus "
        "paper red. Common examples include hydrochloric acid (HCl), sulfuric acid (H₂SO₄), "
        "and citric acid found in citrus fruits."
    ),
    
    'what is base': (
        "A base is a substance that accepts protons or donates electron pairs in solution. "
        "Bases have a pH greater than 7, taste bitter, feel slippery, and turn red litmus "
        "paper blue. Common examples include sodium hydroxide (NaOH), ammonia (NH₃), and "
        "baking soda (sodium bicarbonate)."
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # § COMPUTER SCIENCE & TECHNOLOGY
    # ══════════════════════════════════════════════════════════════════════════
    'what is python': (
        "Python is a high-level, interpreted programming language known for its clear "
        "syntax and readability. Created by Guido van Rossum and released in 1991, it "
        "supports multiple programming paradigms including procedural, object-oriented, "
        "and functional programming. Python is widely used for web development, data science, "
        "artificial intelligence, automation, and scientific computing."
    ),
    
    'what is javascript': (
        "JavaScript is a versatile programming language primarily used for web development. "
        "Created by Brendan Eich in 1995, it enables interactive web pages and runs in "
        "browsers. Modern JavaScript (Node.js) also runs on servers. It's an essential "
        "technology alongside HTML and CSS for creating dynamic websites."
    ),
    
    'what is algorithm': (
        "An algorithm is a step-by-step procedure or set of rules for solving a problem "
        "or accomplishing a task. In computer science, algorithms are precise instructions "
        "that computers follow to process data, perform calculations, or make decisions. "
        "Algorithm efficiency is measured by time and space complexity."
    ),
    
    'what is database': (
        "A database is an organized collection of structured data stored electronically "
        "in a computer system. Databases are managed by Database Management Systems (DBMS) "
        "like MySQL, PostgreSQL, or MongoDB. They enable efficient data storage, retrieval, "
        "modification, and deletion using query languages like SQL."
    ),
    
    'what is api': (
        "API (Application Programming Interface) is a set of rules and protocols that "
        "allows different software applications to communicate with each other. APIs define "
        "methods and data formats that applications can use to request and exchange information. "
        "They enable integration between different systems and services."
    ),
    
    'what is machine learning': (
        "Machine learning is a branch of artificial intelligence focused on building systems "
        "that learn from data and improve their performance over time without explicit "
        "programming. It uses statistical techniques to enable computers to recognize patterns, "
        "make predictions, and make decisions. Types include supervised, unsupervised, and "
        "reinforcement learning."
    ),
    
    'what is ai': (
        "Artificial Intelligence (AI) is the simulation of human intelligence in machines "
        "programmed to think, learn, and solve problems. AI encompasses machine learning, "
        "natural language processing, computer vision, and robotics. Applications range "
        "from virtual assistants and recommendation systems to autonomous vehicles and "
        "medical diagnosis."
    ),
    
    'what is neural network': (
        "A neural network is a computing system inspired by biological neural networks "
        "in animal brains. It consists of interconnected nodes (neurons) organized in layers "
        "that process information and learn patterns from data. Deep neural networks with "
        "many layers power modern AI applications like image recognition and natural language "
        "processing."
    ),
    
    'what is cloud computing': (
        "Cloud computing is the delivery of computing services (servers, storage, databases, "
        "networking, software) over the internet ('the cloud'). It offers on-demand resources, "
        "scalability, and pay-as-you-go pricing. Major providers include AWS, Microsoft Azure, "
        "and Google Cloud. Types include IaaS, PaaS, and SaaS."
    ),
    
    'what is encryption': (
        "Encryption is the process of converting information into a coded format (ciphertext) "
        "to prevent unauthorized access. Only those with the correct decryption key can "
        "convert it back to readable form (plaintext). Encryption protects data security "
        "in communication, storage, and transactions. Common methods include AES and RSA."
    ),
    
    'what is blockchain': (
        "Blockchain is a distributed ledger technology that records transactions across "
        "multiple computers in a way that makes them secure, transparent, and tamper-resistant. "
        "Each block contains data, a timestamp, and a cryptographic hash of the previous block. "
        "Bitcoin and other cryptocurrencies use blockchain technology."
    ),
    
    'what is data structure': (
        "A data structure is a specialized format for organizing, storing, and managing "
        "data in computer memory. Common data structures include arrays, linked lists, stacks, "
        "queues, trees, graphs, and hash tables. Each has different strengths for specific "
        "operations like searching, inserting, or deleting data."
    ),
    
    'what is object oriented programming': (
        "Object-Oriented Programming (OOP) is a programming paradigm based on the concept "
        "of 'objects' that contain data (attributes) and code (methods). Key principles "
        "include encapsulation, inheritance, polymorphism, and abstraction. OOP promotes "
        "code reusability, modularity, and easier maintenance. Languages like Java, Python, "
        "and C++ support OOP."
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # § HISTORY & GEOGRAPHY
    # ══════════════════════════════════════════════════════════════════════════
    'what is world war 2': (
        "World War II (1939-1945) was a global conflict involving most of the world's nations, "
        "divided into two opposing alliances: the Allies (led by the US, UK, Soviet Union) "
        "and the Axis powers (Germany, Italy, Japan). It resulted in an estimated 70-85 million "
        "casualties and led to the formation of the United Nations."
    ),
    
    'what is renaissance': (
        "The Renaissance was a cultural, artistic, and intellectual movement in Europe "
        "from the 14th to 17th century, marking the transition from the Middle Ages to "
        "modernity. It began in Italy and featured renewed interest in classical Greek and "
        "Roman culture, major advances in art (Leonardo, Michelangelo), science, and exploration."
    ),
    
    'what is industrial revolution': (
        "The Industrial Revolution (roughly 1760-1840) was a period of major industrialization "
        "and innovation that began in Great Britain. It featured the transition from hand "
        "production to machines, development of steam power, iron production, and factory "
        "systems. It profoundly changed society, economy, and technology worldwide."
    ),
    
    'what is capital of france': (
        "Paris is the capital and largest city of France. Located in north-central France "
        "on the Seine River, it's known for its art, culture, fashion, and landmarks like "
        "the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris has been a major "
        "center of finance, commerce, and culture for over 2,000 years."
    ),
    
    'what is equator': (
        "The equator is an imaginary line around Earth's center at 0° latitude, dividing "
        "the planet into Northern and Southern Hemispheres. It's approximately 40,075 "
        "kilometers (24,901 miles) long. Regions near the equator experience consistent "
        "warm temperatures and have the shortest day length variations throughout the year."
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # § ECONOMICS & SOCIETY
    # ══════════════════════════════════════════════════════════════════════════
    'what is democracy': (
        "Democracy is a system of government in which power is vested in the people, who "
        "rule either directly or through freely elected representatives. Key principles "
        "include equality, freedom of speech, fair elections, rule of law, and protection "
        "of individual rights. Modern democracies typically operate through representative "
        "institutions like parliaments and congresses."
    ),
    
    'what is capitalism': (
        "Capitalism is an economic system based on private ownership of the means of "
        "production and their operation for profit. Key features include free markets, "
        "competition, capital accumulation, wage labor, and voluntary exchange. Prices "
        "are determined by supply and demand, and economic decisions are decentralized."
    ),
    
    'what is gdp': (
        "GDP (Gross Domestic Product) is the total monetary value of all finished goods "
        "and services produced within a country's borders in a specific time period. It's "
        "the primary indicator used to gauge a nation's economic health and size. GDP can "
        "be calculated using production, income, or expenditure approaches."
    ),
    
    'what is inflation': (
        "Inflation is the rate at which the general level of prices for goods and services "
        "rises, reducing purchasing power over time. It's typically measured by the Consumer "
        "Price Index (CPI). Moderate inflation is normal in growing economies, but high "
        "inflation (hyperinflation) can be economically destructive."
    ),
    
    'what is supply and demand': (
        "Supply and demand is a fundamental economic model describing how prices are "
        "determined in market economies. Supply represents how much producers are willing "
        "to sell at various prices, while demand represents how much consumers are willing "
        "to buy. The equilibrium price occurs where supply equals demand."
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # § ENVIRONMENTAL & EARTH SCIENCE
    # ══════════════════════════════════════════════════════════════════════════
    'what is climate change': (
        "Climate change refers to long-term shifts in global temperatures and weather patterns. "
        "While climate naturally varies, scientific evidence shows that human activities—"
        "particularly burning fossil fuels and deforestation—have been the dominant driver "
        "of warming since the mid-20th century. Effects include rising sea levels, extreme "
        "weather events, ocean acidification, and ecosystem disruption."
    ),
    
    'what is greenhouse effect': (
        "The greenhouse effect is a natural process where certain gases in Earth's atmosphere "
        "trap heat from the sun, keeping the planet warm enough to support life. Greenhouse "
        "gases include carbon dioxide, methane, and water vapor. Human activities have "
        "intensified this effect, contributing to global warming."
    ),
    
    'what is ozone layer': (
        "The ozone layer is a region of Earth's stratosphere containing high concentrations "
        "of ozone (O₃) molecules. Located 15-35 kilometers above Earth's surface, it absorbs "
        "most of the sun's harmful ultraviolet radiation, protecting life on Earth. The "
        "ozone layer has been damaged by human-made chemicals like CFCs."
    ),
    
    'what is water cycle': (
        "The water cycle (hydrologic cycle) is the continuous movement of water on, above, "
        "and below Earth's surface. It involves evaporation from oceans and land, condensation "
        "into clouds, precipitation as rain or snow, and runoff back to bodies of water. "
        "This cycle is crucial for distributing water and regulating Earth's temperature."
    ),
    
    'what is tectonic plates': (
        "Tectonic plates are large, rigid pieces of Earth's lithosphere that fit together "
        "like a jigsaw puzzle and float on the semi-fluid asthenosphere beneath. There are "
        "about 15 major plates and many smaller ones. Their movement causes earthquakes, "
        "volcanic activity, mountain formation, and ocean trench creation."
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # § ASTRONOMY & SPACE
    # ══════════════════════════════════════════════════════════════════════════
    'what is solar system': (
        "The Solar System consists of the Sun and all objects bound to it by gravity, "
        "including eight planets, their moons, dwarf planets, asteroids, comets, and "
        "meteoroids. It formed about 4.6 billion years ago from a giant molecular cloud. "
        "The planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."
    ),
    
    'what is black hole': (
        "A black hole is a region of spacetime where gravity is so strong that nothing, "
        "not even light, can escape from it. They form when massive stars collapse at the "
        "end of their life cycle. The boundary around a black hole is called the event "
        "horizon. Black holes are predicted by Einstein's general theory of relativity."
    ),
    
    'what is galaxy': (
        "A galaxy is a massive system of stars, stellar remnants, interstellar gas, dust, "
        "and dark matter, bound together by gravity. Our Milky Way contains 100-400 billion "
        "stars. Galaxies are classified by shape: spiral (like Milky Way), elliptical, and "
        "irregular. The universe contains an estimated 2 trillion galaxies."
    ),
    
    'what is big bang': (
        "The Big Bang theory is the prevailing cosmological model explaining the universe's "
        "origin. It states that the universe began as an extremely hot, dense point about "
        "13.8 billion years ago and has been expanding ever since. Evidence includes cosmic "
        "microwave background radiation and the observed expansion of the universe."
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # § ADDITIONAL PROGRAMMING & DATA SCIENCE
    # ══════════════════════════════════════════════════════════════════════════
    'what is git': (
        "Git is a distributed version control system used to track changes in source code "
        "during software development. Created by Linus Torvalds in 2005, it allows multiple "
        "developers to work on the same project simultaneously. Git features branching, "
        "merging, and distributed workflows. GitHub, GitLab, and Bitbucket are popular platforms "
        "that use Git."
    ),
    
    'what is sql': (
        "SQL (Structured Query Language) is a standard programming language designed for "
        "managing and manipulating relational databases. It allows users to create, read, "
        "update, and delete data (CRUD operations). Common SQL commands include SELECT, "
        "INSERT, UPDATE, DELETE, JOIN, and WHERE. Major database systems include MySQL, "
        "PostgreSQL, Oracle, and Microsoft SQL Server."
    ),
    
    'what is html': (
        "HTML (HyperText Markup Language) is the standard markup language for creating web "
        "pages. It uses tags to structure content like headings, paragraphs, links, images, "
        "and other elements. HTML works with CSS (styling) and JavaScript (interactivity) "
        "to create modern websites. The latest version is HTML5, which added multimedia "
        "support and semantic elements."
    ),
    
    'what is css': (
        "CSS (Cascading Style Sheets) is a stylesheet language used to describe the "
        "presentation and design of HTML documents. It controls layout, colors, fonts, "
        "spacing, and responsive design. CSS allows separation of content (HTML) from "
        "presentation, making websites easier to maintain and more consistent."
    ),
    
    'what is react': (
        "React is a popular JavaScript library for building user interfaces, particularly "
        "single-page applications. Developed by Facebook (Meta) in 2013, it uses a component-"
        "based architecture and virtual DOM for efficient rendering. React's declarative "
        "approach and reusable components make it a preferred choice for modern web development."
    ),
    
    'what is docker': (
        "Docker is a platform for developing, shipping, and running applications in containers. "
        "Containers package application code with all dependencies, ensuring consistency across "
        "different environments. Docker enables microservices architecture, simplifies deployment, "
        "and improves scalability. It's widely used in DevOps and cloud computing."
    ),
    
    'what is recursion': (
        "Recursion is a programming technique where a function calls itself to solve a problem "
        "by breaking it into smaller instances of the same problem. Each recursive call works "
        "on a simpler version until reaching a base case that stops the recursion. Common "
        "applications include tree traversal, factorial calculation, and sorting algorithms "
        "like quicksort and mergesort."
    ),
    
    'what is binary search': (
        "Binary search is an efficient algorithm for finding a target value in a sorted array. "
        "It repeatedly divides the search interval in half: if the target is less than the "
        "middle element, search the left half; otherwise search the right half. Time complexity "
        "is O(log n), making it much faster than linear search for large datasets."
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # § HEALTH & MEDICINE
    # ══════════════════════════════════════════════════════════════════════════
    'what is virus': (
        "A virus is a microscopic infectious agent that can only replicate inside living cells "
        "of organisms. Viruses consist of genetic material (DNA or RNA) enclosed in a protein "
        "coat. They cause diseases ranging from the common cold to COVID-19, HIV/AIDS, and "
        "influenza. Unlike bacteria, viruses cannot be treated with antibiotics."
    ),
    
    'what is bacteria': (
        "Bacteria are single-celled microorganisms found almost everywhere on Earth. While "
        "some bacteria cause diseases (like strep throat or tuberculosis), most are harmless "
        "or beneficial. They play crucial roles in digestion, nitrogen fixation, and "
        "decomposition. Bacteria reproduce through binary fission and can be treated with "
        "antibiotics."
    ),
    
    'what is immune system': (
        "The immune system is a complex network of cells, tissues, and organs that defend "
        "the body against harmful pathogens like bacteria, viruses, and parasites. It includes "
        "white blood cells, antibodies, the lymphatic system, and organs like the thymus and "
        "spleen. The immune system has innate (immediate) and adaptive (learned) responses."
    ),
    
    'what is vaccine': (
        "A vaccine is a biological preparation that provides immunity to a specific infectious "
        "disease. Vaccines contain weakened or inactive pathogens, or parts of pathogens, that "
        "stimulate the immune system to produce antibodies without causing the disease. "
        "Vaccination has eliminated or greatly reduced diseases like smallpox, polio, and measles."
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # § PHILOSOPHY & LOGIC
    # ══════════════════════════════════════════════════════════════════════════
    'what is philosophy': (
        "Philosophy is the study of fundamental questions about existence, knowledge, values, "
        "reason, mind, and language. Major branches include metaphysics (nature of reality), "
        "epistemology (theory of knowledge), ethics (moral principles), logic (reasoning), "
        "and aesthetics (beauty and art). Famous philosophers include Plato, Aristotle, "
        "Descartes, Kant, and Nietzsche."
    ),
    
    'what is logic': (
        "Logic is the systematic study of valid reasoning and argumentation. It examines "
        "the principles that distinguish correct from incorrect reasoning. Formal logic uses "
        "symbolic systems to represent arguments, while informal logic deals with everyday "
        "reasoning. Key concepts include premises, conclusions, validity, and soundness."
    ),
    
    'what is ethics': (
        "Ethics is the branch of philosophy concerned with moral principles, right and wrong "
        "conduct, and what constitutes a good life. Major ethical theories include utilitarianism "
        "(greatest good for greatest number), deontology (duty-based ethics), virtue ethics "
        "(character-based), and consequentialism (outcome-based)."
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # § ARTS & CULTURE
    # ══════════════════════════════════════════════════════════════════════════
    'what is music theory': (
        "Music theory is the study of how music works, including the practices and possibilities "
        "of music. It encompasses elements like pitch, rhythm, harmony, melody, form, and "
        "timbre. Music theory helps musicians understand structure, composition, and analysis. "
        "Key concepts include scales, chords, intervals, and notation."
    ),
    
    'what is impressionism': (
        "Impressionism was an art movement that originated in France in the 1860s, emphasizing "
        "capturing the momentary effects of light and color. Impressionist painters like Monet, "
        "Renoir, and Degas used visible brushstrokes, open composition, and emphasized accurate "
        "depiction of light over realistic detail."
    ),
    
    'what is literature': (
        "Literature is written works, especially those considered to have artistic or intellectual "
        "value. Major forms include poetry, drama, fiction, and non-fiction prose. Literature "
        "explores human experiences, emotions, and ideas through language. It encompasses "
        "classics from Shakespeare and Homer to contemporary novels and creative writing."
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # § UNITS & MEASUREMENTS (Common Questions)
    # ══════════════════════════════════════════════════════════════════════════
    'how many seconds in a minute': (
        "There are 60 seconds in a minute. This is part of the sexagesimal (base-60) system "
        "inherited from ancient Babylonian mathematics."
    ),
    
    'how many minutes in an hour': (
        "There are 60 minutes in an hour. Combined with 60 seconds per minute, this means "
        "there are 3,600 seconds in an hour."
    ),
    
    'how many hours in a day': (
        "There are 24 hours in a day. This represents one complete rotation of Earth on its axis."
    ),
    
    'how many days in a year': (
        "There are 365 days in a regular year and 366 days in a leap year. Earth takes "
        "approximately 365.25 days to orbit the Sun, so we add a leap day every four years "
        "to keep our calendar aligned with Earth's orbit."
    ),
    
    'how many feet in a mile': (
        "There are 5,280 feet in one mile. A mile is an imperial unit of length commonly "
        "used in the United States and United Kingdom."
    ),
    
    'how many meters in a kilometer': (
        "There are 1,000 meters in a kilometer. The kilometer is a metric unit of length "
        "widely used around the world."
    ),
    
    'how many grams in a kilogram': (
        "There are 1,000 grams in a kilogram. The kilogram is the base unit of mass in "
        "the International System of Units (SI)."
    ),
    
    'how many pounds in a kilogram': (
        "There are approximately 2.205 pounds in a kilogram. To convert kilograms to pounds, "
        "multiply by 2.205. To convert pounds to kilograms, multiply by 0.4536."
    ),
}


def _lookup_builtin(query: str) -> Optional[str]:
    """
    Look up query in built-in knowledge base with fuzzy matching.
    Returns answer string or None if not found.
    """
    # Normalize query
    q = query.lower().strip()
    q = re.sub(r'\?+$', '', q).strip()
    q = re.sub(r'\s+', ' ', q)
    
    # Direct lookup
    if q in _KNOWLEDGE_BASE:
        return _KNOWLEDGE_BASE[q]
    
    # Fuzzy matching for similar questions
    for key, answer in _KNOWLEDGE_BASE.items():
        # Extract main keywords
        key_words = set(re.findall(r'\w+', key)) - {'what', 'is', 'a', 'an', 'the', 'are', 'how', 'does', 'do'}
        query_words = set(re.findall(r'\w+', q)) - {'what', 'is', 'a', 'an', 'the', 'are', 'how', 'does', 'do'}
        
        # If all key words are in query, return answer
        if key_words and key_words.issubset(query_words):
            return answer
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  POLITE RESPONSE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

_POLITE_CANT_ANSWER_TEMPLATES = [
    (
        "I don't have built-in knowledge about {topic}, but I'd be happy to search "
        "for that information online if you'd like. Would you like me to search for it?"
    ),
    (
        "That's beyond my current built-in knowledge base. However, I can look that up "
        "for you through a web search. Shall I proceed with searching for information about {topic}?"
    ),
    (
        "I'm not familiar with {topic} in my immediate knowledge, but I can help you find "
        "reliable information about it. Would you like me to search the web?"
    ),
]


def _generate_polite_cant_answer(query: str) -> str:
    """
    Generate a polite response when the knowledge base doesn't have an answer.
    Extracts the topic from the query and offers to search.
    """
    # Try to extract the main topic from the query
    q = query.lower().strip()
    q = re.sub(r'\?+$', '', q).strip()
    
    # Remove common question prefixes
    topic = re.sub(
        r'^\s*(what\s+is\s+(a\s+|an\s+|the\s+)?|'
        r'tell\s+me\s+about\s+|explain\s+|describe\s+|'
        r'who\s+is\s+|when\s+was\s+|where\s+is\s+|'
        r'how\s+does\s+|why\s+is\s+)',
        '', q, flags=re.I
    ).strip()
    
    if not topic or len(topic) < 2:
        topic = "that"
    
    # Capitalize first letter if it's a proper topic
    if topic != "that" and len(topic) > 3:
        topic = f'"{topic}"'
    
    # Use a varied response template
    import random
    template = random.choice(_POLITE_CANT_ANSWER_TEMPLATES)
    return template.format(topic=topic)


def _get_general_guidance() -> str:
    """Return helpful guidance about what the system can do."""
    return (
        "I'm knowledgeable about many topics including:\n\n"
        "📚 **Science & Math**: Physics, chemistry, biology, mathematics\n"
        "💻 **Technology**: Programming languages, AI, databases, algorithms\n"
        "🌍 **Geography & History**: Countries, events, movements\n"
        "📊 **Economics**: Markets, finance, economic concepts\n"
        "🌱 **Environment**: Climate, ecology, Earth science\n"
        "🔢 **Math**: Word problems, calculations, conversions\n\n"
        "I can answer questions, explain concepts, and help with math problems. "
        "For topics I'm not sure about, I can search for information online. "
        "What would you like to know?"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def get_builtin_answer(query: str) -> Optional[str]:
    """
    Main entry point: check if query can be answered from built-in knowledge.
    
    Priority order:
      1. Word-based math (e.g., "five times two")
      2. FAQ-style knowledge base
      3. Returns None if not found (delegates to web search)
    
    Returns answer string or None if no match found.
    """
    # Try word math first
    math_answer = _parse_word_math(query)
    if math_answer:
        logger.info(f"Built-in answer: word math")
        return math_answer
    
    # Try knowledge base
    kb_answer = _lookup_builtin(query)
    if kb_answer:
        logger.info(f"Built-in answer: knowledge base")
        return kb_answer
    
    # Return None to delegate to web search
    return None


def get_polite_cant_answer(query: str) -> str:
    """
    Generate a polite response when the system can't answer.
    Used when both built-in knowledge and web search fail.
    """
    return _generate_polite_cant_answer(query)


def get_general_guidance() -> str:
    """
    Return helpful guidance about system capabilities.
    Used for help requests or when user seems confused.
    """
    return _get_general_guidance()


def format_answer_with_tone(answer: str, query: str) -> str:
    """
    Format an answer with appropriate conversational tone.
    Adds context or transitions when helpful.
    """
    # For definition queries, keep as-is (already well-formatted)
    if re.match(r'^\s*(what\s+is|define|explain)\b', query, re.I):
        return answer
    
    # For how-to queries, add helpful framing
    if re.match(r'^\s*how\s+(to|do|does)', query, re.I):
        if not answer.startswith("To ") and not answer.startswith("You can"):
            return f"Here's how: {answer}"
    
    return answer
