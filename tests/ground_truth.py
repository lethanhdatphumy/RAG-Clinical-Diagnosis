""""query": Patient symptoms (string)
"expected_diagnosis": The disease name (string)
"expected_keywords": List of medical terms that should appear (list)
"""
from generation.rag_generator import ClinicalRAG

GROUND_TRUTH = [
    {
        "query": "20-year-old Sudanese refugee in Uganda with fever, bleeding from gums, vomiting, diarrhea, and subconjunctival hemorrhage during outbreak",
        "expected_diagnosis": "ebola",
        "expected_keywords": ["ebola", "hemorrhagic", "filovirus", "viral HF"]
    },
    {
        "query": "45-year-old farmer from Bangladesh with prolonged fever, massive splenomegaly, pancytopenia, and darkened skin after sandfly bites",
        "expected_diagnosis": "visceral leishmaniasis",
        "expected_keywords": ["leishmania", "kala-azar", "sandfly", "splenomegaly"]
    },
    {
        "query": "38-year-old from Philippines with fever, jaundice, calf pain, and renal failure after wading through floodwater with rats",
        "expected_diagnosis": "leptospirosis",
        "expected_keywords": ["leptospira", "weil's disease", "jaundice", "renal failure"]
    },
    {
        "query": "28-year-old pregnant woman from Brazil with mild fever, rash, conjunctivitis, and arthralgia after mosquito bites",
        "expected_diagnosis": "zika virus infection",
        "expected_keywords": ["zika", "flavivirus", "microcephaly", "aedes mosquito"]
    },
    {
        "query": "32-year-old safari guide from Tanzania with cyclical fever every 48 hours, chills, sweating, and splenomegaly",
        "expected_diagnosis": "malaria",
        "expected_keywords": ["malaria", "plasmodium", "paroxysmal fever", "splenomegaly"]
    },
    {
        "query": "50-year-old homeless man with chronic cough, hemoptysis, night sweats, weight loss, and cavitary lung lesion",
        "expected_diagnosis": "tuberculosis",
        "expected_keywords": ["tuberculosis", "mycobacterium", "cavitary", "pulmonary"]
    },
    {
        "query": "24-year-old student in dormitory with sudden fever, severe headache, neck stiffness, photophobia, and petechial rash",
        "expected_diagnosis": "meningitis",
        "expected_keywords": ["meningitis", "meningococcal", "petechial", "nuchal rigidity"]
    },
    {
        "query": "36-year-old returning from Caribbean with high fever and severe debilitating joint pain in wrists and ankles after daytime mosquito bites",
        "expected_diagnosis": "chikungunya",
        "expected_keywords": ["chikungunya", "alphavirus", "arthralgia", "aedes"]
    },
    {
        "query": "42-year-old from Amazon rainforest with jaundice, hemorrhage, coffee-ground vomiting, and slow pulse despite high fever, unvaccinated",
        "expected_diagnosis": "yellow fever",
        "expected_keywords": ["yellow fever", "flavivirus", "jaundice", "hemorrhagic"]
    },
    {
        "query": "31-year-old from South Asia with stepladder fever, rose spots, relative bradycardia, and hepatosplenomegaly after eating street food",
        "expected_diagnosis": "typhoid fever",
        "expected_keywords": ["typhoid", "salmonella typhi", "enteric fever", "rose spots"]
    },
    {
        "query": "29-year-old IV drug user with weight loss, oral thrush, lymphadenopathy, recurrent herpes zoster, and low CD4 count",
        "expected_diagnosis": "hiv",
        "expected_keywords": ["hiv", "aids", "immunodeficiency", "opportunistic infection"]
    },
    {
        "query": "27-year-old from West Africa with painless non-healing ulcer on forearm with raised border after insect bites",
        "expected_diagnosis": "cutaneous leishmaniasis",
        "expected_keywords": ["leishmaniasis", "cutaneous", "sandfly", "ulcer"]
    },
    {
        "query": "30-year-old returning from Thailand with high fever, retro-orbital pain, rash, bleeding gums, and low platelets",
        "expected_diagnosis": "dengue fever",
        "expected_keywords": ["dengue", "hemorrhagic", "thrombocytopenia", "aedes"]
    },
    {
        "query": "26-year-old after swimming in Lake Kariba with bloody diarrhea, hepatosplenomegaly, hematuria, and high eosinophils",
        "expected_diagnosis": "schistosomiasis",
        "expected_keywords": ["schistosomiasis", "bilharzia", "freshwater", "eosinophilia"]
    },
    {
        "query": "44-year-old on steroids from Cambodia with diarrhea, wheezing, moving skin rash, extremely high eosinophils, and larvae in stool",
        "expected_diagnosis": "strongyloidiasis",
        "expected_keywords": ["strongyloides", "hyperinfection", "larva currens", "eosinophilia"]
    }
]

test_rag = ClinicalRAG()

for i, query in enumerate(GROUND_TRUTH):
    print(f"\n{'=' * 80}")
    print(f"TESTING {i}")
    print(f"\n{'=' * 80}")

    test_result = test_rag.query(query['query'])

    print(f"\nDIAGNOSIS:")
    print(test_result['result'])
    print("...\n")
    for j, doc in enumerate(test_result['source_documents'],
                            1):  # 1 is the starting number for enumeration (default = 0).
        print(f"\n{i}. {doc.metadata['case_id']}")
        print(f"   {doc.page_content}...")
