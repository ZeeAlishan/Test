import chromadb
import chromadb.utils.embedding_functions as embedding_functions


chroma_client = chromadb.Client()
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

collection = chroma_client.create_collection(name="my_collection", embedding_function=ef)

collection.add(
    #Courses offered at GSU & GT
    documents=[
        "Calculus of Single Variable",
        "Survey of US History",
        "Principles of Physics I",
        "Principles of Physics I Lab",
        "English Composition I",
        "Linear Algebra",
        "Introduction to Object Oriented Programming",
        "Principles of Physics II",
        "Principles of Physics II Lab",
        "Multivariable Calculus",
        "English Composition II",
        "Introduction to Philosophy"
    ],
    ids=["MATH2211",
         "HIST2110",
         "PHYS2211",
         "PHYS2211L",
         "ENGL1101",
         "MATH1554",
         "CS1331",
         "PHYS2212",
         "PHYS2212L",
         "MATH2551",
         "ENGL1102",
         "PHIL2010"
         ]
)

results = collection.query(
    query_texts=["jacobian"], # Chroma will embed this for you
    n_results=5, # how many results to return
    include=['distances','documents']
)
print(results)
