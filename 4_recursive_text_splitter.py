# from langchain.text_splitter import Chara

from langchain_text_splitters import RecursiveCharacterTextSplitter


tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""

# recursive character text splitter is better because it gives us an array of separators that can be used to split text based on how the text to chunk appears eg "\n\n" double line, " " an empty space, "." a period or fullstop etc   
recursive_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", ". ", " ", ""],
    chunk_size = 100,
    chunk_overlap = 0
)

chunks = recursive_splitter.split_text(tesla_text)

for index, chunk in enumerate(chunks):
    print(f"The chunk at {index} is {len(chunk)} characters")
    print(f"\n {chunk}")