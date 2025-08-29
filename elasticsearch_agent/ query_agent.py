import os
import json
from elasticsearch import Elasticsearch, ElasticsearchWarning
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
INDEX = "my_index"

es = Elasticsearch(os.getenv('ES_URL'))


def get_mapping_tool(_: str) -> str:
    """Get index mapping (schema)."""
    try:
        mapping = es.indices.get_mapping(index=INDEX)
        return json.dumps(mapping, indent=4)
    except Exception as e:
        return f"Failed to fetch mapping: {str(e)}"


tools = [
    Tool(
        name="GetIndexMapping",
        func=get_mapping_tool,
        description="Fetch the index schema (fields and types) to check semantic correctness."
    )
]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt_template = """
You are an Elasticsearch Query Validator Agent.

You have access to tools:
- ElasticsearchSearch: checks syntax by running query.
- GetIndexMapping: fetches index schema.

Steps:
1. First, always check syntax with ElasticsearchSearch.
2. If syntax is valid, fetch mapping with GetIndexMapping.
3. Based on query + schema, decide if the query is semantically correct.
4. Output strictly in JSON with:
   {{
     "syntax_valid": true/false,
     "semantic_valid": true/false,
     "reason": "short explanation"
   }}

Query to validate:
{query}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["query"])

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

if __name__ == "__main__":
    query = {
        "query": {
            "match": {
                "name": "sai krishna"
            }
        }
    }

    result = agent.run(prompt.format(query=json.dumps(query, indent=4)))
    print("Final decision:", result)
