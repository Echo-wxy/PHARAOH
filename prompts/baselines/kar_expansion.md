You help a retriever search a reservoir development knowledge graph.
Node types: Region, Pattern, Well, Control. Relation types: refines, combines, transforms.

Request: {{query}}

Entities and relations retrieved for the request so far:
{{retrieved_context}}

1. Rewrite the request as an expanded search query that adds the domain terms, node attributes, and related entities a matching node description would contain.
2. List the node types and relation types that relevant nodes are most likely connected by.
Answer in JSON: {"expanded_query": "...", "node_types": [...], "relation_types": [...]}
