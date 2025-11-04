"""
Knowledge Agent for article recommendation and knowledge graph creation
"""
import spacy
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import logging
import config
from utils.data_models import KnowledgeResult


class KnowledgeAgent:
    """Agent responsible for knowledge graph creation and article recommendations"""
    
    def __init__(self):
        """Initialize the knowledge agent with NLP models and database connections"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = None
        self._load_embedding_model()
        
        # Initialize spaCy for NER
        self.nlp = None
        self._load_spacy_model()
        
        # Initialize Neo4j connection
        self.neo4j_driver = None
        self._connect_to_neo4j()
        
        # Cache for embeddings to avoid recomputation
        self.embedding_cache = {}
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for embeddings"""
        try:
            self.logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
    
    def _load_spacy_model(self):
        """Load spaCy model for Named Entity Recognition"""
        try:
            self.logger.info(f"Loading spaCy model: {config.SPACY_MODEL}")
            self.nlp = spacy.load(config.SPACY_MODEL)
            self.logger.info("spaCy model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading spaCy model: {e}")
            self.logger.info("Please install the model with: python -m spacy download en_core_web_sm")
    
    def _connect_to_neo4j(self):
        """Connect to Neo4j database"""
        if not config.NEO4J_PASSWORD:
            self.logger.warning("Neo4j password not configured - skipping Neo4j connection")
            return
            
        try:
            self.logger.info(f"Connecting to Neo4j at {config.NEO4J_URI}")
            self.neo4j_driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            
            # Test the connection
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            
            self.logger.info("Neo4j connection established successfully")
            
        except Exception as e:
            self.logger.error(f"Error connecting to Neo4j: {e}")
            self.neo4j_driver = None
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array with text embedding
        """
        if not self.embedding_model:
            return np.array([])
        
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            embedding = self.embedding_model.encode([text])[0]
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return np.array([])
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities from text using spaCy
        
        Args:
            text: Text to analyze
            
        Returns:
            List of dictionaries containing entity information
        """
        if not self.nlp:
            return []
        
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Focus on entities relevant to financial analysis
                relevant_labels = {'ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT', 'MONEY', 'PERCENT'}
                
                if ent.label_ in relevant_labels:
                    entities.append({
                        'text': ent.text.strip(),
                        'label': ent.label_,
                        'description': spacy.explain(ent.label_) or ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
            
            # Remove duplicates
            seen = set()
            unique_entities = []
            for entity in entities:
                key = (entity['text'].lower(), entity['label'])
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)
            
            return unique_entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []

    def _extract_events(self, text: str) -> List[Dict[str, str]]:
        """Simple keyword-based event extraction from text"""
        if not text:
            return []
        events: List[Dict[str, str]] = []
        try:
            lower = text.lower()
            keyword_map = {
                'earnings': ['earnings', 'q1', 'q2', 'q3', 'q4', 'quarterly results'],
                'product_launch': ['launch', 'unveil', 'release'],
                'acquisition': ['acquires', 'acquisition', 'buy', 'merger', 'merges'],
                'guidance': ['guidance', 'forecast', 'outlook'],
                'regulatory': ['sec', 'lawsuit', 'regulator', 'regulatory', 'fine']
            }
            found = set()
            for label, keys in keyword_map.items():
                if any(k in lower for k in keys):
                    found.add(label)
            for ev in found:
                events.append({'text': ev.replace('_', ' ').title(), 'label': 'EVENT'})
            return events
        except Exception as e:
            self.logger.error(f"Error extracting events: {e}")
            return []
    
    def _create_cypher_queries(self, entities: List[Dict[str, str]], 
                             article_title: str) -> List[str]:
        """
        Generate Cypher queries to create nodes and relationships
        
        Args:
            entities: List of extracted entities
            article_title: Title of the article for context
            
        Returns:
            List of Cypher query strings
        """
        queries = []
        
        # Create article node
        article_query = f"""
        MERGE (article:Article {{title: $article_title}})
        SET article.created_at = datetime()
        RETURN article
        """
        queries.append(article_query)
        
        # Create entity nodes and relationships
        for entity in entities:
            entity_text = entity['text'].replace("'", "\\'")
            entity_label = entity['label']
            
            # Create entity node based on type
            if entity_label == 'ORG':
                node_query = f"""
                MERGE (entity:Company {{name: $entity_name}})
                SET entity.type = 'organization'
                RETURN entity
                """
            elif entity_label == 'PERSON':
                node_query = f"""
                MERGE (entity:Person {{name: $entity_name}})
                SET entity.type = 'person'
                RETURN entity
                """
            elif entity_label == 'PRODUCT':
                node_query = f"""
                MERGE (entity:Product {{name: $entity_name}})
                SET entity.type = 'product'
                RETURN entity
                """
            else:
                node_query = f"""
                MERGE (entity:Entity {{name: $entity_name, type: $entity_type}})
                RETURN entity
                """
            
            queries.append(node_query)
            
            # Create relationship between article and entity
            relationship_query = f"""
            MATCH (article:Article {{title: $article_title}})
            MATCH (entity {{name: $entity_name}})
            MERGE (article)-[:MENTIONS]->(entity)
            """
            queries.append(relationship_query)
        
        return queries
    
    def recommend_articles(self, articles: List[Dict[str, str]], 
                          query_context: str = "stock market analysis") -> List[Dict[str, str]]:
        """
        Recommend the most relevant articles based on semantic similarity
        
        Args:
            articles: List of article dictionaries
            query_context: Context for finding relevant articles
            
        Returns:
            List of top recommended articles
        """
        self.logger.info(f"Analyzing {len(articles)} articles for recommendations")
        
        if not articles or not self.embedding_model:
            return []
        
        try:
            # Get embedding for the query context
            query_embedding = self._get_text_embedding(query_context)
            
            if len(query_embedding) == 0:
                return articles[:config.TOP_ARTICLES]  # Return first N articles as fallback
            
            # Calculate embeddings and similarities for each article
            article_scores = []
            
            for article in articles:
                # Combine title and description for better context
                article_text = f"{article.get('title', '')} {article.get('description', '')}"
                
                if not article_text.strip():
                    continue
                
                article_embedding = self._get_text_embedding(article_text)
                
                if len(article_embedding) > 0:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        article_embedding.reshape(1, -1)
                    )[0][0]
                    
                    article_scores.append((article, float(similarity)))
            
            # Sort by similarity score (descending)
            article_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top N articles
            top_articles = [article for article, score in article_scores[:config.TOP_ARTICLES]]
            
            self.logger.info(f"Recommended {len(top_articles)} articles")
            return top_articles
            
        except Exception as e:
            self.logger.error(f"Error recommending articles: {e}")
            return articles[:config.TOP_ARTICLES]  # Return first N as fallback
    
    def update_knowledge_graph(self, articles: List[Dict[str, str]]) -> Dict[str, List]:
        """
        Update the knowledge graph with entities and relationships from articles
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Dictionary with extracted entities and created relationships
        """
        self.logger.info(f"Updating knowledge graph with {len(articles)} articles")
        
        all_entities = []
        all_relationships = []
        
        if not self.neo4j_driver:
            self.logger.warning("Neo4j not available - only extracting entities")
        
        for i, article in enumerate(articles):
            try:
                article_title = article.get('title', f'Article_{i}')
                article_content = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
                
                # Extract entities from article
                entities = self._extract_entities(article_content)
                events = self._extract_events(article_content)
                all_entities.extend(entities)
                
                # Update Neo4j if available
                if self.neo4j_driver and (entities or events):
                    queries = self._create_cypher_queries(entities, article_title)
                    # Ensure Event nodes
                    for event in events:
                        event_node_query = """
                        MERGE (e:Event {name: $event_name})
                        SET e.type = 'event'
                        RETURN e
                        """
                        queries.append(event_node_query)
                    
                    with self.neo4j_driver.session() as session:
                        for query in queries:
                            try:
                                if 'article_title' in query:
                                    session.run(query, article_title=article_title)
                                elif 'entity_name' in query and entities:
                                    for entity in entities:
                                        session.run(query, 
                                                  entity_name=entity['text'],
                                                  entity_type=entity['label'],
                                                  article_title=article_title)
                                        
                                        # Track relationships
                                        all_relationships.append({
                                            'article': article_title,
                                            'entity': entity['text'],
                                            'entity_type': entity['label'],
                                            'relationship': 'MENTIONS'
                                        })
                                elif 'event_name' in query and events:
                                    for event in events:
                                        session.run(query, event_name=event['text'])
                            except Exception as query_error:
                                self.logger.error(f"Error executing query: {query_error}")
                        # Create impacted_by relationships between Company and Event
                        if entities and events:
                            for entity in entities:
                                if entity['label'] == 'ORG':
                                    for event in events:
                                        try:
                                            session.run(
                                                """
                                                MATCH (c:Company {name: $company}), (e:Event {name: $event})
                                                MERGE (c)-[:IMPACTED_BY]->(e)
                                                """,
                                                company=entity['text'], event=event['text']
                                            )
                                            all_relationships.append({
                                                'company': entity['text'],
                                                'event': event['text'],
                                                'relationship': 'IMPACTED_BY'
                                            })
                                        except Exception as rel_err:
                                            self.logger.error(f"Error creating IMPACTED_BY: {rel_err}")
                
            except Exception as e:
                self.logger.error(f"Error processing article {i}: {e}")
        
        # Create inter-entity relationships based on co-occurrence
        if self.neo4j_driver:
            self._create_entity_relationships()
        
        self.logger.info(f"Knowledge graph updated with {len(all_entities)} entities and {len(all_relationships)} relationships")
        
        return {
            'entities': all_entities,
            'relationships': all_relationships
        }
    
    def _create_entity_relationships(self):
        """Create relationships between entities that frequently co-occur"""
        if not self.neo4j_driver:
            return
        
        try:
            with self.neo4j_driver.session() as session:
                # Find companies and people that are mentioned together
                relationship_query = """
                MATCH (article:Article)-[:MENTIONS]->(company:Company)
                MATCH (article)-[:MENTIONS]->(person:Person)
                WHERE company <> person
                MERGE (person)-[:ASSOCIATED_WITH]->(company)
                """
                session.run(relationship_query)
                
                # Find products associated with companies
                product_query = """
                MATCH (article:Article)-[:MENTIONS]->(company:Company)
                MATCH (article)-[:MENTIONS]->(product:Product)
                WHERE company <> product
                MERGE (company)-[:PRODUCES]->(product)
                """
                session.run(product_query)
                
        except Exception as e:
            self.logger.error(f"Error creating entity relationships: {e}")
    
    def query_knowledge_graph(self, query: str) -> List[Dict]:
        """
        Query the knowledge graph with Cypher
        
        Args:
            query: Cypher query string
            
        Returns:
            List of query results
        """
        if not self.neo4j_driver:
            self.logger.warning("Neo4j not available")
            return []
        
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(query)
                return [record.data() for record in result]
        except Exception as e:
            self.logger.error(f"Error querying knowledge graph: {e}")
            return []
    
    def analyze(self, articles: List[Dict[str, str]], ticker: str = "") -> KnowledgeResult:
        """
        Perform complete knowledge analysis on articles
        
        Args:
            articles: List of article dictionaries
            ticker: Stock ticker for context
            
        Returns:
            KnowledgeResult with recommendations and graph data
        """
        self.logger.info("Starting knowledge analysis")
        
        # Generate query context based on ticker
        query_context = f"{ticker} stock market financial analysis earnings news" if ticker else "stock market analysis"
        
        # Get article recommendations
        recommended_articles = self.recommend_articles(articles, query_context)
        
        # Update knowledge graph
        graph_data = self.update_knowledge_graph(recommended_articles)
        
        # Generate summary
        entity_count = len(graph_data.get('entities', []))
        relationship_count = len(graph_data.get('relationships', []))
        
        graph_summary = (f"Processed {len(recommended_articles)} articles, "
                        f"extracted {entity_count} entities, "
                        f"created {relationship_count} relationships in knowledge graph.")
        
        result = KnowledgeResult(
            recommended_articles=recommended_articles,
            entities_extracted=graph_data.get('entities', []),
            relationships_created=graph_data.get('relationships', []),
            graph_summary=graph_summary
        )
        
        self.logger.info("Knowledge analysis completed")
        return result
    
    def close(self):
        """Close database connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            self.logger.info("Neo4j connection closed")

