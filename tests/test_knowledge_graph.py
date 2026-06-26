import os
import tempfile
import pytest
from rag_core.contracts.types import KGEntity, KGRelation
from rag_core.adapters.knowledge_graph import NetworkXKnowledgeGraph


def test_knowledge_graph_basic_operations():
    # 1. Initialize empty graph
    kg = NetworkXKnowledgeGraph()
    assert kg.node_count == 0
    assert kg.edge_count == 0

    # 2. Add entities
    entities = [
        KGEntity(entity_id="1", name="Phượng Hải", entity_type="company", properties={"field": "technology"}),
        KGEntity(entity_id="2", name="Nguyễn Văn A", entity_type="person", properties={"role": "CEO"}),
        KGEntity(entity_id="3", name="Phòng Kỹ Thuật", entity_type="department", properties={}),
    ]
    kg.add_entities(entities)
    # The nodes might be created with synthetic IDs
    assert kg.node_count == 3

    # 3. Add relations
    relations = [
        KGRelation(source="Nguyễn Văn A", target="Phượng Hải", relation_type="works_at"),
        KGRelation(source="Nguyễn Văn A", target="Phòng Kỹ Thuật", relation_type="manages"),
    ]
    kg.add_relations(relations)
    assert kg.edge_count == 2


def test_knowledge_graph_query():
    kg = NetworkXKnowledgeGraph()
    entities = [
        KGEntity(entity_id="1", name="Phượng Hải", entity_type="company"),
        KGEntity(entity_id="2", name="Nguyễn Văn A", entity_type="person"),
        KGEntity(entity_id="3", name="SmartSinks", entity_type="product"),
    ]
    relations = [
        KGRelation(source="Nguyễn Văn A", target="Phượng Hải", relation_type="works_at"),
        KGRelation(source="Phượng Hải", target="SmartSinks", relation_type="manufactures"),
    ]
    kg.add_entities(entities)
    kg.add_relations(relations)

    # Query with entity name in the question
    result = kg.query("Ai là người làm việc tại Phượng Hải?", workspace_id="ws-1")
    assert len(result.entities) > 0
    assert any(e.name == "Phượng Hải" for e in result.entities)
    assert len(result.relations) > 0
    assert any(r.relation_type == "works_at" for r in result.relations)
    assert "Phượng Hải → [manufactures] → SmartSinks" in result.natural_language_answer


def test_knowledge_graph_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        kg = NetworkXKnowledgeGraph(persist_path=tmpdir)
        entities = [
            KGEntity(entity_id="1", name="Phượng Hải", entity_type="company"),
            KGEntity(entity_id="2", name="Nguyễn Văn A", entity_type="person"),
        ]
        relations = [
            KGRelation(source="Nguyễn Văn A", target="Phượng Hải", relation_type="works_at"),
        ]
        kg.add_entities(entities)
        kg.add_relations(relations)
        
        # Persist to disk
        kg.persist()
        
        # Load from disk in a new instance
        kg_loaded = NetworkXKnowledgeGraph()
        kg_file = os.path.join(tmpdir, "knowledge_graph.json")
        assert os.path.exists(kg_file)
        
        kg_loaded.load(kg_file)
        assert kg_loaded.node_count == 2
        assert kg_loaded.edge_count == 1
        
        # Check querying still works on loaded graph
        result = kg_loaded.query("Thông tin về Phượng Hải", workspace_id="ws-1")
        assert any(e.name == "Phượng Hải" for e in result.entities)
