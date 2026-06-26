# Ingestion Flow

## Pipeline

```text
Uploaded file
  -> Validate
  -> Detect file type
  -> Parse content
  -> Normalize text blocks
  -> Detect document structure
  -> Resolve knowledge branch
  -> Semantic chunk
  -> Extract chunk features
  -> Persist chunks
  -> Build keyword index
  -> Build vector index
```

## Supported File Strategy

Use a parser registry:

```text
PDF     -> PDF parser with page metadata
DOCX    -> Word parser with heading/table metadata
XLSX    -> Spreadsheet parser with sheet/table metadata
CSV     -> Table parser
TXT     -> Plain text parser
HTML    -> HTML cleaner and section parser
```

## Chunking Strategy

Preferred order:

1. Table of contents.
2. Headings.
3. Sections.
4. Paragraphs.
5. Token fallback.

Avoid fixed chunking as the default.

## Chunk Features

Each chunk should store:

```text
chunk_id
workspace_id
document_id
document_version_id
source_file_name
source_file_hash
page_number
sheet_name
section_title
heading_path
knowledge_branch_path
language
text
token_count
char_count
table_count
image_count
contains_policy_language
contains_product_spec
contains_procedure_steps
chunk_quality_score
embedding_model
chunking_strategy
chunk_version
created_at
```

## Quality Gates

Fail or warn when:

- Text extraction is empty.
- Document has too many scanned pages without OCR.
- Chunk is too short to be useful.
- Chunk is too long for reranking.
- File type is unsupported.
- Branch routing confidence is too low.

