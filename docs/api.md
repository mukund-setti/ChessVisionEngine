# API Reference

## REST API Endpoints

### Health Check
```
GET /health
```

Returns server health status.

**Response:**
```json
{
  "status": "healthy",
  "engine_available": true
}
```

---

### Scan Board Image
```
POST /api/scan
Content-Type: multipart/form-data
```

Upload a chessboard image and get the detected position.

**Request:**
- `file`: Image file (JPEG, PNG)

**Response:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "confidence": 0.95,
  "is_valid": true,
  "validation_errors": [],
  "board_ascii": "r n b q k b n r\np p p p p p p p\n. . . . . . . .\n..."
}
```

---

### Analyze Position
```
POST /api/analyze
Content-Type: application/json
```

Analyze a chess position with Stockfish.

**Request:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
  "depth": 20
}
```

**Response:**
```json
{
  "fen": "...",
  "best_move": "e7e5",
  "score": "+0.25",
  "pv": ["e7e5", "g1f3", "b8c6"],
  "is_valid": true,
  "validation_errors": []
}
```

---

### Validate FEN
```
GET /api/validate?fen=<FEN_STRING>
```

Validate a FEN string.

**Response:**
```json
{
  "fen": "...",
  "is_valid": true,
  "is_legal": true,
  "errors": [],
  "warnings": []
}
```

---

### Get Legal Moves
```
GET /api/legal-moves?fen=<FEN_STRING>
```

Get all legal moves for a position.

**Response:**
```json
{
  "fen": "...",
  "turn": "white",
  "is_check": false,
  "is_checkmate": false,
  "moves": [
    {
      "uci": "e2e4",
      "san": "e4",
      "from": "e2",
      "to": "e4",
      "is_capture": false,
      "is_check": false
    }
  ]
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Invalid request (bad image, invalid FEN) |
| 503 | Engine not available |
| 500 | Internal server error |

---

## Python API

### Board Detection
```python
from src.detection import BoardDetector, PieceClassifier

detector = BoardDetector()
board = detector.detect_board("chessboard.jpg")

classifier = PieceClassifier()
classification = classifier.classify_pieces(board)
```

### FEN Generation
```python
from src.chess_logic import FENGenerator, PositionValidator

generator = FENGenerator()
fen = generator.generate(classification)

validator = PositionValidator()
result = validator.validate(fen)
```

### Engine Analysis
```python
from src.engine import StockfishWrapper

with StockfishWrapper() as engine:
    result = engine.analyze(fen, depth=20)
    print(f"Best move: {result.best_move}")
    print(f"Evaluation: {result.score}")
```