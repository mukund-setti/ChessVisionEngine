# Chess Vision Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Scan a photo of a chessboard, reconstruct the position, and analyze it with a chess engine.

## Features

- **Board Detection**: Automatically detect and extract chessboard from photos
- **Piece Recognition**: Identify chess pieces using computer vision / ML
- **Position Reconstruction**: Convert detected pieces to FEN notation
- **Digital Board Display**: Render the position on an interactive digital board
- **Engine Analysis**: Run Stockfish analysis on the detected position
- **Move Suggestions**: Get best move recommendations and evaluation

## Quick Start

### Prerequisites

- Python 3.10+
- Stockfish chess engine

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mukund-setti/ChessVisionEngine.git
cd ChessVisionEngine
```

2. Create a virtual environment:
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download Stockfish:
```bash
python scripts/download_stockfish.py
```

5. Copy environment file and configure:
```bash
copy .env.example .env
# Edit .env with your Stockfish path
```

6. Run the application:
```bash
python -m src.ui.app
```

7. Open http://localhost:8080 in your browser

### Using Docker
```bash
docker-compose up --build
```

## Usage

### Web Interface

1. Start the server: `python -m src.ui.app`
2. Open http://localhost:8080
3. Upload or drag-and-drop a chessboard image
4. View the detected position and engine analysis

### Command Line
```bash
# Analyze a single image
python -m src.main analyze --image path/to/chessboard.jpg

# Analyze with custom depth
python -m src.main analyze --image board.jpg --depth 25

# Start the web server
python -m src.main serve --port 8080

# Analyze a FEN position directly
python -m src.main engine "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

# Live webcam analysis
python -m src.main live --camera 0
```

### Python API
```python
from src.detection import BoardDetector, PieceClassifier
from src.chess_logic import FENGenerator
from src.engine import StockfishWrapper

# Load and process image
detector = BoardDetector()
classifier = PieceClassifier()

board_image = detector.detect_board("chessboard.jpg")
pieces = classifier.classify_pieces(board_image)

# Generate FEN
fen_generator = FENGenerator()
fen = fen_generator.generate(pieces)

# Analyze with engine
engine = StockfishWrapper()
analysis = engine.analyze(fen, depth=20)
print(f"Best move: {analysis.best_move}")
print(f"Evaluation: {analysis.score}")
```

## Project Structure
```
ChessVisionEngine/
├── src/
│   ├── detection/          # Board and piece detection
│   │   ├── board_detector.py
│   │   ├── piece_classifier.py
│   │   └── image_processor.py
│   ├── chess_logic/        # Chess rules and FEN generation
│   │   ├── fen_generator.py
│   │   ├── position_validator.py
│   │   └── board_state.py
│   ├── engine/             # Chess engine integration
│   │   ├── stockfish_wrapper.py
│   │   └── analysis.py
│   ├── ui/                 # Web interface
│   │   └── app.py
│   ├── utils/              # Shared utilities
│   │   ├── config.py
│   │   └── logging_config.py
│   └── main.py             # CLI entry point
├── models/                 # Trained ML models
├── data/                   # Training/test data
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── docs/                   # Documentation
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

## How It Works

### 1. Board Detection
The system uses computer vision techniques to:
- Detect the chessboard edges using Hough line detection
- Apply perspective transformation to get a top-down view
- Segment the board into 64 individual squares

### 2. Piece Recognition
Each square is classified using a CNN model trained on chess piece images:
- Empty square
- White/Black: King, Queen, Rook, Bishop, Knight, Pawn

### 3. Position Reconstruction
The detected pieces are converted to FEN (Forsyth-Edwards Notation):
- Standard chess position notation
- Includes piece positions, turn, castling rights, en passant

### 4. Engine Analysis
Stockfish analyzes the position and provides:
- Best move recommendation
- Position evaluation (centipawns)
- Principal variation (best line)
- Mate detection

## Configuration

Copy `.env.example` to `.env` and configure:
```env
STOCKFISH_PATH=C:\path\to\stockfish.exe
ENGINE_DEPTH=20
ENGINE_THREADS=4
MODEL_PATH=models/piece_classifier.onnx
LOG_LEVEL=INFO
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check |
| `/api/scan` | POST | Scan board image |
| `/api/analyze` | POST | Analyze FEN position |
| `/api/validate` | GET | Validate FEN string |
| `/api/legal-moves` | GET | Get legal moves |

See [docs/api.md](docs/api.md) for detailed API documentation.

## Training Your Own Model

1. Collect training data (images of chess pieces)

2. Organize data:
```
data/
├── train/
│   ├── empty/
│   ├── white_king/
│   ├── white_queen/
│   └── ...
└── val/
    ├── empty/
    └── ...
```

3. Train the model:
```bash
python scripts/train_model.py --data data --output models --epochs 50
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Roadmap

- [x] Basic board detection
- [x] Piece classification framework
- [x] FEN generation
- [x] Stockfish integration
- [x] Web UI
- [ ] Pre-trained model
- [ ] Mobile app (React Native)
- [ ] Real-time video analysis
- [ ] Support for different board styles
- [ ] PGN export
- [ ] Opening book integration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Stockfish](https://stockfishchess.org/) - Open source chess engine
- [python-chess](https://python-chess.readthedocs.io/) - Chess library for Python
- [OpenCV](https://opencv.org/) - Computer vision library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework