# Architecture

## System Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                       Chess Vision Engine                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    Image     │───▶│    Board     │───▶│    Piece     │       │
│  │    Input     │    │   Detector   │    │  Classifier  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                  │               │
│                                                  ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Stockfish  │◀───│     FEN      │◀───│    Board     │       │
│  │    Engine    │    │   Generator  │    │Classification│       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │   Analysis   │───▶│   Digital    │                           │
│  │    Result    │    │    Board     │                           │
│  └──────────────┘    └──────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Detection Module (`src/detection/`)

#### BoardDetector
- Locates and extracts chessboard from images
- Uses Hough line detection or contour detection
- Applies perspective transformation

#### PieceClassifier
- Identifies chess pieces in each square
- CNN model (ResNet18 backbone)
- 13 classes: empty + 6 pieces × 2 colors

### 2. Chess Logic Module (`src/chess_logic/`)

#### FENGenerator
- Converts board classification to FEN notation
- Infers castling rights from piece positions

#### PositionValidator
- Validates chess positions
- Checks king count, pawn placement, legality

#### BoardState
- Represents and manipulates chess positions
- Legal move generation

### 3. Engine Module (`src/engine/`)

#### StockfishWrapper
- Interfaces with Stockfish engine
- Position analysis, best move calculation

#### PositionAnalyzer
- Comprehensive position analysis
- Threat detection, weakness identification

### 4. UI Module (`src/ui/`)

#### FastAPI Application
- REST API endpoints
- Web interface with drag-and-drop

## Technology Stack

- **Python 3.10+**: Core language
- **FastAPI**: Web framework
- **python-chess**: Chess logic
- **OpenCV**: Computer vision
- **PyTorch**: Deep learning
- **Stockfish**: Chess engine

## Data Flow

1. **Image Upload** → User uploads chessboard photo
2. **Board Detection** → Detect edges, perspective transform, extract 64 squares
3. **Piece Classification** → CNN classifies each square
4. **FEN Generation** → Convert to standard notation
5. **Engine Analysis** → Stockfish evaluates position
6. **Result Display** → Show digital board + best moves

## Model Architecture
```
Input: [B, 3, 64, 64]
    │
    ▼
ResNet18 Backbone
    │
    ▼
Feature Vector: [B, 512]
    │
    ▼
Dropout (0.5) → Linear (512→256) → ReLU → Dropout (0.3) → Linear (256→13)
    │
    ▼
Output: [B, 13] class logits
```
```

---

### File 32: `models/.gitkeep`
```
# Trained models directory
# Model files are git-ignored due to size
```

---

### File 33: `data/.gitkeep`
```
# Training data directory
```

---

### File 34: `data/raw/.gitkeep`
```
# Raw training images
```

---

### File 35: `data/processed/.gitkeep`
```
# Processed training data
```

---

### File 36: `notebooks/.gitkeep`
```
# Jupyter notebooks for experimentation
```

---

### File 37: `LICENSE`
```
MIT License

Copyright (c) 2024 Mukund Setti

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.